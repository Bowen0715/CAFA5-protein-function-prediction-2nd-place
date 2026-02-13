#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from typing import List, Tuple

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel


def preprocess(seq: str) -> str:
    """Replace rare AA letters and space-separate tokens for ProtT5."""
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


@torch.no_grad()
def embed_batch(
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    seqs: List[str],
    max_len: int = None,
) -> torch.Tensor:
    """
    Embed a batch of protein sequences (protein-level embeddings).
    Returns: (B, d_model) tensor on GPU.
    """
    seqs_spaced = [preprocess(s) for s in seqs]

    enc = tokenizer(
        seqs_spaced,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        truncation=(max_len is not None),
        max_length=max_len,
        return_special_tokens_mask=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    h = out.last_hidden_state  # (B, L, d_model)

    attn = enc["attention_mask"].bool()          # (B, L)
    special = enc["special_tokens_mask"].bool()  # (B, L)
    keep = attn & (~special)                     # (B, L)

    # masked mean pooling => (B, d_model)
    keep_f = keep.unsqueeze(-1).type_as(h)
    summed = (h * keep_f).sum(dim=1)
    denom = keep_f.sum(dim=1).clamp_min(1.0)
    emb = summed / denom
    return emb


def split_windows(seq: str, window: int, stride: int) -> List[str]:
    """
    Split sequence into overlapping windows for long proteins.
    Returns list of substrings (non-spaced).
    """
    L = len(seq)
    if L <= window:
        return [seq]
    out = []
    for start in range(0, L, stride):
        end = start + window
        out.append(seq[start:end])
        if end >= L:
            break
    return out


@torch.no_grad()
def embed_one_with_windows(
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    seq: str,
    window: int,
    stride: int,
    chunk_batch_size: int,
) -> torch.Tensor:
    """
    For one protein, do sliding-window embedding and average across windows.
    Returns: (d_model,) on CPU (float16).
    """
    chunks = split_windows(seq, window=window, stride=stride)

    chunk_embs = []
    for i in range(0, len(chunks), chunk_batch_size):
        sub = chunks[i:i + chunk_batch_size]
        e = embed_batch(tokenizer, model, device, sub, max_len=window)  # (b, d_model)
        chunk_embs.append(e)
    e_all = torch.cat(chunk_embs, dim=0)  # (n_chunks, d_model)
    e_mean = e_all.mean(dim=0)            # (d_model,)
    return e_mean.detach().cpu().half()


def count_fasta_records(fasta_path: str) -> int:
    n = 0
    for _ in SeqIO.parse(fasta_path, "fasta"):
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=str, required=True, help="Input FASTA path")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument("--model", type=str, default="Rostlab/prot_t5_xl_half_uniref50-enc")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for short proteins (<=window)")
    ap.add_argument("--window", type=int, default=1024, help="Window length for long proteins")
    ap.add_argument("--stride", type=int, default=512, help="Stride for sliding windows")
    ap.add_argument("--chunk_batch_size", type=int, default=16, help="Batch size when embedding chunks of one long protein")
    ap.add_argument("--save_every", type=int, default=2000, help="Flush to disk every N proteins")
    ap.add_argument("--fp16", action="store_true", help="Force fp16 on GPU (recommended for this model)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # Load tokenizer/model (will download into model_dir if not present)
    tokenizer = T5Tokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        do_lower_case=False,
    )
    model = T5EncoderModel.from_pretrained(
        args.model,
        local_files_only=True,
    ).to(device)

    model.eval()
    if device.type == "cuda" and args.fp16:
        model = model.half()

    d_model = int(model.config.d_model)
    print(f"[INFO] model d_model = {d_model}")

    # Count proteins (needed for memmap sizing)
    print("[INFO] counting FASTA records ...")
    n_total = count_fasta_records(args.fasta)
    print(f"[INFO] total proteins = {n_total}")

    # Prepare output files
    emb_path = os.path.join(args.outdir, "embeddings.memmap.float16")
    ids_path = os.path.join(args.outdir, "ids.npy")
    meta_path = os.path.join(args.outdir, "meta.json")

    # Memmap: (N, d_model) float16
    emb_mm = np.memmap(emb_path, dtype=np.float16, mode="w+", shape=(n_total, d_model))
    ids = np.empty((n_total,), dtype=object)

    # Iterate FASTA with progress bar
    records = SeqIO.parse(args.fasta, "fasta")
    pbar = tqdm(total=n_total, desc="Embedding proteins", unit="prot")

    i = 0
    short_batch_ids: List[str] = []
    short_batch_seqs: List[str] = []
    short_batch_indices: List[int] = []

    def flush_short_batch():
        nonlocal short_batch_ids, short_batch_seqs, short_batch_indices, emb_mm, ids
        if not short_batch_seqs:
            return
        # embed short batch (truncate at window)
        e = embed_batch(tokenizer, model, device, short_batch_seqs, max_len=args.window)  # (B, d_model)
        e = e.detach().cpu().half().numpy()
        for k, idx in enumerate(short_batch_indices):
            emb_mm[idx] = e[k]
            ids[idx] = short_batch_ids[k]
        short_batch_ids = []
        short_batch_seqs = []
        short_batch_indices = []

    last_saved = 0

    for rec in records:
        seq_id = rec.id
        seq = str(rec.seq)
        L = len(seq)

        ids[i] = seq_id  # will overwrite in flush too, but safe here

        if L <= args.window:
            short_batch_ids.append(seq_id)
            short_batch_seqs.append(seq)
            short_batch_indices.append(i)
            if len(short_batch_seqs) >= args.batch_size:
                flush_short_batch()
        else:
            # long protein: flush any pending short batch first to keep ordering stable
            flush_short_batch()
            emb = embed_one_with_windows(
                tokenizer, model, device,
                seq=seq,
                window=args.window,
                stride=args.stride,
                chunk_batch_size=args.chunk_batch_size,
            )
            emb_mm[i] = emb.numpy()
            ids[i] = seq_id

        i += 1
        pbar.update(1)

        # periodic saving / flushing
        if (i - last_saved) >= args.save_every:
            emb_mm.flush()
            # save ids incrementally (overwrite full file, but cheap at 82k)
            np.save(ids_path, ids)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fasta": args.fasta,
                        "model": args.model,
                        "d_model": d_model,
                        "total": n_total,
                        "window": args.window,
                        "stride": args.stride,
                        "batch_size": args.batch_size,
                        "chunk_batch_size": args.chunk_batch_size,
                        "dtype": "float16",
                        "embeddings_file": emb_path,
                        "ids_file": ids_path,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            last_saved = i
            pbar.set_postfix({"saved": i})

    # flush remaining short batch
    flush_short_batch()
    pbar.close()

    # final flush
    emb_mm.flush()
    np.save(ids_path, ids)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fasta": args.fasta,
                "model": args.model,
                "d_model": d_model,
                "total": n_total,
                "window": args.window,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "chunk_batch_size": args.chunk_batch_size,
                "dtype": "float16",
                "embeddings_file": emb_path,
                "ids_file": ids_path,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[DONE]")
    print(f"IDs saved to: {ids_path}")
    print(f"Embeddings memmap saved to: {emb_path}")
    print(f"Meta saved to: {meta_path}")
    print(f"Shape: ({n_total}, {d_model}), dtype=float16")

# # How to read back
# import numpy as np
# emb = np.memmap("/root/autodl-tmp/T5_embeds/embeddings.memmap.float16",
#                 dtype=np.float16, mode="r", shape=(82404, 1024))
# ids = np.load("/root/autodl-tmp/T5_embeds/ids.npy", allow_pickle=True)
# print(ids[0], emb[0].shape)


if __name__ == "__main__":
    main()
