#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from typing import List

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


def preprocess(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq.strip().upper())
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq)
    return seq


@torch.no_grad()
def embed_batch(
    tokenizer: AutoTokenizer,
    model: EsmModel,
    device: torch.device,
    seqs: List[str],
    max_len: int = None,
) -> torch.Tensor:
    """
    Embed a batch of protein sequences (protein-level embeddings).
    Returns: (B, d_model) tensor on GPU.
    """
    seqs = [preprocess(s) for s in seqs]

    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=(max_len is not None),
        max_length=max_len,
        return_special_tokens_mask=True,
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    h = out.last_hidden_state  # (B, L, d_model)

    attn = enc["attention_mask"].bool()          # (B, L)
    special = enc["special_tokens_mask"].bool()  # (B, L)
    keep = attn & (~special)                     # (B, L)

    keep_f = keep.unsqueeze(-1).type_as(h)
    summed = (h * keep_f).sum(dim=1)
    denom = keep_f.sum(dim=1).clamp_min(1.0)
    emb = summed / denom                         # (B, d_model)
    return emb


def split_windows(seq: str, window: int, stride: int) -> List[str]:
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
    tokenizer: AutoTokenizer,
    model: EsmModel,
    device: torch.device,
    seq: str,
    window: int,
    stride: int,
    chunk_batch_size: int,
) -> torch.Tensor:
    """
    For one long protein: sliding-window embedding and average across windows.
    Returns: (d_model,) on CPU float16.
    """
    seq = preprocess(seq)
    chunks = split_windows(seq, window=window, stride=stride)

    chunk_embs = []
    for i in range(0, len(chunks), chunk_batch_size):
        sub = chunks[i:i + chunk_batch_size]
        e = embed_batch(tokenizer, model, device, sub, max_len=window)  # (b, d_model)
        chunk_embs.append(e)
    e_all = torch.cat(chunk_embs, dim=0)  # (n_chunks, d_model)
    e_mean = e_all.mean(dim=0)
    return e_mean.detach().cpu().half()


def count_fasta_records(fasta_path: str) -> int:
    n = 0
    for _ in SeqIO.parse(fasta_path, "fasta"):
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    # esm2_t36_3B_UR50D
    ap.add_argument("--model", type=str, default="esm2_t36_3B_UR50D")

    ap.add_argument("--batch_size", type=int, default=16, help="short proteins batch size")
    ap.add_argument("--window", type=int, default=1022, help="window length for long proteins")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--chunk_batch_size", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--local_only", action="store_true", help="only load from local cache/path")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_only,
        do_lower_case=False,
    )
    model = EsmModel.from_pretrained(
        args.model,
        local_files_only=args.local_only,
    ).to(device)

    model.eval()
    if device.type == "cuda" and args.fp16:
        model = model.half()

    d_model = int(model.config.hidden_size)
    print(f"[INFO] model hidden_size = {d_model}")

    print("[INFO] counting FASTA records ...")
    n_total = count_fasta_records(args.fasta)
    print(f"[INFO] total proteins = {n_total}")

    emb_path = os.path.join(args.outdir, "embeddings.memmap.float16")
    ids_path = os.path.join(args.outdir, "ids.npy")
    meta_path = os.path.join(args.outdir, "meta.json")

    emb_mm = np.memmap(emb_path, dtype=np.float16, mode="w+", shape=(n_total, d_model))
    ids = np.empty((n_total,), dtype=object)

    records = SeqIO.parse(args.fasta, "fasta")
    pbar = tqdm(total=n_total, desc="Embedding proteins", unit="prot")

    i = 0
    short_batch_ids, short_batch_seqs, short_batch_indices = [], [], []

    def flush_short_batch():
        nonlocal short_batch_ids, short_batch_seqs, short_batch_indices, emb_mm, ids
        if not short_batch_seqs:
            return
        e = embed_batch(tokenizer, model, device, short_batch_seqs, max_len=args.window)  # (B, d)
        e = e.detach().cpu().half().numpy()
        for k, idx in enumerate(short_batch_indices):
            emb_mm[idx] = e[k]
            ids[idx] = short_batch_ids[k]
        short_batch_ids, short_batch_seqs, short_batch_indices = [], [], []

    last_saved = 0

    for rec in records:
        seq_id = rec.id
        seq = str(rec.seq)
        L = len(seq)

        ids[i] = seq_id

        if L <= args.window:
            short_batch_ids.append(seq_id)
            short_batch_seqs.append(seq)
            short_batch_indices.append(i)
            if len(short_batch_seqs) >= args.batch_size:
                flush_short_batch()
        else:
            flush_short_batch()
            emb = embed_one_with_windows(
                tokenizer, model, device,
                seq=seq, window=args.window, stride=args.stride,
                chunk_batch_size=args.chunk_batch_size
            )
            emb_mm[i] = emb.numpy()
            ids[i] = seq_id

        i += 1
        pbar.update(1)

        if (i - last_saved) >= args.save_every:
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
                        "written": i,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            last_saved = i
            pbar.set_postfix({"saved": i})

    flush_short_batch()
    pbar.close()

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
                "written": n_total,
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


if __name__ == "__main__":
    main()
