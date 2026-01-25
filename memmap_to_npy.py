#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert embedding memmap file to a standard .npy array,
while strictly aligning rows by sequence IDs from a feather file.

Typical use case:
- embeddings.memmap.float16 + ids.npy + meta.json are produced from FASTA
- downstream model expects train_embeds.npy aligned to train_seq.feather order
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert embedding memmap to .npy with ID alignment"
    )
    parser.add_argument(
        "--embed-dir",
        type=str,
        required=True,
        help="Directory containing embeddings.memmap.float16, ids.npy, meta.json",
    )
    parser.add_argument(
        "--feather",
        type=str,
        required=True,
        help="Feather file defining the desired sequence order (e.g. train_seq.feather)",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="Column name in feather file containing sequence IDs (default: id)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for aligned .npy file (e.g. train_embeds.npy)",
    )
    parser.add_argument(
        "--out-dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Output dtype (default: float32, recommended for GBDT)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    embed_dir = args.embed_dir
    feather_path = args.feather
    out_path = args.out
    out_dtype = np.float32 if args.out_dtype == "float32" else np.float16

    # --------------------------------------------------
    # 1. Load metadata
    # --------------------------------------------------
    meta_path = os.path.join(embed_dir, "meta.json")
    ids_path = os.path.join(embed_dir, "ids.npy")
    memmap_path = os.path.join(embed_dir, "embeddings.memmap.float16")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"ids.npy not found: {ids_path}")
    if not os.path.exists(memmap_path):
        raise FileNotFoundError(f"embeddings.memmap.float16 not found: {memmap_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_total = int(meta["total"])
    d_model = int(meta["d_model"])
    in_dtype = np.float16 if meta.get("dtype", "float16") == "float16" else np.float32

    print(f"[INFO] memmap shape = ({n_total}, {d_model}), dtype={in_dtype}")

    # --------------------------------------------------
    # 2. Open memmap and load FASTA-order IDs
    # --------------------------------------------------
    emb_mm = np.memmap(
        memmap_path,
        dtype=in_dtype,
        mode="r",
        shape=(n_total, d_model),
    )

    ids = np.load(ids_path, allow_pickle=True)
    id2row = {str(pid): i for i, pid in enumerate(ids)}

    # --------------------------------------------------
    # 3. Load feather file (defines target order)
    # --------------------------------------------------
    df = pd.read_feather(feather_path)

    if args.id_col not in df.columns:
        raise KeyError(
            f"ID column '{args.id_col}' not found in feather file. "
            f"Available columns: {list(df.columns)}"
        )

    seq_ids = df[args.id_col].astype(str).values
    print(f"[INFO] sequences in feather = {len(seq_ids)}")

    # --------------------------------------------------
    # 4. Reorder embeddings according to feather order
    # --------------------------------------------------
    out = np.empty((len(seq_ids), d_model), dtype=out_dtype)

    missing = 0
    for i, sid in enumerate(seq_ids):
        row = id2row.get(sid)
        if row is None:
            # This should not normally happen; fill zeros for safety
            out[i] = 0.0
            missing += 1
        else:
            out[i] = emb_mm[row].astype(out_dtype, copy=False)

    if missing > 0:
        print(f"[WARN] {missing} sequence IDs not found in embeddings (filled with zeros)")

    # --------------------------------------------------
    # 5. Save aligned .npy file
    # --------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, out)

    print(f"[DONE] saved aligned embeddings to: {out_path}")
    print(f"[DONE] shape={out.shape}, dtype={out.dtype}")


if __name__ == "__main__":
    main()
