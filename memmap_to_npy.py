#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Convert embeddings.memmap.float16 to _embeds.npy")
    p.add_argument("--memmap", type=str, required=True,
                   help="Path to embeddings.memmap.float16")
    p.add_argument("--ids-npy", type=str, default=None,
                   help="Path to ids.npy (recommended). If not given, will try memmap-dir/ids.npy")
    p.add_argument("--meta-json", type=str, default=None,
                   help="Path to meta.json (optional). If not given, will try memmap-dir/meta.json")
    p.add_argument("--d-model", type=int, default=None,
                   help="Embedding dimension if meta.json missing (e.g., 1280 for esm2_t33_650M)")
    p.add_argument("--out", type=str, required=True,
                   help="Output embeds.npy path")
    p.add_argument("--out-dtype", type=str, default="float32",
                   choices=["float16", "float32"],
                   help="Output dtype (default float32)")

    # Optional alignment to feather order (like your earlier need)
    p.add_argument("--feather", type=str, default=None,
                   help="Optional feather to reorder rows by ID")
    p.add_argument("--id-col", type=str, default="id",
                   help="ID column in feather (default: id)")
    p.add_argument("--missing", type=str, default="zero",
                   choices=["zero", "error"],
                   help="If feather has missing IDs: zero-fill or error")

    return p.parse_args()


def main():
    args = parse_args()
    out_dtype = np.float32 if args.out_dtype == "float32" else np.float16

    memmap_path = args.memmap
    embed_dir = os.path.dirname(memmap_path)

    ids_path = args.ids_npy or os.path.join(embed_dir, "ids.npy")
    meta_path = args.meta_json or os.path.join(embed_dir, "meta.json")

    if not os.path.exists(memmap_path):
        raise FileNotFoundError(memmap_path)
    if not os.path.exists(ids_path):
        raise FileNotFoundError(
            f"ids.npy not found: {ids_path}\n"
            f"请提供 --ids-npy，或把 ids.npy 放到 memmap 同目录。"
        )

    ids = np.load(ids_path, allow_pickle=True).astype(str)
    N = len(ids)

    # 1) decide d_model + in_dtype
    in_dtype = np.float16  # your file name implies float16
    d_model = None

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        d_model = int(meta["d_model"])
        meta_total = int(meta["total"])
        meta_dtype = meta.get("dtype", "float16")
        in_dtype = np.float16 if meta_dtype == "float16" else np.float32

        if meta_total != N:
            print(f"[WARN] meta.total({meta_total}) != len(ids)({N}). Will trust ids.npy for N.")
    else:
        if args.d_model is None:
            raise FileNotFoundError(
                f"meta.json not found at {meta_path}, and --d-model not provided.\n"
                f"请加上 --d-model 1280(esm2_650M) 或 1024/2560 等你实际维度。"
            )
        d_model = int(args.d_model)

    # 2) sanity check file size
    itemsize = np.dtype(in_dtype).itemsize
    expected_bytes = N * d_model * itemsize
    actual_bytes = os.path.getsize(memmap_path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"memmap 文件大小不匹配，说明 N 或 d_model 不对。\n"
            f"expected bytes = {expected_bytes} (N={N}, d_model={d_model}, dtype={in_dtype}, itemsize={itemsize})\n"
            f"actual bytes   = {actual_bytes}\n"
            f"解决：确认 ids.npy 是否对应这个 memmap；确认 d_model；或检查是否是别的 dtype。"
        )

    print(f"[INFO] memmap: {memmap_path}")
    print(f"[INFO] ids:    {ids_path} (N={N})")
    print(f"[INFO] dtype:  {in_dtype}, d_model={d_model}")

    emb_mm = np.memmap(memmap_path, dtype=in_dtype, mode="r", shape=(N, d_model))

    # 3) if no alignment requested: dump directly
    if args.feather is None:
        out = np.asarray(emb_mm, dtype=out_dtype)  # materialize to normal ndarray
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.save(args.out, out)
        print(f"[DONE] saved embeds.npy: {args.out} shape={out.shape} dtype={out.dtype}")
        return

    # 4) alignment to feather order
    feather_path = args.feather
    df = pd.read_feather(feather_path)
    if args.id_col not in df.columns:
        raise KeyError(f"'{args.id_col}' not in feather columns: {list(df.columns)}")

    target_ids = df[args.id_col].astype(str).values
    id2row = {pid: i for i, pid in enumerate(ids)}

    out = np.empty((len(target_ids), d_model), dtype=out_dtype)
    missing = 0
    for i, sid in enumerate(target_ids):
        row = id2row.get(sid)
        if row is None:
            missing += 1
            if args.missing == "error":
                raise KeyError(f"ID not found in ids.npy: {sid}")
            out[i] = 0
        else:
            out[i] = emb_mm[row].astype(out_dtype, copy=False)

    if missing:
        print(f"[WARN] missing IDs = {missing} (handled by {args.missing})")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, out)
    print(f"[DONE] saved aligned embeds.npy: {args.out} shape={out.shape} dtype={out.dtype}")


if __name__ == "__main__":
    main()
