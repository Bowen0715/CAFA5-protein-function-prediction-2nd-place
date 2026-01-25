#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser("Create *_ids.npy from *_seq.feather (EntryID order)")
    ap.add_argument("--feather", type=str, required=True, help="Path to train_seq.feather / test_seq.feather")
    ap.add_argument("--id-col", type=str, default="EntryID", help="ID column in feather (default: EntryID)")
    ap.add_argument("--out", type=str, required=True, help="Output path, e.g. train_ids.npy / test_ids.npy")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_feather(args.feather)

    if args.id_col not in df.columns:
        raise KeyError(f"ID column '{args.id_col}' not found. Columns={list(df.columns)}")

    ids = df[args.id_col].astype(str).values  # (N,) strings

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, ids)
    print(f"[DONE] saved ids to: {args.out}")
    print(f"[DONE] shape={ids.shape}, dtype={ids.dtype}")
    print("[DONE] first 5:", ids[:5])


if __name__ == "__main__":
    main()
