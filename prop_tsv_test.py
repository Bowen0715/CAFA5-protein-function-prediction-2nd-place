import argparse
import os
import sys
import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True)
parser.add_argument('-g', '--graph', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)

parser.add_argument('-d', '--device', type=str, default="0")
parser.add_argument('-b', '--batch_size', type=int, default=30000)
parser.add_argument('-bi', '--batch_inner', type=int, default=5000)


def propagate_max(mat, G):
    indexer = cp.arange(mat.shape[0], dtype=cp.int32)

    for f in G.order:
        adj = G.terms_list[f]['children']
        if len(adj) == 0:
            continue
        adj = cp.asarray(adj, dtype=cp.int32)
        # 如果你的 kernel 需要 int64，就把上面 dtype 改回 int64，但建议统一 int32
        propagate_col_kernel(indexer, adj, cp.int32(f), cp.int32(adj.shape[0]), cp.int32(mat.shape[1]), mat.ravel())


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    import cupy as cp
    import numpy as np
    import pandas as pd

    from protlib.metric import get_funcs_mapper, get_ns_id, obo_parser, Graph, propagate_col_kernel

    # ========= 1) 用 pandas 读 TSV（CPU，稳定） =========
    trainTerms = pd.read_csv(args.path, sep='\t', usecols=['EntryID', 'term'], dtype={'EntryID': 'string', 'term': 'string'})

    # ========= 2) 建图（和你原来一样） =========
    ontologies = []
    for ns, terms_dict in obo_parser(args.graph).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    # ========= 3) 建 id 映射（CPU） =========
    back_prot_id = pd.Index(trainTerms['EntryID'].drop_duplicates())
    length = len(back_prot_id)

    prot_id = pd.Series(np.arange(length, dtype=np.int32), index=back_prot_id)
    trainTerms['id'] = trainTerms['EntryID'].map(prot_id).astype(np.int32)

    flg = True

    # 让 trainTerms 按 id 排序，后面切 batch 更快更稳
    trainTerms = trainTerms.sort_values('id', kind='mergesort').reset_index(drop=True)
    ids_host = trainTerms['id'].to_numpy()  # numpy int32

    for i in tqdm.tqdm(range(0, length, args.batch_size)):
        lo = i
        hi = min(i + args.batch_size, length)

        # ========= 4) CPU 上用 numpy searchsorted 找范围（避免 cudf query） =========
        start = int(np.searchsorted(ids_host, lo, side='left'))
        end   = int(np.searchsorted(ids_host, hi, side='left'))

        sample = trainTerms.iloc[start:end]   # pandas slice（view）
        batch_len = hi - lo

        for G in ontologies:
            # term -> term_id 映射（CPU dict）
            mapper = get_funcs_mapper(G)  # {term_str: term_id}
            sample_term_id = sample['term'].map(mapper)  # pandas Series float/Int + NaN

            # 只保留能映射到 term_id 的行
            ok = sample_term_id.notna().to_numpy()
            if not ok.any():
                continue

            # 取坐标：row_id (batch内) + term_id
            # sample['id'] 是全局 id（lo..hi-1），变成 batch 内行号
            rid_host = (sample.loc[ok, 'id'].to_numpy(dtype=np.int32) - np.int32(lo))
            cid_host = sample_term_id.loc[ok].to_numpy(dtype=np.int32)

            # ========= 5) GPU 上建 mat + scatter_add =========
            rid = cp.asarray(rid_host, dtype=cp.int32)
            cid = cp.asarray(cid_host, dtype=cp.int32)

            mat = cp.zeros((batch_len, G.idxs), dtype=cp.float32)
            mat.scatter_add((rid, cid), 1)
            mat = cp.clip(mat, 0, 1)

            propagate_max(mat, G)

            # 反向映射：term_id -> term_str（CPU list，避免 cudf 索引）
            rev_terms = get_funcs_mapper(G, False)  # list/array by index

            for j in range(0, mat.shape[0], args.batch_inner):
                row, col = cp.nonzero(mat[j: j + args.batch_inner])

                if row.size == 0:
                    continue

                # row/col 拉回 CPU 做输出拼表（小块，开销可接受）
                row_h = cp.asnumpy(row).astype(np.int32)
                col_h = cp.asnumpy(col).astype(np.int32)

                entry_ids = back_prot_id[(lo + j + row_h)].to_numpy()
                terms_out = np.asarray(rev_terms, dtype=object)[col_h]

                ns_id, ns_str = get_ns_id(G)
                asp = ns_str.upper() + 'O'

                out_df = pd.DataFrame({
                    'EntryID': entry_ids,
                    'term': terms_out,
                    'aspect': asp
                }).sort_values(['EntryID', 'term'], ascending=True)

                out_df.to_csv(args.output, sep='\t', index=False, mode=('w' if flg else 'a'), header=flg)
                flg = False
