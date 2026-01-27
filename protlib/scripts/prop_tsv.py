import argparse
import os
import sys

import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str)
parser.add_argument('-g', '--graph', type=str)
parser.add_argument('-o', '--output', type=str)

parser.add_argument('-d', '--device', type=str, default="1")
parser.add_argument('-b', '--batch_size', type=int, default=30000)
parser.add_argument('-bi', '--batch_inner', type=int, default=5000)


def propagate_max(mat, G):
    indexer = cp.arange(mat.shape[0], dtype=cp.int64)

    for f in G.order:

        adj = G.terms_list[f]['children']

        if len(adj) == 0:
            continue

        adj = cp.asarray(adj, dtype=cp.int64)
        propagate_col_kernel(indexer, adj, cp.int64(f), cp.int64(adj.shape[0]), 
                             cp.int64(mat.shape[1]), mat.ravel())

    return


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import cupy as cp
    import cudf
    import numpy as np
    import pandas as pd

    try:
        from protlib.metric import get_funcs_mapper, get_ns_id, obo_parser, Graph, propagate_col_kernel
    except Exception:
        get_funcs_mapper, get_ns_id, obo_parser, Graph = [None] * 4

    # trainTerms = cudf.read_csv(args.path, sep='\t', usecols=['EntryID', 'term'])
    trainTerms = pd.read_csv(args.path, sep='\t', usecols=['EntryID', 'term'], dtype={'EntryID': 'string', 'term': 'string'})

    ontologies = []
    for ns, terms_dict in obo_parser(args.graph).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    back_prot_id = pd.Index(trainTerms['EntryID'].drop_duplicates())
    length = len(back_prot_id)
    prot_id = pd.Series(np.arange(length, dtype=np.int32), index=back_prot_id)
    trainTerms['id'] = trainTerms['EntryID'].map(prot_id).astype(np.int32)
    flg = True

    trainTerms = trainTerms.sort_values('id', kind='mergesort').reset_index(drop=True)
    ids_host = trainTerms['id'].to_numpy()  # numpy int32

ont_mappers = []
ont_rev_terms = []
ont_ns = []
for G in ontologies:
    ont_mappers.append(get_funcs_mapper(G))
    ont_rev_terms.append(get_funcs_mapper(G, False))
    ont_ns.append(get_ns_id(G))

for i in tqdm.tqdm(range(0, length, args.batch_size)):
    lo = i
    hi = min(i + args.batch_size, length)

    start = int(np.searchsorted(ids_host, lo, side='left'))
    end   = int(np.searchsorted(ids_host, hi, side='left'))

    sample = trainTerms.iloc[start:end]
    batch_len = hi - lo

    for k, G in enumerate(ontologies):
        mapper = ont_mappers[k]
        rev_terms = ont_rev_terms[k]
        ns_id, ns_str = ont_ns[k]

        sample_term_id = sample['term'].map(mapper)
        ok = sample_term_id.notna().to_numpy()
        if not ok.any():
            continue

        rid_host = (sample.loc[ok, 'id'].to_numpy(dtype=np.int32) - np.int32(lo))
        cid_host = sample_term_id.loc[ok].to_numpy(dtype=np.int32)

        m = (rid_host >= 0) & (rid_host < batch_len) & (cid_host >= 0) & (cid_host < G.idxs)
        rid_host = rid_host[m]
        cid_host = cid_host[m]
        if rid_host.size == 0:
            continue

        rid = cp.asarray(rid_host, dtype=cp.int32)
        cid = cp.asarray(cid_host, dtype=cp.int32)

        mat = cp.zeros((batch_len, G.idxs), dtype=cp.float32)
        mat.scatter_add((rid, cid), 1)
        mat = cp.clip(mat, 0, 1)

        propagate_max(mat, G)

        for j in range(0, mat.shape[0], args.batch_inner):
            row, col = cp.nonzero(mat[j: j + args.batch_inner])
            if row.size == 0:
                continue

            row_h = cp.asnumpy(row).astype(np.int32)
            col_h = cp.asnumpy(col).astype(np.int32)

            idx = lo + j + row_h

            valid = idx < length
            idx = idx[valid]
            col_h = col_h[valid]

            entry_ids = back_prot_id[idx].to_numpy()
            terms_out = np.asarray(rev_terms, dtype=object)[col_h]

            asp = ns_str.upper() + 'O'

            out_df = pd.DataFrame({
                'EntryID': entry_ids,
                'term': terms_out,
                'aspect': asp
            }).sort_values(['EntryID', 'term'], ascending=True)

            out_df.to_csv(args.output, sep='\t', index=False, mode=('w' if flg else 'a'), header=flg)
            flg = False
