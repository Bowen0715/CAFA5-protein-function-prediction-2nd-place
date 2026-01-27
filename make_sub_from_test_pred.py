import os
import joblib
import numpy as np
import pandas as pd
import yaml

from protlib.metric import obo_parser, Graph, get_topk_targets

def build_cols_as_go_terms(graph_path, base_path, split):
    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    cols = []
    for n, k in enumerate(split):
        # 这里返回的是“索引”，不是 GO:xxxx
        idxs = get_topk_targets(
            ontologies[n],
            k,
            train_path=os.path.join(base_path, "Train")
        )

        # 用该 ontology 的 terms_list 映射成 GO term
        term_list = [x["id"] for x in ontologies[n].terms_list]  # e.g., ["GO:0000001", ...]
        cols.extend([term_list[int(i)] for i in idxs])

    return cols, split

def write_topk_per_ns(out_path, entry_ids, pred, cols, split, topk=500, chunk=2000):
    bp_n, mf_n, cc_n = split
    slices = {
        "bp": slice(0, bp_n),
        "mf": slice(bp_n, bp_n + mf_n),
        "cc": slice(bp_n + mf_n, bp_n + mf_n + cc_n),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        for st in range(0, pred.shape[0], chunk):
            ed = min(st + chunk, pred.shape[0])
            pchunk = pred[st:ed]
            echunk = entry_ids[st:ed]

            for slc in slices.values():
                block = pchunk[:, slc]
                B, M = block.shape
                if M == 0:
                    continue
                k = min(topk, M)

                idx = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
                probs = np.take_along_axis(block, idx, axis=1)
                order = np.argsort(-probs, axis=1)
                idx = np.take_along_axis(idx, order, axis=1)
                probs = np.take_along_axis(probs, order, axis=1)

                base = slc.start or 0
                for i in range(B):
                    eid = echunk[i]
                    for j in range(k):
                        term = cols[base + int(idx[i, j])]
                        # 关键 sanity: term 必须是 GO:xxxx
                        # 如果你这里又看到数字，那就是 cols 构建错了
                        f.write(f"{eid}\t{term}\t{float(probs[i, j]):.8f}\n")

def main():
    config_path = "/root/autodl-tmp/CAFA5-protein-function-prediction-2nd-place/config.yaml"
    model_name = "pb_t54500_raw"   # 改成你自己的
    pred_pkl = f"/root/autodl-tmp/cafa6/models/{model_name}/test_pred.pkl"  # 改成你自己的
    out_path = "/root/submission.tsv"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_path = config["base_path"]
    graph_path = os.path.join(base_path, "Train/go-basic.obo")
    model_config = config["base_models"][model_name]
    split = [model_config["bp"], model_config["mf"], model_config["cc"]]

    cols, split = build_cols_as_go_terms(graph_path, base_path, split)

    # ✅ sanity check：必须打印 GO:xxxx
    print("[CHECK] first 10 cols:", cols[:10])
    assert cols[0].startswith("GO:"), f"cols[0] is not GO term: {cols[0]}"

    test_pred = joblib.load(pred_pkl)
    print("[CHECK] test_pred:", test_pred.shape, test_pred.dtype)

    # EntryID 顺序：最稳妥是直接读 test_seq.feather 的 EntryID（通常就是 test_pred 的顺序）
    # 如果你确认 get_features_simple 可能 reorder，才需要用你 pipeline 的 test_idx 去对齐。
    test_feather = os.path.join(base_path, config["helpers_path"], "fasta/test_seq.feather")
    df = pd.read_feather(test_feather)
    entry_ids = df["EntryID"].astype(str).to_numpy()

    assert len(entry_ids) == test_pred.shape[0], (len(entry_ids), test_pred.shape[0])

    write_topk_per_ns(out_path, entry_ids, test_pred, cols, split, topk=500, chunk=2000)
    print("[OK] wrote:", out_path)

if __name__ == "__main__":
    main()
