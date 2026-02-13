import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def build_id_index(protein_ids: List[str]) -> Dict[str, int]:
    """
    Map protein ID -> row index in embedding matrix.
    """
    return {pid: i for i, pid in enumerate(protein_ids)}


def load_ppi_edges(
    ppi_path: str,
    sep: str = "\t",
    col_a: str = "protein1",
    col_b: str = "protein2",
    col_score: str = "combined_score",
    score_min: float = 0.0,
) -> pd.DataFrame:
    """
    Load PPI edges and filter by score.
    Expected columns: protein1, protein2, combined_score (STRING: 0-1000).
    """
    df = pd.read_csv(ppi_path, sep=sep)
    df = df[[col_a, col_b, col_score]].copy()
    df = df[df[col_score] >= score_min]
    df.rename(columns={col_a: "a", col_b: "b", col_score: "score"}, inplace=True)
    return df


def build_neighbor_list(
    edges: pd.DataFrame,
    make_undirected: bool = True,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build adjacency list: node -> [(neighbor, score), ...]
    """
    nbrs = defaultdict(list)
    for a, b, s in edges[["a", "b", "score"]].itertuples(index=False):
        nbrs[a].append((b, float(s)))
        if make_undirected:
            nbrs[b].append((a, float(s)))
    return nbrs


def score_transform(
    scores: np.ndarray,
    mode: str = "linear_0_1",
    eps: float = 1e-12
) -> np.ndarray:
    """
    Transform raw STRING scores into weights.
    STRING combined_score is typically 0..1000.

    mode:
      - "linear_0_1": w = score/1000
      - "softmax":   w = softmax(score)
      - "power":     w = (score/1000)^2
      - "log":       w = log(1 + score)
    """
    s = scores.astype(np.float32)
    if mode == "linear_0_1":
        w = s / 1000.0
    elif mode == "power":
        w = (s / 1000.0) ** 2
    elif mode == "log":
        w = np.log1p(s)
    elif mode == "softmax":
        # stable softmax
        x = s - np.max(s)
        ex = np.exp(x)
        w = ex / (np.sum(ex) + eps)
        return w
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # normalize to sum=1 (for weighted average)
    denom = np.sum(w) + eps
    return w / denom


def ppi_neighborhood_aggregate(
    protein_ids: List[str],
    embeddings: np.ndarray,
    neighbor_list: Dict[str, List[Tuple[str, float]]],
    id_to_idx: Optional[Dict[str, int]] = None,
    weight_mode: str = "linear_0_1",
    topk: Optional[int] = 50,
    score_min: float = 0.0,
    include_self: bool = False,
    self_weight: float = 1.0,
    default: str = "zeros",  # "zeros" or "self"
    dtype_out=np.float32,
) -> np.ndarray:
    """
    Compute z_ppi(p) = sum_{q in N(p)} w_pq * emb(q)  (weights normalized to sum=1)

    - topk: keep only top-k neighbors by score (recommended for speed & noise control)
    - score_min: filter neighbors below this score
    - include_self: optionally add the protein itself as a "neighbor"
    - default:
        - "zeros": if no valid neighbors, return all-zeros vector
        - "self":  if no valid neighbors, return emb(p)
    """
    if id_to_idx is None:
        id_to_idx = build_id_index(protein_ids)

    n, d = embeddings.shape
    out = np.zeros((n, d), dtype=dtype_out)

    for pid in protein_ids:
        i = id_to_idx.get(pid, None)
        if i is None:
            continue

        nbrs = neighbor_list.get(pid, [])
        # filter by score and by existence of neighbor embedding
        filtered = []
        for nb, s in nbrs:
            if s < score_min:
                continue
            j = id_to_idx.get(nb, None)
            if j is None:
                continue
            filtered.append((j, s))

        if include_self:
            filtered.append((i, float(self_weight)))

        if len(filtered) == 0:
            if default == "self":
                out[i] = embeddings[i].astype(dtype_out)
            # else keep zeros
            continue

        # top-k by score (desc)
        if topk is not None and len(filtered) > topk:
            filtered.sort(key=lambda x: x[1], reverse=True)
            filtered = filtered[:topk]

        idxs = np.array([x[0] for x in filtered], dtype=np.int64)
        scores = np.array([x[1] for x in filtered], dtype=np.float32)

        w = score_transform(scores, mode=weight_mode)  # normalized weights
        # weighted average of neighbor embeddings
        out[i] = (w[:, None] * embeddings[idxs]).sum(axis=0).astype(dtype_out)

    return out


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # 1) Your embeddings & ids
    # protein_ids = [...]
    # embeddings = np.load("llm_embeds.npy")  # shape (N, D)

    # 2) Load STRING edges
    # df_edges = load_ppi_edges(
    #     "string_edges.tsv",
    #     sep="\t",
    #     col_a="protein1",
    #     col_b="protein2",
    #     col_score="combined_score",
    #     score_min=400,   # e.g. keep medium/high confidence
    # )

    # 3) Build neighbor list
    # nbrs = build_neighbor_list(df_edges, make_undirected=True)

    # 4) Aggregate PPI neighborhood vectors
    # id_to_idx = build_id_index(protein_ids)
    # z_ppi = ppi_neighborhood_aggregate(
    #     protein_ids=protein_ids,
    #     embeddings=embeddings,
    #     neighbor_list=nbrs,
    #     id_to_idx=id_to_idx,
    #     weight_mode="linear_0_1",  # or "softmax"
    #     topk=50,
    #     score_min=400,
    #     include_self=False,
    #     default="zeros",
    # )

    # 5) Fuse features (simple concat)
    # X = np.concatenate([embeddings, z_ppi], axis=1)
    pass
