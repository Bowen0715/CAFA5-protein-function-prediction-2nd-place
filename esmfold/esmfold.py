import os
from Bio import SeqIO
from transformers import AutoTokenizer, EsmForProteinFolding
import torch
import numpy as np

@torch.no_grad()
def esmfold_protein_embedding(
    seq: str,
    model,
    tokenizer,
    device: str = "cuda",
    num_recycles: int | None = None,
    which: str = "s_s",   # "s_s" / "states" / "s_z"
    pool: str = "mean",   # "mean" / "cls"(不推荐，ESM一般不用CLS) / "none"
) -> np.ndarray:
    """
    返回：
      - pool="mean": (d,) 的 protein-level 向量
      - pool="none":
          which="s_s" or "states": (L, d)
          which="s_z": (L, L, d)
    """
    model.eval().to(device)

    # ESM/ESMFold tokenizer: 建议 add_special_tokens=False
    inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ESMFold 的 forward 支持 num_recycles（不传就用 config 默认/最大值逻辑）
    outputs = model(**inputs, num_recycles=num_recycles)

    if which == "s_s":
        x = outputs.s_s  # (B, L, d)  per-residue embeddings from ESM-2 stem concat :contentReference[oaicite:1]{index=1}
    elif which == "states":
        x = outputs.states  # (B, L, d) folding trunk hidden states :contentReference[oaicite:2]{index=2}
    elif which == "s_z":
        x = outputs.s_z  # (B, L, L, d) pairwise residue embeddings :contentReference[oaicite:3]{index=3}
    else:
        raise ValueError("which must be one of: 's_s', 'states', 's_z'")

    # attention_mask: 1=有效残基，0=padding（这里一般不会padding，除非你做batch padding）
    attn = inputs.get("attention_mask", None)

    if pool == "none":
        return x[0].detach().float().cpu().numpy()

    if which == "s_z":
        # pairwise 的 pooling：常见做法是对 (L,L) 做 masked mean
        if attn is None:
            z = x[0]  # (L,L,d)
            return z.mean(dim=(0, 1)).detach().float().cpu().numpy()
        else:
            m = attn[0].bool()  # (L,)
            z = x[0]            # (L,L,d)
            pair_mask = (m[:, None] & m[None, :]).unsqueeze(-1)  # (L,L,1)
            z = z * pair_mask
            denom = pair_mask.sum().clamp_min(1).to(z.dtype)
            return (z.sum(dim=(0, 1)) / denom).detach().float().cpu().numpy()

    # which in {"s_s","states"}: (B,L,d) -> protein vector
    h = x[0]  # (L,d)

    if attn is None:
        return h.mean(dim=0).detach().float().cpu().numpy()

    m = attn[0].bool()  # (L,)
    h = h[m]
    if h.numel() == 0:
        # 极端情况：全是padding
        return torch.zeros((x.shape[-1],), device=h.device).cpu().numpy()

    return h.mean(dim=0).detach().float().cpu().numpy()


def batch_esmfold_embeddings_to_npy(
    fasta_path: str,
    out_npy: str,
    model_name: str = "facebook/esmfold_v1",
    device: str = "cuda",
    which: str = "s_s",          # "s_s" / "states" / "s_z"
    pool: str = "mean",
    num_recycles: int | None = None,
    max_len: int | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name)

    ids = []
    vecs = []

    for rec in SeqIO.parse(fasta_path, "fasta"):
        pid = rec.id
        seq = str(rec.seq)

        if max_len is not None and len(seq) > max_len:
            seq = seq[:max_len]

        v = esmfold_protein_embedding(
            seq=seq,
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_recycles=num_recycles,
            which=which,
            pool=pool,
        )

        ids.append(pid)
        vecs.append(v)

    X = np.stack(vecs, axis=0)  # (N,d) if pool="mean"
    os.makedirs(os.path.dirname(out_npy) or ".", exist_ok=True)
    np.save(out_npy, X)

    # 如果你还想保存 id 对应关系：
    np.save(out_npy.replace(".npy", "_ids.npy"), np.array(ids, dtype=object))

