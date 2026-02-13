import re
import time
import requests
import pandas as pd
from typing import List, Dict, Tuple, Optional

# ==========
# Config
# ==========
STRING_BASE = "https://string-db.org/api"   # 也可用版本固定地址如 https://version-12-0.string-db.org/api
CALLER_IDENTITY = "cafa6_ppi_feature_build" # 随便写个你自己的标识
SPECIES = 9606                              # 你示例是人类 OX=9606
TIMEOUT = 60

def parse_uniprot_accessions_from_fasta(fasta_path: str) -> List[str]:
    """
    Parse UniProt accessions from headers like:
    >sp|A0A0C5B5G6|MOTSC_HUMAN ...
    >sp|A0JNW5|BLT3B_HUMAN ...
    """
    accs = []
    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            # UniProt style: >db|ACCESSION|ENTRY_NAME ...
            m = re.match(r"^>[^|]*\|([^|]+)\|", line.strip())
            if m:
                accs.append(m.group(1))
            else:
                # fallback: take first token after ">"
                accs.append(line[1:].split()[0])
    # unique, keep order
    seen = set()
    out = []
    for x in accs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def string_get_string_ids(
    identifiers: List[str],
    species: int,
    limit: int = 1,
    echo_query: int = 1,
    chunk_size: int = 500
) -> pd.DataFrame:
    """
    Map UniProt IDs -> STRING stringId using /get_string_ids.
    STRING recommends mapping first. :contentReference[oaicite:1]{index=1}
    """
    url = f"{STRING_BASE}/tsv/get_string_ids"
    rows = []

    for start in range(0, len(identifiers), chunk_size):
        chunk = identifiers[start:start+chunk_size]
        params = {
            "identifiers": "\r".join(chunk),
            "species": species,
            "limit": limit,
            "echo_query": echo_query,
            "caller_identity": CALLER_IDENTITY,
        }
        r = requests.post(url, data=params, timeout=TIMEOUT)
        r.raise_for_status()

        # Parse TSV
        lines = r.text.strip().split("\n")
        if not lines:
            continue

        header = lines[0].split("\t")
        for line in lines[1:]:
            if not line.strip():
                continue
            vals = line.split("\t")
            rows.append(dict(zip(header, vals)))

        time.sleep(0.2)  # polite

    df = pd.DataFrame(rows)
    # keep best match (limit=1 already does)
    return df


def string_interaction_partners(
    string_ids: List[str],
    species: int,
    required_score: int = 400,
    limit: int = 100,
    chunk_size: int = 100
) -> pd.DataFrame:
    """
    Fetch interaction partners for each STRING ID using /interaction_partners. :contentReference[oaicite:2]{index=2}
    Returns a long table; we will post-process to (protein_a, protein_b, score).
    """
    url = f"{STRING_BASE}/tsv/interaction_partners"
    all_rows = []

    for start in range(0, len(string_ids), chunk_size):
        chunk = string_ids[start:start+chunk_size]
        params = {
            "identifiers": "\r".join(chunk),
            "species": species,
            "required_score": required_score,  # 0-1000
            "limit": limit,                    # partners per query protein
            "caller_identity": CALLER_IDENTITY,
        }
        r = requests.post(url, data=params, timeout=TIMEOUT)
        r.raise_for_status()

        lines = r.text.strip().split("\n")
        if not lines:
            continue

        header = lines[0].split("\t")
        for line in lines[1:]:
            if not line.strip():
                continue
            vals = line.split("\t")
            all_rows.append(dict(zip(header, vals)))

        time.sleep(0.2)

    return pd.DataFrame(all_rows)


def build_edge_table(
    df_partners: pd.DataFrame,
    mapping_df: pd.DataFrame,
    score_col_candidates: List[str] = ("combined_score", "score")
) -> pd.DataFrame:
    """
    Convert STRING interaction_partners output into edge table:
      protein_a, protein_b, score
    Also map STRING IDs back to your UniProt accessions when possible.
    """
    if df_partners.empty:
        return pd.DataFrame(columns=["protein_a", "protein_b", "score", "protein_a_uniprot", "protein_b_uniprot"])

    # identify columns for nodes & score
    # interaction_partners typically has: stringId_A, stringId_B, preferredName_A, preferredName_B, score, ...
    col_a = "stringId_A" if "stringId_A" in df_partners.columns else None
    col_b = "stringId_B" if "stringId_B" in df_partners.columns else None
    if col_a is None or col_b is None:
        raise ValueError(f"Cannot find stringId_A/stringId_B columns in: {df_partners.columns.tolist()}")

    score_col = None
    for c in score_col_candidates:
        if c in df_partners.columns:
            score_col = c
            break
    if score_col is None:
        # some versions use "score"
        if "score" in df_partners.columns:
            score_col = "score"
        else:
            raise ValueError(f"Cannot find a score column in: {df_partners.columns.tolist()}")

    edges = df_partners[[col_a, col_b, score_col]].copy()
    edges.columns = ["protein_a", "protein_b", "score"]
    edges["score"] = pd.to_numeric(edges["score"], errors="coerce")

    # Build mapping: UniProt -> stringId (get_string_ids returns columns: queryItem (optional), stringId, ...)
    # echo_query=1 gives queryItem
    uniprot_to_string = {}
    string_to_uniprot = {}
    if not mapping_df.empty and "stringId" in mapping_df.columns and "queryItem" in mapping_df.columns:
        for _, row in mapping_df.iterrows():
            uq = str(row["queryItem"])
            sid = str(row["stringId"])
            uniprot_to_string[uq] = sid
            # note: multiple UniProt can map to same stringId; keep first
            string_to_uniprot.setdefault(sid, uq)

    edges["protein_a_uniprot"] = edges["protein_a"].map(string_to_uniprot)
    edges["protein_b_uniprot"] = edges["protein_b"].map(string_to_uniprot)

    # drop NaN score
    edges = edges.dropna(subset=["score"])

    # optional: de-duplicate undirected edges (keep max score)
    # normalize order
    a = edges["protein_a"].astype(str).values
    b = edges["protein_b"].astype(str).values
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    edges["_u"] = list(zip(lo, hi))
    edges = edges.sort_values("score", ascending=False).drop_duplicates("_u").drop(columns=["_u"])

    return edges.reset_index(drop=True)


if __name__ == "__main__":
    fasta_path = "/root/autodl-tmp/cafa6/Train/train_head.fasta"

    # 1) parse UniProt accessions
    uniprot_accs = parse_uniprot_accessions_from_fasta(fasta_path)
    print(f"[INFO] Parsed {len(uniprot_accs)} UniProt accessions from FASTA")

    # 2) map UniProt -> STRING IDs
    df_map = string_get_string_ids(uniprot_accs, species=SPECIES, limit=1, echo_query=1)
    print(f"[INFO] Mapped {df_map['queryItem'].nunique() if not df_map.empty else 0} IDs to STRING")

    # keep only successfully mapped IDs
    mapped_string_ids = df_map["stringId"].dropna().astype(str).unique().tolist()
    print(f"[INFO] Got {len(mapped_string_ids)} unique STRING IDs")

    # 3) fetch interaction partners
    df_partners = string_interaction_partners(
        mapped_string_ids,
        species=SPECIES,
        required_score=400,   # 400/700 常用：400中等，700高置信
        limit=200             # 每个蛋白最多取多少个 partner
    )
    print(f"[INFO] Retrieved {len(df_partners)} raw partner rows")

    # 4) build edge table
    import numpy as np
    edges = build_edge_table(df_partners, df_map)
    print(edges.head())

    out_tsv = "/root/autodl-tmp/cafa6/embeds/string_ppi_edges.tsv"
    edges.to_csv(out_tsv, sep="\t", index=False)
    print(f"[INFO] Saved edge table to: {out_tsv}")
