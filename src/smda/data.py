"""Stable row IDs, strict feature alignment, split and overlap checks."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def prompt_id(prompt):
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def pair_id(prompt, response):
    return hashlib.sha256((prompt.strip() + "\0" + response.strip()).encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".csv":
        import csv
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON list of records.")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def feature_matrix(rows, feature_ids=None):
    if not rows:
        raise ValueError("No evaluation rows.")
    mapping = {}
    for key in rows[0]:
        m = re.match(r"^feat_(\d+)(?:_|$)", key)
        if m:
            fid = int(m.group(1))
            if fid in mapping:
                raise ValueError(f"Ambiguous columns for feature {fid}.")
            mapping[fid] = key
    ids = sorted(mapping) if feature_ids is None else list(map(int, feature_ids))
    if len(ids) != len(set(ids)) or not ids or set(ids) - mapping.keys():
        raise ValueError("Missing or duplicated feature IDs.")
    X = np.asarray([[row[mapping[i]] for i in ids] for row in rows], dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite activations.")
    return X, ids


def unique_prompts(rows):
    ids = [prompt_id(row["prompt"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evaluation prompts; resolve duplicates before splitting.")
    return ids


def validate_pairs(rows, eval_prompts=()):
    seen, evaluation = set(), {prompt_id(p) for p in eval_prompts}
    result = []
    for row in rows:
        if not row.get("prompt", "").strip() or not row.get("response", "").strip():
            raise ValueError("Empty SFT prompt/response.")
        if row.get("label") not in {"harmful_refusal", "harmless_compliance"}:
            raise ValueError("Unknown SFT label.")
        pid = pair_id(row["prompt"], row["response"])
        if pid in seen:
            raise ValueError("Duplicate SFT pair; use an explicit selection manifest.")
        if prompt_id(row["prompt"]) in evaluation:
            raise ValueError("SFT/evaluation prompt overlap.")
        seen.add(pid)
        result.append({**row, "pair_id": pid})
    if not result:
        raise ValueError("Empty SFT set.")
    return result


def experiment_indices(rows, seed=42):
    """Deterministic reconstructed splits; NOT the unavailable historical cache."""
    unique_prompts(rows)
    rng = np.random.default_rng(seed)
    harmful = np.array([i for i, r in enumerate(rows) if r["prompt_label"] == "harmful"])
    harmless = np.array([i for i, r in enumerate(rows) if r["prompt_label"] == "harmless"])
    refused = np.array([r["refused"] for r in rows])
    if not np.isin(refused, [0, 1]).all():
        raise ValueError("Expected binary ground-truth refusal labels.")
    a, b = harmful[refused[harmful] == 1], harmful[refused[harmful] == 0]
    n = min(len(a), len(b))
    if not n or min(len(harmful), len(harmless)) < 2500:
        raise ValueError("Insufficient data for paper-sized splits.")
    splits = {"harmful_natural": harmful,
              "harmful_balanced": rng.permutation(np.r_[rng.choice(a, n, replace=False), rng.choice(b, n, replace=False)]),
              "harmful_harmless": rng.permutation(np.r_[rng.choice(harmful, 2500, replace=False), rng.choice(harmless, 2500, replace=False)])}
    for s in (0, 1, 2):
        splits[f"appendix_seed{s}"] = np.random.default_rng(s).choice(harmful, 2000, replace=False)
    return {k: v.tolist() for k, v in splits.items()}
