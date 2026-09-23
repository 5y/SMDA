"""Fit CPU policies from pinned cached activations. Does not validate LLM extraction."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from smda.data import read_rows, feature_matrix, unique_prompts, experiment_indices, write_json, file_sha256
from smda.policy import fit_policy, select_alpha
from smda.metrics import classification_metrics


def run(data, output, *, alpha=5.0, cv=False, seed=42, split_manifest=None):
    rows = read_rows(data)
    ids = unique_prompts(rows)
    X, features = feature_matrix(rows)
    if len(features) != 75:
        raise ValueError(f"Paper configuration expects 75 features, found {len(features)}.")
    Y = np.asarray([r["refusal_logprob"] for r in rows], dtype=float)
    labels = np.asarray([r["refused"] for r in rows])
    if not np.isfinite(Y).all() or not np.isin(labels, [0,1]).all():
        raise ValueError("Invalid targets or labels.")
    if split_manifest:
        import json
        manifest = json.loads(Path(split_manifest).read_text())
        lookup = {pid:i for i,pid in enumerate(ids)}
        splits = {name: [lookup[p] for p in seq] for name, seq in manifest.items()}
        provenance = "supplied split manifest"
    else:
        splits = experiment_indices(rows, seed)
        provenance = "reconstructed splits; historical membership unavailable"
    out = Path(output)
    summaries = []
    all_manifests = {}
    for name, seq in splits.items():
        idx = np.asarray(seq, dtype=int)
        if len(set(seq)) != len(seq):
            raise ValueError("Repeated prompt in experiment split.")
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels[idx])
        selected_alpha, search = (select_alpha(X[tr], Y[tr], [0.01, 0.1, 1, 5, 10, 100], seed=seed)
                                  if cv else (alpha, []))
        metadata = {"experiment": name, "source_sha256": file_sha256(data), "seed": seed,
                    "fit_intercept": False, "feature_scaling": "none", "split_provenance": provenance,
                    "status": "cached-artifact reanalysis; not verified paper reproduction",
                    "alpha_selection": "10-fold training-only MSE" if cv else "fixed paper alpha",
                    "alpha_search": search}
        model = fit_policy(X[tr], Y[tr], features, alpha=selected_alpha, labels=labels[tr], metadata=metadata)
        model.save(out / "models" / f"{name}.json")
        # Attribution uses its own full-evaluation-set fit, separate from Table 1 testing.
        full = fit_policy(X[idx], Y[idx], features, alpha=selected_alpha, metadata={**metadata, "role": "full-set attribution"})
        full.save(out / "models" / f"{name}_attribution.json")
        train = classification_metrics(model.classify(X[tr], features), labels[tr])
        test = classification_metrics(model.classify(X[te], features), labels[te])
        summaries.append({"experiment": name, "n": len(idx), "alpha": selected_alpha, "threshold": model.threshold,
            "train": train, "test": test, "test_majority_baseline": float(max(labels[te].mean(), 1-labels[te].mean()))})
        all_manifests[name] = {"evaluation": [ids[i] for i in idx], "train": [ids[i] for i in tr], "test": [ids[i] for i in te]}
    write_json(out / "results" / "cached_metrics.json", summaries)
    write_json(out / "results" / "split_manifest.json", all_manifests)
    write_json(out / "results" / "evaluation_splits.json", {k: v["evaluation"] for k,v in all_manifests.items()})
    return summaries


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="runs/cached")
    p.add_argument("--cv", action="store_true", help="New training-only alpha search, not the unpublished original grid")
    p.add_argument("--split-manifest", help="JSON mapping experiment names to ordered prompt SHA256 IDs")
    a = p.parse_args()
    for row in run(a.data, a.output, cv=a.cv, split_manifest=a.split_manifest):
        print(row["experiment"], "n=", row["n"], "test accuracy=", round(row["test"]["accuracy"], 4))
