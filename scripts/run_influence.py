"""Fresh, paper-equation SMDA runs; corrected semantics require new result labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from smda.data import (read_rows, feature_matrix, unique_prompts, experiment_indices,
                       validate_pairs, write_json, file_sha256)
from smda.influence import decompose_influence, linearized_features
from smda.policy import fit_policy


def run(config_path, eval_path, sft_path, output, *, limit=None, split_manifest=None):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from smda.llama import load_model, residual_activations, refusal_logprobs, single_example_update
    cfg = json.loads(Path(config_path).read_text())
    if cfg["encoder"] != "paper_relu_proxy":
        raise ValueError("Only the explicitly named paper ReLU proxy is supported here.")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rows = read_rows(eval_path)
    prompt_ids = unique_prompts(rows)
    prompts = [r["prompt"] for r in rows]
    # Feature values are ignored: only the published basis and text/labels are used.
    _, feature_ids = feature_matrix(rows)
    if len(feature_ids) != 75:
        raise ValueError("Expected the published 75-feature basis.")
    pairs = validate_pairs(read_rows(sft_path), prompts)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        pairs = pairs[:limit]
    if split_manifest:
        memberships = json.loads(Path(split_manifest).read_text())
        lookup = {pid:i for i,pid in enumerate(prompt_ids)}
        splits = {name: [lookup[p] for p in ids] for name, ids in memberships.items()}
    else:
        splits = experiment_indices(rows, cfg["seed"])
    for idx in splits.values():
        if not idx or len(idx) != len(set(idx)):
            raise ValueError("Empty/duplicated split indices.")
    manifest = {"config": cfg, "eval_sha256": file_sha256(eval_path), "sft_sha256": file_sha256(sft_path),
                "feature_ids": feature_ids, "splits": splits, "pair_ids": [r["pair_id"] for r in pairs],
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "result_status": "corrected reimplementation; not historical paper results"}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    out = Path(output)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Output directory belongs to a different run. Choose a new directory.")
    write_json(manifest_path, {**manifest, "fingerprint": fingerprint})
    token = os.environ.get("HF_TOKEN")
    weights_path = hf_hub_download(cfg["sae_id"], "sae_weights.safetensors", revision=cfg["sae_revision"], token=token)
    weights = load_file(weights_path)
    w = weights["W_enc"].float().numpy()[:, feature_ids].astype(np.float64)
    b = weights["b_enc"].float().numpy()[feature_ids].astype(np.float64)
    del weights
    model, tokenizer = load_model(cfg["model_id"], cfg["model_revision"], device=cfg["device"], dtype="float32", token=token)
    write_json(out / "token_diagnostics.json", {"target_decodings": [tokenizer.decode(x) for x in cfg["target_sequences"]],
               "sae_weights_sha256": file_sha256(weights_path), "block_index": cfg["layer"], "position": "last user eot"})
    common = {"batch_size": cfg["eval_batch_size"], "max_tokens": cfg["max_prompt_tokens"]}

    def evaluate():
        h = residual_activations(model, tokenizer, prompts, layer=cfg["layer"], **common)
        y = refusal_logprobs(model, tokenizer, prompts, cfg["target_sequences"], **common)
        return h, y

    h0, y0 = evaluate()
    X = np.maximum(h0 @ w + b, 0)
    np.savez_compressed(out / "baseline.npz", h=h0, X=X, Y=y0, feature_ids=feature_ids)
    for name, idx in splits.items():
        fit_policy(X[idx], y0[idx], feature_ids, alpha=cfg["alpha"], metadata={"fingerprint": fingerprint}).save(out / "policies" / f"{name}.json")
    for number, row in enumerate(pairs):
        target = out / "pairs" / (row["pair_id"] + ".json")
        if target.exists():
            saved = json.loads(target.read_text())
            if saved.get("fingerprint") != fingerprint:
                raise ValueError("Incompatible saved pair result.")
            print("resume", number + 1, flush=True)
            continue
        with single_example_update(model, tokenizer, row["prompt"], row["response"],
            learning_rate=cfg["learning_rate"], max_length=cfg["sft_max_length"]) as step:
            h1, y1 = evaluate()
        dx = linearized_features(h1-h0, h0, w, b)
        dy = y1-y0
        record = {"pair_id": row["pair_id"], "label": row["label"], "fingerprint": fingerprint,
                  "step": step, "mean_delta_y": float(dy.mean()), "experiments": {}}
        for name, idx in splits.items():
            influence = decompose_influence(X[idx], y0[idx], dx[idx], dy[idx], alpha=cfg["alpha"])
            record["experiments"][name] = {"feature_ids": feature_ids,
                "delta_w_x": influence.delta_w_x.tolist(), "delta_w_y": influence.delta_w_y.tolist(),
                "delta_w": influence.delta_w.tolist(), "weight_alignment_score": influence.score,
                "weight_alignment_by_feature": influence.weight_alignment_by_feature.tolist()}
        tmp = target.with_suffix(".tmp")
        write_json(tmp, record)
        tmp.replace(target)
        print(f"pair {number+1}/{len(pairs)} saved", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/influence.json")
    p.add_argument("--eval-data", required=True)
    p.add_argument("--sft-data", required=True)
    p.add_argument("--output", default="runs/influence")
    p.add_argument("--limit", type=int, help="Smoke test only; omitted for all pairs")
    p.add_argument("--split-manifest")
    a = p.parse_args()
    run(a.config, a.eval_data, a.sft_data, a.output, limit=a.limit, split_manifest=a.split_manifest)
