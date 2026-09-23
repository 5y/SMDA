"""Score refusal-prefix probabilities or generate local judge inputs."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from smda.data import read_rows, unique_prompts, write_json, file_sha256
from smda.metrics import soft_metrics


def run(model_id, revision, data_path, output, *, generate=False, batch_size=8, dtype="float32", device="cuda"):
    import torch
    from smda.llama import load_model, refusal_logprobs, prompt_tokens
    rows = read_rows(data_path)
    ids = unique_prompts(rows)
    prompts = [r["prompt"] for r in rows]
    labels = [r.get("prompt_label", r.get("label")) for r in rows]
    if not all(x in {"harmful", "harmless"} for x in labels):
        raise ValueError("Evaluation labels must be harmful/harmless.")
    model, tokenizer = load_model(model_id, revision, dtype=dtype, device=device, token=os.environ.get("HF_TOKEN"))
    records = []
    if generate:
        for pid, prompt, label in zip(ids, prompts, labels):
            inputs = torch.tensor([prompt_tokens(tokenizer, prompt)], device=device)
            with torch.no_grad():
                output_ids = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=30,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
            records.append({"prompt_id":pid, "prompt":prompt, "label":label,
                            "response":tokenizer.decode(output_ids[0, inputs.shape[1]:], skip_special_tokens=True)})
        metrics = None
    else:
        y = refusal_logprobs(model, tokenizer, prompts, [[40,649,956],[40,649,1431]], batch_size=batch_size)
        metrics = soft_metrics(np.exp(y), np.array(labels)=="harmful")
        records = [{"prompt_id":pid, "label":label, "refusal_logprob":float(lp), "prob_i_cant":float(np.exp(lp))}
                   for pid,label,lp in zip(ids, labels, y)]
    write_json(output, {"model":model_id, "revision":revision, "dtype":dtype,
        "data_sha256":file_sha256(data_path), "mode":"generation" if generate else "soft_prefix_probability",
        "metrics":metrics, "records":records})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True, help="Immutable commit for Hub models; use local for a local path")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=["float32","bfloat16"], default="float32")
    p.add_argument("--device", default="cuda")
    a=p.parse_args()
    run(a.model, None if a.revision=="local" else a.revision, a.data, a.output,
        generate=a.generate, batch_size=a.batch_size, dtype=a.dtype, device=a.device)
