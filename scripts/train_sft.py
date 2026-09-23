"""Full fine-tuning from an explicit subset; save locally with a provenance record."""
import argparse
import json
import os
from pathlib import Path
from smda.data import read_rows, validate_pairs, file_sha256, write_json


def run(config_path, data_path, output):
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    from smda.llama import load_model, sft_encoding
    cfg = json.loads(Path(config_path).read_text())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full paper-style SFT requires a CUDA GPU with bf16 support.")
    rows = validate_pairs(read_rows(data_path))
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory for a fresh fine-tuning run.")
    set_seed(cfg["seed"])
    model, tokenizer = load_model(cfg["model_id"], cfg["model_revision"], dtype="bfloat16", token=os.environ.get("HF_TOKEN"))
    # Original retraining text dataset trained all tokens. Influence uses response-only loss.
    encoded = [sft_encoding(tokenizer, r["prompt"], r["response"], max_length=cfg["max_length"],
                            response_only=cfg["loss_scope"] == "assistant") for r in rows]
    model.config.use_cache = False

    def collate(batch):
        longest = max(len(r["input_ids"]) for r in batch)
        pads = {"input_ids": tokenizer.pad_token_id, "attention_mask": 0, "labels": -100}
        return {k: torch.tensor([r[k] + [pad]*(longest-len(r[k])) for r in batch], dtype=torch.long) for k,pad in pads.items()}

    args = TrainingArguments(output_dir=str(out), num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=2e-5,
        optim="adamw_bnb_8bit", bf16=True, gradient_checkpointing=True,
        save_strategy="epoch", save_total_limit=1, logging_steps=5, report_to=[],
        seed=cfg["seed"], data_seed=cfg["seed"], push_to_hub=False,
        lr_scheduler_type="linear", warmup_ratio=0, weight_decay=0, max_grad_norm=1.0)
    trainer = Trainer(model=model, args=args, train_dataset=encoded, data_collator=collate)
    result = trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    write_json(out / "smda_training_manifest.json", {"config":cfg, "data_sha256":file_sha256(data_path),
        "pair_ids":[r["pair_id"] for r in rows], "n_pairs":len(rows), "metrics":result.metrics,
        "note":"Fresh run. Match to historical results requires artifact and environment verification."})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/sft.json")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a=p.parse_args()
    run(a.config, a.data, a.output)
