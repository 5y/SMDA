"""Optional original SAELens training recipe, version-matched to saved SAE metadata.

Not run during release preparation. Produces a NEW basis: do not reuse historical
feature IDs/labels with a newly trained SAE until features have been selected again.
"""
import argparse
import importlib.metadata


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Local Hugging Face dataset directory containing text")
    p.add_argument("--output", default="runs/sae")
    a=p.parse_args()
    if importlib.metadata.version("sae-lens") != "6.37.6":
        raise RuntimeError("Use a separate environment with sae-lens==6.37.6, matching the artifact metadata.")
    from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, BatchTopKTrainingSAEConfig
    from sae_lens.config import LoggingConfig
    cfg = LanguageModelSAERunnerConfig(
        sae=BatchTopKTrainingSAEConfig(d_in=3072, d_sae=12288, k=40, normalize_activations="expected_average_only_in"),
        model_name="meta-llama/Llama-3.2-3B-Instruct", hook_name="blocks.10.hook_resid_post",
        dataset_path=a.dataset, is_dataset_tokenized=False, streaming=False, prepend_bos=True,
        context_size=96, training_tokens=20_000_000, train_batch_size_tokens=4096, lr=2e-4,
        n_batches_in_buffer=64, store_batch_size_prompts=32, feature_sampling_window=1000,
        dead_feature_window=5000, dead_feature_threshold=1e-4,
        logger=LoggingConfig(log_to_wandb=False), n_checkpoints=5, checkpoint_path=a.output,
        save_final_checkpoint=True, device="cuda", seed=42, dtype="float16")
    LanguageModelSAETrainingRunner(cfg).run()
