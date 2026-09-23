# Reproduction guide

## A. CPU walkthrough: available now

Run the README commands or the single Colab. The artifact lock downloads 7,826 cached feature rows, the 75-feature label dataset, the final 200-pair SFT table, and selected small reference datasets. Files are pinned by immutable commit and SHA256.

`fit_cached.py` builds six deterministic evaluation subsets, stratifies each into 80% training / 20% test, fits no-intercept Ridge with λ=5, and optimizes the classification threshold using only training labels. It exports a classification model and a separate full-set attribution model for each subset. `--cv` performs a **new** training-only, 10-fold MSE search over `[.01,.1,1,5,10,100]`; this grid is a release choice because the paper's original Ridge grid was not recovered.

The default reconstructed subsets have the paper's counts (5,326 / 3,986 / 5,000 and three 2,000-row appendix subsets), but their exact memberships are not asserted to match the historical cache. Models include feature order and data hashes. Every train/test prompt ID is exported. Supply `--split-manifest` with a JSON map of experiment name to ordered prompt IDs to use an author-verified evaluation membership.

Actual run results are in `results/cached_metrics.json`; a comparison to paper Table 1 is in `results/paper_comparison.csv`. The higher or lower accuracy of this reanalysis is not a claimed improvement to the paper's method. Cached activations/log-probabilities retain any issues in their original extraction.

## B. Recover the correct training subsets

```bash
python scripts/prepare_subsets.py \
  --data data/downloads/affect_of_removing_misalligned_examples-full.parquet \
  --output data/curated
```

Expected counts: full 200; corrected quantitative 114 (53 harmful + 61 harmless); qualitative 166 (66 harmful + 100 harmless). The code checks labels, unique pair hashes, strict booleans, the 47+39 removal counts, and the primary-rater 34 qualitative removals. It does not use the corrupted `misalligned_quant` field or a consensus/intersection qualitative mask.

## C. Fresh influence: optional GPU experiment

First explicitly remove the three SFT/evaluation overlaps for a **new 197-pair experiment**:

```bash
python scripts/prepare_influence_data.py \
  --sft-data data/curated/full.json \
  --eval-data data/downloads/logprob_mda_dataset_NEW.parquet \
  --output data/influence_disjoint.json --drop-overlapping-sft
python scripts/run_influence.py \
  --eval-data data/downloads/logprob_mda_dataset_NEW.parquet \
  --sft-data data/influence_disjoint.json --output runs/influence-smoke --limit 1
```

Then omit `--limit` and choose a new output directory for the full new experiment. Do not report this as the paper's 200-pair run. To reproduce the historical experiment, the authors must supply a verified disjoint 200-pair/evaluation manifest and reconcile the encoder/layer semantics first.

All baseline h/X/Y are recomputed with the same model, tokenizer, feature IDs, padding and dtype used after the update. Historical Y and fresh h are never mixed. A run fingerprint covers config, input files, feature IDs, splits and selected pairs. Each completed pair is written atomically; incompatible resumes fail. The two pathways and combined coefficient vector are retained separately.

The default evaluation batch is 8, rather than the notebook's 192, to reduce memory. `max_prompt_tokens` is 8,192 as an explicit release choice; inputs exceeding it fail instead of silently losing chat suffix tokens. The original code truncated to 256 in several cells. These changes mean fresh results must be labeled as reruns. The batch-size invariance test covers the scoring semantics, not bitwise invariance across real CUDA kernels.

Float32 Llama-3.2-3B weights require roughly 13 GB before gradients/activations. Exact restoration additionally needs about one model-sized CPU snapshot. A large-memory GPU and adequate host RAM are needed; **no free-Colab fit or runtime guarantee is made**. Actual GPU peak memory, wall time and total GPU-hours were not measured here. No remote job was started.

## D. Full SFT and evaluation

```bash
python scripts/train_sft.py --data data/curated/full.json --output runs/full
python scripts/train_sft.py --data data/curated/quant_removed_corrected.json --output runs/quant_removed_corrected
python scripts/train_sft.py --data data/curated/qual_removed.json --output runs/qual_removed
```

Configuration: 3 epochs, batch 1, accumulation 16, learning rate 2e-5, bf16, AdamW 8-bit, sequence length 1,024, seed 42, full-parameter training. Original formatted-text SFT is represented as full-sequence loss; do not silently change to assistant-only loss. Trainer scheduler/clipping defaults are made explicit in this release, but are not proven to match the unknown original library environment. Different dataset sizes yield different optimizer-step totals.

Use `evaluate_model.py` on an author-verified harmful/harmless evaluation set to produce soft metrics. Pass `--generate` on the fixed public 200-prompt judge subset to save continuations of at most 30 tokens. For a local model use `--revision local`. For a Hub model use its immutable revision in `configs/artifacts.lock.json`. Generation and probability evaluation are separate outputs.

```bash
python scripts/evaluate_model.py --model runs/qual_removed --revision local \
  --data data/downloads/affect_of_removing_misalligned_examples-judge_subset.parquet \
  --output runs/qual_removed/generations.json --generate
```

The paper judge prompt is documented in `docs/judge_prompt.txt`. External API judging is not executed automatically. Preserve raw responses, parsed categories, judge model/version, temperature, request IDs and errors in your judging run. Convert completed results into rows with `prompt_id`, `label`, `category`, then run `score_judgments.py`. Invalid or missing judgments cause an error rather than becoming compliance labels. Use the same fixed prompt IDs for every model.

## E. SAE training and labeling

The public checkpoint metadata specifies SAELens 6.37.6. `train_sae.py` preserves the source training recipe in a separate optional environment: 20M tokens, 12,288 features, BatchTopK k=40, context 96, activation batch 4,096, learning rate 2e-4, seed 42. It accepts an already prepared local HF dataset directory with a `text` field. Exact raw-data reconstruction, dataset revision, token packing order, and full training environment are not recovered here; this script alone is not a claim of end-to-end SAE reproduction.

A newly trained SAE has a new feature basis. Re-select and label features before using it. The provided main workflow uses the published 75-feature basis and does not call a labeling API. Complete layer-selection probing and annotation pipelines remain documented source methods, not fully reproduced experiments in this release.
