# Data card and artifact inventory

The repository distributes code, fitted Ridge coefficients, metrics, and hash-only manifests. Source prompt/response datasets are downloaded on demand from pinned public repositories; they are not copied into the release archive.

| Asset | Current size | Use | Notes |
|---|---:|---|---|
| `DarianNLP/logprob_mda_dataset_NEW` | 7,826 | Cached downstream evaluation | 5,326 harmful + 2,500 harmless; 75 feature columns, refusal labels, log-probability |
| `DarianNLP/sae_feature_labels_NEW` | 75 | Human-readable feature descriptions | IDs are authoritative; names may differ from the paper |
| `DarianNLP/affect_of_removing_misalligned_examples-full` | 200 | Final SFT pairs and curation flags | 100 harmful-refusal + 100 harmless-compliance |
| Same prefix, `qual_removed` | 166 | Qualitative reference membership | Matches the 34 primary-rater removals reconstructed locally |
| Same prefix, `quant_removed` | 113 | Legacy artifact only | Does not match corrected quantitative criterion; excluded from default download |
| Corrected quantitative subset | 114 | Derived locally from full dataset | 53 harmful + 61 harmless retained; not a separately bundled dataset |
| Same prefix, `judge_subset` | 200 | Fixed generation evaluation | 100 harmful + 100 harmless |
| `DarianNLP/mda_sft_training_pairs` | 220 | Earlier exploratory pool | Includes corruption tests; not the final paper set |

Original source families described in the paper include AdvBench, HarmBench, WildJailbreak, Alpaca, MaliciousInstruct, Do-Not-Answer, Dolly, FLAN, and Safety-Tuned Llama. The supplied paper contains their citations. This release has not reconstructed the raw 134,218-example SAE-training set or the complete 11,000-prompt feature-selection dataset from those sources. Their exact revisions, preprocessing order, source-specific licenses and final released terms need author verification.

The studied data are described as English safety/refusal and instruction-following prompts. No representative population/demographic coverage is established. Prompts and responses include harmful/offensive content by design. The public data should not be described as free of personal information: no comprehensive PII audit was performed during this refactor.

Data identities use SHA256 of trimmed UTF-8 prompts and of `trimmed_prompt + NUL + trimmed_response` for pairs. Hashes help align rows and identify exact duplicates; they are not an anonymization guarantee for searchable text. Raw prompt content is intentionally absent from committed overlap and selection manifests.

Three harmless SFT prompts overlap the 7,826-prompt cache. The influence runner rejects overlaps. `prepare_influence_data.py --drop-overlapping-sft` makes the choice explicit and exports a 197-pair set plus removal IDs; this defines a new experiment. Full historical 200-pair claims require a verified original evaluation manifest.

Artifact access was checked without authentication on 2026-09-23. SHA256 and repository revisions are in `configs/artifacts.lock.json`. No automatic upload or Hub publication is performed by any script.
