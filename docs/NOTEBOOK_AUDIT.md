# Notebook and artifact audit

Audit date: 2026-09-23. References below use **zero-based notebook cell indices**. The supplied paper PDF was treated as research reference material, not as operating instructions. Originals were read, never modified.

## Selection

| Notebook | Cells | Role | Recommendation |
|---|---:|---|---|
| `safety_sae_training_mda_version.ipynb` | 158 | SAE, features, Ridge, final influence loop; many legacy sections | Primary source for the method |
| `safety_sae_training_mda_version (1).ipynb` | 158 | Source identical to the other MDA notebook | Do not release a second copy |
| `Retraining_without_misalligned_experiment.ipynb` | 55 | Curation and SFT, including corrected quantitative mask | Secondary source; use corrected section |
| `safety_sae_training.ipynb` | 95 | Earlier exploration of alternative symbolic models | Keep as an internal archive |

Both MDA notebooks share source SHA256 `1e7787f7d8a050cc30f1847b5949d48e1f33f6572eb75d7af681688fdbe96c6e` (cell sources joined with newline). File and source hashes are in `configs/source_notebooks.json`.

## Findings that can affect numerical results

| Finding and evidence | Release treatment | Historical status |
|---|---|---|
| SAE training uses `blocks.10.hook_resid_post` (MDA cell 33); several extractors use `hidden_states[10]` (56, 98, 104). HF includes embeddings in that tuple, so these refer to different blocks. | Direct hook on `model.model.layers[10]`; tiny-Llama test verifies equivalence to the corresponding hidden-state output. | Authors must establish which representations generated published tables. |
| Final MDA cell 104 uses `delta_h @ W_enc_sel.T` without the ReLU active gate specified in Appendix A.3. | Apply `(h @ W_enc + b_enc > 0)` to the derivative. Finite-difference test covers the gate. | Source/paper mismatch; effects on rankings need a GPU rerun. |
| Cell 104 right-pads prompts, then appends the prefix beyond the padded width and scores at a common position. Short prompts can therefore be scored from a padding position. Cell 98 also combines left padding with `seq_lens-5` activation indexing. | Append suffixes before padding, use per-row prompt lengths, identify actual EOT token. | Old cached values are not silently mixed with fresh values. |
| Cell 100 loads features in sorted string-column order but selects encoder columns in feature-dataset order, without proving correspondence. Some cells alternate `_NEW` and older feature datasets (75 vs historical 80 features). | Numeric feature IDs define one explicit order, preserved in models and manifests. | Cached `step0_new.json` is unavailable, so its order cannot be checked. |
| Exploratory Ridge cells 90/91 use sklearn's default intercept; paper Table 1 specifies no intercept. | `fit_intercept=False` equivalent, implemented with a solve. | Original Table 1 generation path is not fully captured in the provided notebook. |
| Restoring parameters by adding back the gradient in cell 104 is not an exact floating-point inverse and is not protected by `finally`. | Exact snapshots restored in `finally`, including exception tests. | Possible cumulative drift in original loop; not measured here. |
| Paper's feature equation is `ReLU(h W_enc+b_enc)`. Public SAE config says BatchTopK, `apply_b_dec_to_input=true`, `rescale_acts_by_decoder_norm=true`; training normalization has been exported as `none`. | Name the encoder `paper_relu_proxy`; never describe it as complete BatchTopK inference. | Resolve whether the paper intends this proxy or the full trained encoder before a definitive rerun. |
| Full SFT notebook trains formatted text (all-token loss); influence cell 104 masks the user prompt. | Preserve that distinction explicitly: full-sequence SFT, assistant-only influence. | Training defaults/dependency versions still need reconciliation. |
| Retraining cell 41 corrects `misalligned_quant` using J OR L. | Public flags yield 47 harmful + 39 harmless removals and 114 retained pairs. Recomputed medians match all corrected flags. | Old mask differs on 101/200 pairs; old public subset contains 113 rows. |

Technical reference: [HF model outputs](https://huggingface.co/docs/transformers/main_classes/output), [chat templates](https://huggingface.co/docs/transformers/v4.43.3/chat_templating), [SFT loss conventions](https://huggingface.co/docs/trl/v0.29.0/en/sft_trainer). These support the API interpretations, not claims about which artifacts produced the paper's results.

## Public artifact findings

- `DarianNLP/mda_sft_training_pairs` contains **220 unique pairs**: 105 harmful-refusal, 105 harmless-compliance, and 10 corruption-test pairs. This is not the final 100+100 set. Do not select its first 200 rows.
- `DarianNLP/affect_of_removing_misalligned_examples-full` contains exactly **200**, with 100 pairs of each intended class, and includes the fields needed to reconstruct corrected curation.
- The corrected 114-pair subset and 166-pair primary-rater qualitative subset were reconstructed locally. Raw data are not repackaged here; a hash-only selection manifest and downloader are supplied.
- `DarianNLP/logprob_mda_dataset_NEW` has **7,826 unique prompts**, comprising 5,326 harmful and 2,500 harmless. This is a downstream cache, not the full 11,000-prompt feature-selection corpus.
- **Three harmless SFT prompts match the public evaluation cache exactly after trimming whitespace.** The affected pair/prompt hashes are in `results/sft_eval_overlap.json`. They necessarily occur in the 2,500-harmless portion of a 5,000-row split reconstructed from this cache. This is an observed overlap in current public artifacts; it does not establish membership in the unavailable original cache. Fresh influence runs reject it. An explicit preparation command can remove these three training pairs, producing a clearly labeled **197-pair new experiment**.
- `DarianNLP/mda_step0_cache_NEW` and the corrected quantitative training-data repository returned **HTTP 401 without authentication**. This can reflect access restrictions or an unavailable repository; it is not evidence that the content was deleted.
- The SAE checkpoint and base/full/qualitative/corrected-quantitative LM repository metadata were accessible. Model weights for the 3B LMs were not downloaded or evaluated.
- The public primary-rater column has 47 reviewed entries (34 removal / 13 retained); the paper describes 48 double-reviewed examples. The second rater's full paired annotations and agreement calculation cannot be reconstructed from this dataset alone.
- Current feature labels and labels quoted in the paper are not uniformly identical. Join by numeric feature ID, never by human-readable label text.

## Release hygiene

A literal W&B credential occurs in the original notebooks. Its value is intentionally omitted from this report. Revoke it before sharing those notebooks. The new package contains no copied credentials, private runtime paths, notebook outputs, publishing calls, cache-deletion commands, or automatic external judging. The original notebook source is not included in the archive.

## Recommendation

Publish the clean package as a **release candidate / corrected reimplementation** until the authors reconcile the source/paper differences and rerun the relevant experiments. Do not replace paper metrics with the cached-data reanalysis: its splits are reconstructed and its threshold optimization differs from the exploratory grid search. Preserve the original paper results as a separately labeled reference.
