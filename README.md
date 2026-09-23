# Symbolic Mechanistic Data Attribution (SMDA)

**Reza Habibi, Darian Lee, and Magy Seif El-Nasr**  
University of California, Santa Cruz

Research release candidate for **Symbolic Mechanistic Data Attribution: Tracing Training Influence to Learned Behavioral Policies**.

SMDA fits a Ridge policy over selected SAE features and decomposes the effect of a single supervised update into feature-activation (ΔX) and refusal-probability (ΔY) pathways.

**Status:** the CPU reanalysis and numerical/model-adapter tests have been run. The full 3B-model influence loop, SAE training, and GPU fine-tuning have not been run in this release. The supplied notebooks and public artifacts contain discrepancies that prevent claiming exact reproduction of the paper. Read [the audit](docs/NOTEBOOK_AUDIT.md) before comparing numbers.

## Start here

Open [notebooks/SMDA.ipynb](notebooks/SMDA.ipynb) in Google Colab. Upload the accompanying `SMDA-release.zip` when prompted. This works before the repository is published and avoids a placeholder GitHub installation URL. The default walkthrough uses CPU-accessible cached features. GPU work is optional and explicitly enabled.

Locally, use Python 3.10–3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/download_artifacts.py
python scripts/fit_cached.py --data data/downloads/logprob_mda_dataset_NEW.parquet --output runs/cached
python scripts/prepare_subsets.py --data data/downloads/affect_of_removing_misalligned_examples-full.parquet --output data/curated
python -m pytest -q
```

For a fixed CPU dependency set, install `requirements-cpu.txt` before the editable package. The optional GPU adapter is installed with `pip install -e '.[llm,train]'`. `requirements-colab-gpu.txt` is a proposed Linux CUDA environment, not a tested record of the original experiments. Keep SAE training in a separate environment; its original library version is 6.37.6.

The download script checks immutable Hub revisions and file SHA256 hashes. It does not download Llama weights. Gated Llama access is only needed for fresh GPU experiments; provide `HF_TOKEN` through your environment or Colab Secrets.

## Which notebooks and models should be released?

Use `safety_sae_training_mda_version.ipynb` as the source for the **method**, and the corrected section of `Retraining_without_misalligned_experiment.ipynb` as the source for **curation and retraining**. The `(1)` MDA notebook is an exact source duplicate. The original `safety_sae_training.ipynb` mainly contains exploratory alternatives. Release this package and its single walkthrough, rather than the four work-in-progress notebooks.

There are three different model artifacts:

1. **Base LM:** `meta-llama/Llama-3.2-3B-Instruct`, used with the published layer-10 SAE. It is the reference for attribution.
2. **Symbolic model:** 75-feature, no-intercept Ridge regression, λ=5. Actual fitted JSON policies from a cached-data reanalysis are included in `models/`. These are usable research models, explicitly labeled as reanalysis rather than the paper's original weights.
3. **Fine-tuned LM:** `qual_removed` is the paper's strongest fine-tuned condition on F1 in both evaluation protocols. Retain base/full/corrected-quantitative comparisons. It is not uniformly better than base: judged accuracy is 0.840 versus 0.850 for base. Existing checkpoint identifiers and revisions are recorded in the model card; no new 3B checkpoint was trained here.

## Repository layout

```text
notebooks/SMDA.ipynb         One clean Colab walkthrough
src/smda/                   Policy, influence, Llama adapter, data checks, metrics
scripts/                    Download, fit, influence, curation, train, evaluate
configs/                    Experiment settings and immutable artifact lock
models/                     Fitted cached-data Ridge policies (JSON)
results/                    Actual reanalysis metrics, split IDs, overlap audit
tests/                      Numerical derivatives, masking, padding, restoration
docs/                       Audit, reproduction guide, model/data cards, ACL mapping
CITATION.cff                Author and paper metadata
LICENSE_NOTICE.md           Licensing status requiring an author decision
```

## Scientific conventions

- `layer=10` means the output of zero-based block 10, captured by a direct hook. The representation comes from the last user end-of-turn token.
- Refusal targets are the two token sequences in Appendix A.2: `[40,649,956]` and `[40,649,1431]`. Each sequence is scored under its own history; suffixes are appended before padding.
- The policy follows the paper's **dense ReLU encoder proxy**. It is not a full SAELens BatchTopK inference implementation. The saved SAE config includes additional centering/gating behavior; this difference needs scientific review.
- ΔX includes the baseline ReLU active gate. Matrix solves and the policy derivative use float64. The Llama influence forward/backward passes use float32.
- Every attribution example starts from exactly the same base weights, restored from CPU snapshots even if evaluation raises an exception. There is no gradient-norm normalization.
- Scalar `ΔW · W` measures alignment with the existing coefficient vector. It does not by itself prove that refusal probabilities increased on all prompts.
- Threshold selection and optional cross-validation use only policy-training rows; held-out test rows are evaluated once. Attribution uses a separately labeled full-evaluation-set fit.

## What still needs author review before publication?

The most consequential issues are the missing historical baseline/split cache, three SFT/evaluation prompt overlaps, layer/encoder mismatches, and the old quantitative mask. The corrected 114-pair subset can now be rebuilt from the public full dataset. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for exact commands and [NOTEBOOK_AUDIT.md](docs/NOTEBOOK_AUDIT.md) for evidence. Publishing also requires choosing code/model/data release licenses and checking artifact access.

This structure supports [ARR's reproducibility and artifact documentation guidance](https://aclrollingreview.org/responsibleNLPresearch/). That guidance does not prescribe a single Python repository layout. The checklist mapping here is not an assertion that every requirement has already been met.

Please cite the paper using `CITATION.cff`. Publication venue, year, DOI, and final public URL should be completed from the final bibliographic record.
