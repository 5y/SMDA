# Model card

## Identity and scope

SMDA is a research attribution framework, not a new standalone language-model architecture. Authors: Reza Habibi, Darian Lee, Magy Seif El-Nasr. This release candidate contains fitted symbolic Ridge models and code for working with separately hosted language models and an SAE.

Intended use: inspect how one supervised update changes an interpretable surrogate of a model's refusal behavior, compare feature/target pathways, and investigate training-data curation. Intended users are NLP and interpretability researchers. English safety-relevant prompts are the studied setting. Neither the Ridge score nor the prefix probability is a safety certification or a general classifier of harmfulness.

## Components

| Component | Configuration | Artifact/status |
|---|---|---|
| Base language model | Llama-3.2-3B-Instruct, roughly 3.2B parameters | Gated upstream artifact, revision pinned; no weights bundled |
| SAE | 3,072 → 12,288; BatchTopK k=40; post-block 10; SAELens 6.37.6 | Public checkpoint metadata verified; full checkpoint inference not validated here |
| Symbolic policy | 75 numeric feature IDs, Ridge λ=5, no intercept or scaling | 12 JSON files: 6 held-out classification fits and 6 full-set attribution fits, actually trained locally on cached features |
| Fine-tuned LM | Full-parameter SFT, 3 epochs, bf16, AdamW 8-bit | Existing full / qualitative / corrected-quantitative model IDs recorded below; not rerun or verified by inference |

The saved SAE was trained as BatchTopK, but the paper and downstream notebook define selected features using a dense ReLU proxy. Fresh influence code explicitly implements this paper equation; it does not quietly apply the checkpoint's input-centering, decoder-norm rescaling or BatchTopK threshold.

## Existing language-model artifacts

- [Base model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Published SAE](https://huggingface.co/DarianNLP/sae-llama-3.2-3b-instruct-layer10-safety)
- [Full fine-tuning](https://huggingface.co/DarianNLP/affect_of_removing_misalligned_examples-full)
- [Qualitative-removal fine-tuning](https://huggingface.co/DarianNLP/affect_of_removing_misalligned_examples-qual_removed)
- [Corrected quantitative-removal checkpoint](https://huggingface.co/5yhuggin/affect_of_removing_misalligned_examples-quant-corrected-20260830)

Immutable revisions are in `configs/artifacts.lock.json`. Existence/metadata checks do not prove that these exact commits generated the paper's tables. The corrected quantitative checkpoint's model-card license metadata is incomplete; authors should repair cards and release terms before publication.

## Loading a symbolic model

```python
from smda import RidgePolicy
from smda.data import read_rows, feature_matrix

policy = RidgePolicy.load("models/harmful_natural.json")
rows = read_rows("data/downloads/logprob_mda_dataset_NEW.parquet")
X, feature_ids = feature_matrix(rows, policy.feature_ids)
scores = policy.predict(X, feature_ids)
refused = policy.classify(X, feature_ids)
```

`scores` are surrogate log-probability predictions; Ridge predictions need not lie in the range of valid log probabilities. `classify` uses the training-selected threshold. File suffix `_attribution` denotes a fit on the full experiment evaluation set with no classification threshold. Do not use that full-set model to report held-out classification performance.

## Evaluation and limitations

The paper reports test accuracies 0.673 / 0.639 / 0.696 for the three main symbolic policies. This release's cached-data reanalysis obtained approximately 0.670 / 0.630 / 0.765 under its reconstructed splits and exact training-threshold search. The difference is unresolved; the included models are labeled reanalysis rather than original paper models.

For fine-tuned LMs, the paper reports qualitative-removal F1 of 0.876 (soft prefix metric) and 0.857 (judge), higher than the other fine-tuned variants. Judged accuracy is 0.840, below base at 0.850 and above full SFT at 0.795. These are paper-reported values, not rerun measurements.

Known limitations: modest surrogate fidelity; prefix-based refusal proxy; semantic uncertainty/polysemantic SAE features; first-order local approximation; one base LM and a small SFT set; unstable ranking under evaluation-set changes; incomplete artifact provenance; three overlapping harmless SFT/evaluation prompts in current public artifacts. Positive `ΔW·W` denotes coefficient-vector alignment, which is not equivalent to higher refusal on every prompt.

Training data include offensive and safety-sensitive content. Human/LLM labels can encode model and annotator biases. Do not infer demographic coverage or safe deployment from these experiments. See the supplied paper's limitations and ethics discussion for the research context.

## Licensing and compute

Upstream model licenses/access conditions apply. New code and derived symbolic-model release terms await the authors' decision in `LICENSE_NOTICE.md`. Raw datasets and base/fine-tuned LM weights are not bundled. Local CPU tests and cached policy fitting were executed; full GPU runs, peak memory, training cost, and emissions were not measured here.
