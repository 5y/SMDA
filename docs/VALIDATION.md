# Validation record

Performed locally on 2026-09-23. Exact dependency versions are in `results/validation_environment.json`.

| Check | Result |
|---|---|
| Unit/integration tests | **17 passed**: Ridge against sklearn; both influence pathways against central finite differences; ReLU gating; feature ordering; threshold edge cases; curation flags; soft metrics; contrastive feature selection; tiny-Llama token/layer semantics; exact restoration on success and exception |
| Notebook schema | nbformat v4 validated; 20 clean cells; no saved outputs or execution counts in delivered notebook |
| Notebook execution | All default cells executed locally using nbclient; no cell errors; GPU/SFT flags disabled |
| Cached policy workflow | Six experiments fitted; 12 model JSONs saved; all use 75 explicit numeric feature IDs |
| Result consistency | Independent notebook execution produced identical cached metrics to the delivered run |
| Artifact download | Default public files downloaded at locked revisions and verified by SHA256 |
| Curation reconstruction | Full 200; corrected quantitative 114; qualitative 166 |
| Quantitative criterion | J OR L matches direct recomputation from influence-score medians; 101 rows differ from legacy mask |
| Qualitative membership | Reconstructed 166-pair set matches pinned public qualitative dataset exactly |
| Overlap check | Three exact trimmed-prompt overlaps, all harmless; explicit new disjoint set contains 197 SFT pairs |
| Script interfaces | All nine command-line scripts accept `--help`; Python compilation passed |
| Release scan | No copied credential patterns, user filesystem paths, or calls that publish to the Hub |

The local execution environment was macOS x86_64 with Python 3.12.14, NumPy 1.26.4 and PyTorch 2.2.2. The tiny Llama used randomly initialized weights and required no model download. The Google Colab browser/runtime itself was not tested. The Linux CI configuration is provided but was not executed on GitHub.

The local sandbox emitted non-fatal Arrow CPU-cache inspection warnings and prevented psutil process enumeration during notebook-kernel shutdown. All notebook cells completed, the result file was saved, and outputs matched the independent fit. This shutdown warning is recorded in the environment JSON.

Not validated: the gated 3B-model GPU inference/update loop, CUDA memory/runtime, full SFT, SAE training, exact historical experiment membership, paper figure/table replication, remote judging, and the optional Colab ZIP-upload UI. The GPU requirements file is a proposed environment, not a historical lock or a tested CUDA installation.
