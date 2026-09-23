# ACL / ARR artifact documentation mapping

This is a repository preparation guide, not a completed conference submission form. Consult the final venue's current instructions. [ARR's Responsible NLP Research guidance](https://aclrollingreview.org/responsibleNLPresearch/) covers limitations, artifact provenance and terms, data statistics, compute, experimental setup and reporting. It does not mandate one repository tree.

| Topic | Evidence prepared in this package | Remaining author work |
|---|---|---|
| Limitations and scope | Model card, audit, corrected-vs-historical distinctions | Resolve discrepancies with paper claims and update manuscript if needed |
| Artifact citations/versions | Citation metadata; source notebook hashes; immutable artifact lock | Final publication metadata and complete upstream citations |
| Licensing and intended use | License notice; research-use scope; separate upstream artifacts | Choose code and derived-model licenses; verify and state each dataset's terms |
| Data documentation/statistics | Data card; 200/114/166 selection manifests; overlap hashes | Raw SAE/11K source revisions, full preparation recipe, PII review, paired rater records |
| Compute and infrastructure | Tested CPU environment and validation report; explicit GPU settings | Original GPU types, peak memory, total run/search hours, budgets and new GPU validation |
| Hyperparameters/model selection | JSON configs; training-only CV API; explicit threshold selection | Recover original Ridge search grid and historical train/test/evaluation memberships |
| Metrics/uncertainty | Separate soft and judged metrics; six subset outputs | Exact paper rerun, seed-level influence comparison, appropriate uncertainty reporting |
| Implementation | Installable package, one Colab, numerical tests, CI configuration | Execute GPU pipeline at the intended release commit and record the environment |
| Human annotation | Primary-rater criterion and paper judge prompt documented | Paired annotations, agreement code, complete rubric/instructions and relevant participant documentation |
| AI assistance | This refactor and documentation were assisted by Codex | Authors review scientific correctness and make the disclosure required by the venue |

The repository is a reviewable public-release draft. It should not be advertised as a fully reproduced or ACL-certified artifact until the missing evidence has been supplied.
