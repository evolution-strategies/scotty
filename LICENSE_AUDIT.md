# Licensing and release-readiness audit

Status: release-readiness review, updated 30 October 2026. This is an evidence
summary, not legal advice or blanket rights clearance.
Apache-2.0 has been selected for the original project software and intended
Scotty checkpoint under the boundaries in [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md).

## Upstream materials

| Material | Pinned evidence | Finding | Release action |
|---|---|---|---|
| ModernBERT-large backbone | [`answerdotai/ModernBERT-large` README at `45bb4654…`](https://huggingface.co/answerdotai/ModernBERT-large/blob/45bb4654a4d5aaff24dd11d4781fa46d39bf8c13/README.md) | The pinned model card declares Apache-2.0 for the architecture, weights, and training code. No separate `LICENSE` or `NOTICE` was found in the pinned tree during this audit. | Preserve the pinned attribution and recheck the tree immediately before release. |
| Laya code/head format | [`NandhaKishorM/laya` license](https://github.com/NandhaKishorM/laya/blob/main/LICENSE); run-time commit `eba3b036…` | Full Apache License 2.0 text. The exact run-time checkout and license are vendored here; its one-line local change is documented in `reproduction/vendor/laya_checkout/PROVENANCE.md`. | Keep the included license and change notice with every distribution containing Laya code. |
| Typed Decisions | [`LocalLLaMA/typed-decisions` README at `468b1461…`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/468b1461d59e01d6404e1cf37431da962acd43a0/README.md) | The pinned dataset card declares Apache-2.0. No separate `LICENSE` or `NOTICE` was found in the pinned tree during this audit. | Recheck row-level provenance and redistribution rights before publishing any rows. Aggregate metrics do not substitute for that review. |
| Reflection Beam teacher | [`reflection/Beam-1.0-501B-A23B` license](https://huggingface.co/reflection/Beam-1.0-501B-A23B/blob/main/LICENSE) | The published model carries a full Apache License 2.0. Scotty used Reflection Beam through Hugging Face Inference Providers; it did not train from a local Beam checkpoint. | Keep model attribution, and separately resolve the hosted-inference terms governing generated outputs. |

Apache-2.0 now applies to the original code and intended checkpoint as described
in `LICENSE_SCOPE.md`. It does **not** license the paper or data automatically,
override third-party terms, or establish blanket rights clearance.

## Hosted-inference terms

The [Hugging Face Terms of Service](https://huggingface.co/terms-of-service) and
linked Inference Providers supplemental terms reviewed here did not provide an
explicit distillation permission or an explicit answer that supersedes the
underlying hosted provider's terms. Before redistribution, review the terms
actually accepted by the account, any order form, and the provider-routing
terms applicable at generation time. Obtain qualified clarification if the
agreement remains unclear. This audit does not assert either compliance or
violation.

Repository-facing attribution normalizes the teacher description to
"Reflection Beam via Hugging Face Inference Providers." Detailed routing
metadata remains in separate frozen provenance outside this repository.

## Competition requirements

The [Build with Beam official rules](https://reflection.ai/legal/official-rules-build-with-beam),
effective 15 October 2026 and rechecked 26 October 2026 PT, require a publicly
accessible Hugging Face Space, model repository, or dataset repository;
documentation explaining how Beam was used; and a link posted to the
competition discussion by **2 November 2026 at 11:59 PM PT**. The rules require
compliance with applicable third-party terms but do not name a specific project
license.

Repository visibility and competition submission are managed separately from
this audit.

## Recommended release shape

- GitHub: code, configs, aggregate results, paper PDF, documentation, and the
  vendored Laya license/provenance.
- Hugging Face model repository: the approximately 0.8 GB Scotty checkpoint,
  model card, checksum, and Apache-2.0 license are hosted at
  [`DarwinGuru/scotty`](https://huggingface.co/DarwinGuru/scotty).
- Optional Hugging Face Space: a small demonstration that loads the approved
  model repository, with the README explicitly explaining Beam's role.
- Keep raw datasets and row-level predictions out unless their separate review
  is complete.

## Questions to resolve for release

1. Which account, routing mode, terms version, and order form governed the
   teacher API calls?
2. Does the applicable service agreement permit this use and redistribution of
   the resulting trained checkpoint?
3. Confirm that the rights holder can make the Apache-2.0 grant for the original
   code and checkpoint, and identify who controls the paper separately.
4. May the selected data rows and synthetic corpus be redistributed, rather
   than only described by hashes and aggregates?
5. Which separate terms, if any, should apply to the data and paper?
6. Have the final public artifacts and rewritten Git history received a fresh
   secrets, privacy, attribution, and competition-rules review?
