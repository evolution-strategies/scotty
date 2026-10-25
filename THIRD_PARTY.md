# Third-party sources and attribution

- ModernBERT backbone: `answerdotai/ModernBERT-large`, revision
  `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`; its pinned model card declares
  Apache-2.0.
- Typed head format and related code concepts: Laya contributors,
  <https://github.com/NandhaKishorM/laya>. The exact run-time checkout is
  included under `reproduction/vendor/laya_checkout/` with its upstream
  Apache-2.0 license and provenance. This preserves the third-party license; it
  does not license the rest of this repository.
- Typed Decisions data: `LocalLLaMA/typed-decisions`, revision
  `468b1461d59e01d6404e1cf37431da962acd43a0`; its pinned dataset card declares
  Apache-2.0.
- Synthetic teacher: Reflection Beam 1.0 501B-A23B, accessed through Hugging
  Face Inference Providers. The published Beam model carries an Apache-2.0
  license, while the applicable hosted-inference terms remain a separate
  release question. Detailed routing metadata is retained in private frozen
  provenance outside Git.

The reproduction bundle contains third-party and project-local code and is not
presented as wholly original work. Apache-2.0 applies only within
[`LICENSE_SCOPE.md`](LICENSE_SCOPE.md). Before public release, preserve all
upstream notices and resolve the redistribution status of every included file,
the checkpoint, both datasets, and the paper PDF.

See [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md) for pinned sources, service-terms
questions, and the proposed release split. That audit is not blanket legal
clearance.
