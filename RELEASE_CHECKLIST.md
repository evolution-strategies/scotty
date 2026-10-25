# Public-release checklist

Keep the repository private until every required item is reviewed and the owner
explicitly approves the visibility change.

- [ ] Confirm competition rules, submission timing, and anonymity requirements.
- [ ] Confirm author list, affiliation, acknowledgments, and paper title.
- [ ] Select a project license; none is granted now.
- [ ] Resolve ModernBERT, Laya, Typed Decisions, synthetic-corpus, and ESANN-class terms.
- [ ] Decide whether the model checkpoint may be distributed and where; do not add Git LFS costs silently.
- [ ] Review raw/row-level outputs for licensing, personal, proprietary, or harmful content.
- [ ] Replace or package missing project helper modules and record their provenance.
- [ ] Recreate a fully pinned environment and rerun smoke inference on the frozen checkpoint.
- [ ] Run multiple seeds or state clearly that only seed 0 is available.
- [ ] Preserve the single-seed, retrospective-test, calibration, and generalization caveats.
- [ ] Run a fresh secret scan across the entire Git history and release artifacts.
- [ ] Verify no local paths, internal endpoints, credentials, raw data, or checkpoints entered Git history.
- [ ] Add a DOI only if one is actually issued; the citation draft intentionally has none.
- [ ] Obtain explicit approval before changing GitHub visibility from private to public.
