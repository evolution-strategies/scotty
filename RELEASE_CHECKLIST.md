# Public-release checklist

Keep the repository private until every required item is reviewed and the owner
explicitly approves the visibility change.

- [ ] Confirm competition rules, submission timing, and anonymity requirements.
- [ ] Confirm author list, affiliation, acknowledgments, and paper title.
- [x] Apply Apache-2.0 to original project software and intended checkpoint weights with explicit paper/data/third-party exclusions.
- [ ] Resolve the open questions in `LICENSE_AUDIT.md`; upstream Apache-2.0 declarations alone are not blanket clearance.
- [ ] Decide whether the model checkpoint may be distributed and where; do not add Git LFS costs silently.
- [ ] Review raw/row-level outputs for licensing, personal, proprietary, or harmful content.
- [x] Package the project helper modules and record their provenance.
- [ ] Recreate a fully pinned environment and rerun smoke inference on the frozen checkpoint.
- [ ] Run multiple seeds or state clearly that only seed 0 is available.
- [ ] Preserve the single-seed, retrospective-test, calibration, and generalization caveats.
- [ ] Run a fresh secret scan across the entire Git history and release artifacts.
- [x] Remove paper source from every commit reachable through published Git refs; retain the provider-cache limitation documented in the audit record.
- [ ] Verify no local paths, internal endpoints, credentials, raw data, or checkpoints entered Git history.
- [ ] Add a DOI only if one is actually issued; the citation draft intentionally has none.
- [ ] Obtain explicit approval before changing GitHub visibility from private to public.
- [ ] If entering the competition, publish an approved Hugging Face model, dataset, or Space with Beam-use documentation and post it before 2 November 2026 at 11:59 PM PT.
