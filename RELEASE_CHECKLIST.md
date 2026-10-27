# Public-release checklist

Keep the repository private until every required item is reviewed and the owner
explicitly approves the visibility change.

- [x] Recheck the official competition rules and deadline on 26 October 2026 PT; no anonymity requirement is stated.
- [ ] Confirm the entrant is an eligible individual aged 18 or older, or identify every team member and the designated representative.
- [ ] Confirm the entrant is not excluded by employment, household, sanctions, export-control, or local-law restrictions.
- [ ] Confirm author list, affiliation, acknowledgments, and paper title.
- [x] Apply Apache-2.0 to original project software and intended checkpoint weights with explicit paper/data/third-party exclusions.
- [ ] Resolve the open questions in `LICENSE_AUDIT.md`; upstream Apache-2.0 declarations alone are not blanket clearance.
- [x] Host the verified complete checkpoint privately at `DarwinGuru/scotty-1`; keep it out of Git history.
- [ ] Obtain explicit approval before changing the Hugging Face model repository from private to public.
- [ ] Review raw/row-level outputs for licensing, personal, proprietary, or harmful content.
- [x] Package the project helper modules and record their provenance.
- [x] Rerun one documented smoke inference on the exact frozen checkpoint in the recovered environment.
- [ ] Recreate a clean, fully pinned environment and repeat checkpoint inference.
- [x] State clearly that only seed 0 is available; do not imply multiple-seed evidence.
- [x] Preserve the single-seed, retrospective-test, calibration, and generalization caveats.
- [x] Run a fresh credential/secret scan across the entire reachable Git history and release artifacts on 26 October 2026 PT.
- [x] Remove paper source from every commit reachable through published Git refs; retain the provider-cache limitation documented in the audit record.
- [x] Verify no machine paths, internal endpoints, credentials, raw data, or checkpoints entered reachable Git history; the only local-path string is a negative unit-test assertion.
- [ ] Add a DOI only if one is actually issued; the citation draft intentionally has none.
- [ ] Obtain explicit approval before changing GitHub visibility from private to public.
- [x] Prepare judge-facing Hugging Face documentation and a discussion-post draft in `submission/README.md`.
- [ ] Publish the approved Hugging Face model and post its link before 2 November 2026 at 11:59 PM PT.
