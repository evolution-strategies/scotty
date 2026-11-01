# Release checklist

Complete the applicable items before competition submission or general release.

- [x] Recheck the official competition rules and deadline on 26 October 2026 PT; no anonymity requirement is stated.
- [ ] Confirm the entrant is an eligible individual aged 18 or older, or identify every team member and the designated representative.
- [ ] Confirm the entrant is not excluded by employment, household, sanctions, export-control, or local-law restrictions.
- [ ] Confirm author list, affiliation, acknowledgments, and paper title.
- [x] Apply Apache-2.0 to original project software and intended checkpoint weights with explicit paper/data/third-party exclusions.
- [ ] Resolve the open questions in `LICENSE_AUDIT.md`; upstream Apache-2.0 declarations alone are not blanket clearance.
- [x] Host the verified complete checkpoint at `DarwinGuru/scotty`; keep it out of Git history.
- [ ] Confirm that Hugging Face visibility satisfies the competition requirements.
- [ ] Review raw/row-level outputs for licensing, personal, proprietary, or harmful content.
- [x] Package the project helper modules and record their provenance.
- [x] Rerun one documented smoke inference on the exact final checkpoint.
- [ ] Repeat that smoke inference from a fresh download of the recorded Hugging Face revision.
- [ ] Recreate a clean, fully pinned environment and repeat checkpoint inference.
- [x] Separate the three-seed validation summary from the hosted seed-0 deployment checkpoint.
- [x] Preserve the group-safe source-audit, calibration, and generalization caveats; do not call the audit external or fresh-distribution testing.
- [x] Run a fresh credential/secret scan across the entire reachable Git history and release artifacts on 26 October 2026 PT.
- [x] Remove paper source from every commit reachable through published Git refs; retain the provider-cache limitation documented in the audit record.
- [x] Verify no machine paths, internal endpoints, credentials, raw data, or checkpoints entered reachable Git history; the only local-path string is a negative unit-test assertion.
- [ ] Add a DOI only if one is actually issued; the citation draft intentionally has none.
- [ ] Confirm that the public Hugging Face repository independently provides the model, inference source, paper, and license information; GitHub visibility is managed separately.
- [x] Prepare judge-facing Hugging Face documentation and a discussion-post draft in `submission/README.md`.
- [ ] Publish the approved Hugging Face model and post its link before 2 November 2026 at 11:59 PM PT.
