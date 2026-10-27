# Build with Beam submission package

Status: prepared privately on 26 October 2026 PT. Nothing in this directory
publishes the repositories or enters the competition.

## Official call

- [Competition discussion](https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5)
- [Official rules](https://reflection.ai/legal/official-rules-build-with-beam)
- Submission deadline: **2 November 2026 at 11:59 PM Pacific Time**.
- Required entry: a publicly accessible Hugging Face Space, model repository,
  or dataset repository; documentation of the project and how Beam was used;
  and a link posted to the competition discussion before the deadline.
- Judging: creativity/originality, technical execution, usefulness/impact, and
  innovative/effective use of Beam, each weighted 25%.
- Prizes: $9,000, $7,000, and $4,000 for first through third place.

The official rules control if this summary differs from them. They also cover
eligibility, taxes, required winner verification, publicity, and the license
granted to Reflection for competition administration and promotion.

## Submission assets

| Asset | Location | Current state | Needed for entry |
|---|---|---|---|
| Complete checkpoint and model card | <https://huggingface.co/DarwinGuru/scotty-1> | Private | Must be public during judging |
| Source, reproducibility, results, and paper | <https://github.com/evolution-strategies/scotty-1> | Private | Public is strongly recommended so judges can run the documented loader |
| Paper PDF | `paper/scotty-1-paper.pdf` | Packaged | Keep manuscript license separate |
| Discussion entry | Draft below | Not posted | Post only after links are anonymously accessible |

Raw datasets and row-level predictions are not part of the proposed entry.

## Judge-facing story

**Problem.** Generative foundation models can supply broad decision knowledge,
but many practical systems need only a quick, constrained choice among known
options.

**Approach.** Use Reflection Beam as a synthetic teacher across 18 typed-
decision task families, then train a 421M-parameter ModernBERT model with a
shared candidate-scoring head. Candidates are evaluated jointly in one encoder
pass rather than by independent prompting.

**Evidence.** Scotty-1 reaches 83.03% raw and 76.39% position-balanced accuracy
on a protected three-family Beam holdout. A matched-update Typed-only control
reaches 56.34% and 51.64%, respectively. The control is included to make the
coverage-specialization tradeoff explicit rather than claim universal gains.

**Usefulness.** The result is a downloadable local decision model with a small,
auditable output space, complete checkpoint, checksum, loader, paper, aggregate
results, and reproduction records.

## Ready-to-post discussion entry

> **Scotty-1 — compact typed decisions learned from Beam**
>
> Scotty-1 turns synthetic supervision from Reflection Beam 1.0 501B-A23B into
> a 421M-parameter local candidate-scoring model. Beam generated typed decisions
> across 18 task families; Scotty-1 jointly scores all candidate answers in one
> ModernBERT encoder pass instead of generating free-form text.
>
> On a protected holdout containing three entire Beam-generated task families,
> Scotty-1 reaches 83.03% raw and 76.39% position-balanced accuracy. The repo
> includes a matched-update Typed-only control, full checkpoint, checksum,
> loading instructions, aggregate results, reproducibility documentation, and a
> six-page paper. We report the single-seed and retrospective-test limitations
> directly.
>
> 🤗 Model: https://huggingface.co/DarwinGuru/scotty-1
>
> 💻 Source and paper: https://github.com/evolution-strategies/scotty-1

## Final publication sequence

1. Confirm entrant/team identity and eligibility under the official rules.
2. Resolve the rights and hosted-inference questions in `LICENSE_AUDIT.md` and
   confirm the competition's promotional license is acceptable.
3. Run the repository verifier, tests, history/secret scan, checkpoint checksum,
   and a fresh anonymous-read test.
4. Update release-state wording and provenance in one final commit.
5. With explicit approval, make the GitHub repository public.
6. With explicit approval, make `DarwinGuru/scotty-1` public.
7. Confirm both URLs load anonymously and that the model files download.
8. With explicit approval, post the prepared entry to the official discussion.
9. Record the public commits, timestamps, and posted discussion URL.

Steps 5, 6, and 8 are external publication actions and are intentionally not
performed by preparation work alone.
