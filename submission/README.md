# Build with Beam submission package

This directory contains a concise draft for the competition discussion. It does
not post or submit anything.

## Official call

- Competition discussion: <https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5>
- Official rules: <https://reflection.ai/legal/official-rules-build-with-beam>
- Deadline: **2 November 2026 at 11:59 PM Pacific Time**.
- Required entry: a publicly accessible Hugging Face Space, model, or dataset;
  documentation of the project and Beam's role; and a discussion link before
  the deadline.

The official rules control if this summary differs from them.

## Submission assets

- Model and documentation: <https://huggingface.co/DarwinGuru/scotty>
- Inference source: <https://huggingface.co/DarwinGuru/scotty/blob/main/run_scotty.py>
- Paper: <https://huggingface.co/DarwinGuru/scotty/blob/main/paper/scotty-paper.pdf>
- Demo: <https://spark-29fd.taile0f6e8.ts.net>

## Ready-to-post discussion entry

> **Scotty — compact decisions learned from Beam**
>
> Scotty turns typed supervision from Reflection Beam 1.0 501B-A23B into a
> 421-million-parameter local decision model. Given a situation, question, and
> candidate answers, it scores all candidates jointly in one ModernBERT forward
> pass instead of generating free-form text.
>
> Beam generated decisions across 18 task families. Scotty trains with effective
> batches split equally between Beam-generated examples and Typed Decisions.
> Beam supplies supervision, not model weights. The generation calls stayed
> within the $25 participant credit budget, excluding local training and hardware.
>
> Across three seeds, Scotty's mean validation objective is
> 0.859532 ± 0.004603. On a group-safe source audit, the seed-0 model reaches
> objective 0.841031, 76.67% Typed hard accuracy, 88.62% Beam raw accuracy, and
> 91.54% Beam position-balanced accuracy. The audit uses withheld groups from
> the same source collections and is not an external benchmark.
>
> 🤗 Model: https://huggingface.co/DarwinGuru/scotty
>
> 💻 Inference code: https://huggingface.co/DarwinGuru/scotty/blob/main/run_scotty.py
>
> 📄 Paper: https://huggingface.co/DarwinGuru/scotty/blob/main/paper/scotty-paper.pdf
>
> 🧪 Demo: https://spark-29fd.taile0f6e8.ts.net (login-protected; access available on request)

## Before posting

1. Confirm entrant identity and eligibility.
2. Confirm the competition's promotional terms are acceptable.
3. Run repository tests, checkpoint verification, and a fresh anonymous download.
4. Confirm the public Hugging Face model, source, and paper are accessible to
   judges; arrange demo access separately.
5. Change visibility only with explicit approval.
6. Post the discussion entry only with explicit approval.
