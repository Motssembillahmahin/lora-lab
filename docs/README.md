# Docs

All documentation for the LoRA Lab lives here and is written **incrementally**
as the project progresses.

## Structure

- **`journal.md`** — running narrative. What we tried each session, what
  happened, what was learned, what's next. Append-only.
- **`math/`** — mathematical deep-dives. The core of this repo.
  - `01-lora-derivation.md` — the low-rank update, derived (to write)
  - `02-rank-and-alpha.md` — what r and alpha actually control (to write)
  - `03-loss-masking.md` — why we mask prompt tokens in instruction tuning (to write)
  - `04-quantization-nf4.md` — QLoRA's NF4 + double quantization, conceptually (to write)
  - `05-optimizer-memory.md` — where training memory actually goes (to write)
- **`decisions/`** — ADRs. One numbered file per meaningful choice.
  - `0001-use-uv-and-makefile.md` — tooling decision (seeded)
  - `0002-cpu-only-plain-lora.md` — why no local QLoRA (seeded)

## Conventions

- Math in markdown with LaTeX notation (`$...$`, `$$...$$`).
- ADRs use Context / Decision / Consequences.
- When code and its explanation change, they land in the same commit.
