# LoRA Lab

A hands-on learning project for mastering LoRA / QLoRA fine-tuning — from the
math up through a working training pipeline — on a CPU-only machine.

> **This is a LEARNING repository, not a production system.** The goal is depth
> of understanding, not shipping a model. Optimize every decision for what it
> teaches me, not for raw efficiency.

---

## Who I am / how to work with me

I'm a senior backend software engineer (Python, ~4 yrs). I learn best when:

- You **teach me different ways to do the same thing** and explain the tradeoffs,
  rather than handing me one blessed answer.
- You **leave room for me to find my own approach** — pose the problem, sketch
  the options, and let me choose. Don't over-prescribe.
- You connect new ideas back to engineering concepts I already know (distributed
  systems, async, databases, infra) where the analogy is honest.
- You go deep on **the math**. I want to genuinely understand the linear algebra
  and optimization behind LoRA, not just call `get_peft_model`. Derive things.
  Show the matrix shapes. Don't hand-wave.

Treat me as a capable peer. Push back when I'm wrong. Don't flatter.

## Environment (hard constraints)

- Machine: **AMD Ryzen 5 PRO 5650U, 6c/12t, 14 GB RAM, CPU-only (no CUDA/ROCm)**.
- Always use the **CPU PyTorch build**.
- **Never suggest bitsandbytes or true 4-bit QLoRA** for local runs — no GPU.
  We *study* QLoRA's math and mechanics conceptually, and can demonstrate the
  NF4 / double-quant ideas in small numpy/torch experiments, but real QLoRA
  training is out of scope locally. Note this honestly whenever it comes up.
- Keep models small (Qwen2.5-0.5B is the default), `MAX_LEN` short, and dataset
  slices tiny so runs finish in minutes on CPU.

## Tooling conventions (non-negotiable)

- **`uv` is the package and project manager.** Never use bare `pip` or
  `poetry`. Use `uv add`, `uv run`, `uv sync`, `uv venv`. The project is defined
  by `pyproject.toml` + `uv.lock`.
- **A `Makefile` is the single entry point for every command.** Anything I'd run
  more than once (setup, train, infer, merge, lint, format, test, docs) gets a
  make target. I should be able to discover the whole workflow via `make help`.
- Project layout follows standard Python conventions (`src/` layout, configs in
  `configs/`, docs in `docs/`). Keep it clean — I care about repo hygiene.

## Documentation discipline (incremental)

- **`/docs` is the home for all documentation.** Everything we learn or decide
  gets written down there, incrementally, as we go — not in one big dump at the
  end.
- `docs/math/` holds the mathematical deep-dives (LoRA derivation, rank/alpha,
  quantization, optimizer memory math). Use LaTeX-style notation in markdown.
- `docs/decisions/` holds short ADRs (Architecture Decision Records) — one file
  per meaningful choice, numbered, with context / decision / consequences.
- `docs/journal.md` is the running narrative: what we tried, what happened, what
  I learned. Append to it every session.
- `experiments/log.md` is the structured run log: one entry per training run,
  each tied to a git commit SHA.
- When you implement or explain something new, **update the relevant doc in the
  same change.** Code and its explanation land together.

## Agents (multi-agent workflow)

I want to work with **several specialized subagents**, defined in
`.claude/agents/`, created as the project needs them. Don't dump them all at
once — introduce an agent when a real task calls for it, and document why in an
ADR. Starting set (already scaffolded, refine as needed):

- `math-tutor` — derives and explains the math; read-only, no code edits.
- `experiment-runner` — proposes a config diff, runs training, reports metrics.
- `doc-writer` — keeps `/docs` and the journal current; read + write to docs.
- `code-reviewer` — reviews diffs for repo hygiene and correctness; read-only.

Follow the 2026 convention: each agent is a markdown file with YAML frontmatter
(`name`, `description`, `tools`, optional `model`). Give research/review agents
**read-only tool sets** and let the parent session do the edits.

## Conventions

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`.
- All hyperparameters live in `configs/*.yaml`. Change params there, not in code.
- Outputs (adapters, checkpoints, datasets) go to gitignored dirs. Never commit
  weights.
- Every experiment: branch -> edit config -> `make train` -> log result with SHA
  -> commit. Git history should read like a learning path.

## First tasks for Claude Code (when I open this repo)

1. Read this file and `README.md`, then `git init` and make the first commit of
   the scaffold (do NOT commit `data/`, `outputs/`, or the venv).
2. Initialize the project with `uv` (`uv init` semantics already partly set up in
   `pyproject.toml`); create the venv and sync deps via the Makefile.
3. Walk me through the repo structure and the Makefile targets before running
   anything.
4. Then ask me which thread I want to pull first: the **math** (start
   `docs/math/01-lora-derivation.md`) or the **first training run**.

## Don't

- Don't commit anything in `data/`, `outputs/`, or the virtual env.
- Don't switch to GPU-only libraries or pretend QLoRA runs on this CPU.
- Don't give me one answer when there are interesting alternatives — show them.
- Don't skip the math to get to working code faster. The math is the point.
