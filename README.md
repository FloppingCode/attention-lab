# attention-lab

Staged curriculum from attention mechanics to flow matching and diffusion on manifolds. Theory documents and executable notebooks, governed by an explicit pedagogy contract.

## Governing documents

- [`LEARNING_SPEC.md`](./LEARNING_SPEC.md) — pedagogy contract. Chapter structure, exercise standard, notebook conventions, content style. Read first.
- [`CURRICULUM.md`](./CURRICULUM.md) — the full 8-phase, 34-chapter map with dependencies. Read second.

The `AI-applied-math-tutor` skill (global, `~/.claude/skills/AI-applied-math-tutor/`) supplies the learner profile and default pedagogy. The spec overrides where they differ.

## Layout

All chapter content lives under [`phases/`](./phases/), one directory per curriculum phase:

```
phases/phase-<N>-<slug>/
  README.md       # phase overview, chapter table, experiment hook
  theory/         # chapter markdown files (when present)
  notebooks/      # paired .ipynb files (when present)
```

- **Theory documents** declare a type (`theory-only`, `theory + notebook`, `experiment-led`) and ordering (`theory-first`, `notebook-first`, `mixed`) in their header.
- **Notebooks** follow QHMPC / F: **Question, Hypothesis, Method, Prediction, Confidence** at the top; **Finding** at the bottom.
- **`learnings.md`** — rolling log: `YYYY-MM-DD · [chapter/notebook] · 2–3 sentences on what surprised, clicked, or got updated`.

Exercises ship without solutions — the reader works them.

## How to work a chapter

1. Read the chapter header — note its type and ordering.
2. If theory-first: read end-to-end, do the exercises, write predictions (§X.7) **before opening the notebook**.
3. If notebook-first: open the notebook, state a naive hypothesis in the QHMPC block, run, be surprised, then read the chapter.
4. Check your predictions against the Findings in the notebook.
5. Append 2–3 sentences to `learnings.md`.

## Curriculum overview

Full map in [`CURRICULUM.md`](./CURRICULUM.md). Summary by phase:

| Phase | Top-level question | Chapters | Status |
|---|---|---|---|
| **0** Conventions | What notation does this repo use? | 0 | new |
| **I** Attention mechanics | What is one attention computation doing? | 1–3 | complete |
| **II** Transformer internals | How do the pieces compose into a block? | 4–9 | mix of stubs and new |
| **III** Training dynamics | How do you read a loss curve and design an ablation? | 10–14 | new (experiment-led) |
| **IV** Interpretability | How do you see a trained circuit? | 15–17 | stubs (experiment-led) |
| **V** Bridge to generation | How does a transformer define a distribution? | 18–20 | new |
| **VI** Flow matching and diffusion | The research layer. | 21–28 | new |
| **VII** Geometric / manifold-valued | Attention and generation on manifolds. | 29–34 | new (research-frontier) |

**Chapter markers.** `✓` complete · `◐` drafted, audit pending · `stub` placeholder · `new` not yet written · `cut` removed.

## Phase I — existing files

| # | Theory | Notebook | Status |
|---|---|---|---|
| 1 | [Scaled dot-product attention](phases/phase-1-attention-mechanics/theory/01_scaled_dot_product.md) | [01_attention_by_hand.ipynb](phases/phase-1-attention-mechanics/notebooks/01_attention_by_hand.ipynb) | ✓ |
| 2 | [Softmax & temperature](phases/phase-1-attention-mechanics/theory/02_softmax.md) | — | ✓ |
| 3 | [Why $\sqrt{d_k}$?](phases/phase-1-attention-mechanics/theory/03_why_sqrt_dk.md) | [03_variance_scaling.ipynb](phases/phase-1-attention-mechanics/notebooks/03_variance_scaling.ipynb) | ✓ |

Chapters 4+ are tracked in `CURRICULUM.md` and scaffolded as empty phase directories under [`phases/`](./phases/). Written on demand in dependency order.

## Setup

```bash
cd ~/attention-lab
uv venv
uv pip install -e ".[dev]"
uv run jupyter lab       # or open .ipynb files directly in VS Code
```

## Cut from earlier scaffold

- `phases/phase-0-conventions/theory/00_linear_algebra_primer.md` — **cut.** The reader has measure-theoretic / functional-analytic fluency; a primer is inappropriate. Replaced by Phase 0 "Conventions & notation" (see `CURRICULUM.md`).
