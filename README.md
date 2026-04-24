# attention-lab

A learning repo for building deep, rigorous understanding of attention mechanisms through **theory + experiments**.

Two parallel tracks:

- **`theory/`** — assignment-style documents. Definitions, theorems, proofs, and exercises. Work them with pen and paper (or markdown).
- **`notebooks/`** — experimental Jupyter notebooks. Implement, visualize, perturb, and learn by playing.

Every theory chapter has a matching notebook. Do the theory first; then the notebook should feel like confirmation, not revelation.

## How to use this repo

1. Read the theory chapter end-to-end.
2. Do the exercises *before* looking at anything else. Check your work against the solution notes.
3. Open the matching notebook. Run it cell by cell. Change parameters. Break it.
4. Write 2–3 sentences in `learnings.md` about what surprised you.

Do not skip exercises. The plots are easy; the derivations are where understanding forms.

## Curriculum

| # | Theory | Notebook | Core question |
|---|--------|----------|---------------|
| 0 | [Linear algebra primer](theory/00_linear_algebra_primer.md) | — | Are your prereqs solid? |
| 1 | [Scaled dot-product attention](theory/01_scaled_dot_product.md) | [01_attention_by_hand.ipynb](notebooks/01_attention_by_hand.ipynb) | What is Q·Kᵀ actually measuring? |
| 2 | [Softmax & temperature](theory/02_softmax.md) | 02_softmax_temperature.ipynb | How does softmax concentrate mass? |
| 3 | [Why √dₖ?](theory/03_why_sqrt_dk.md) | 03_variance_scaling.ipynb | Why does unscaled attention saturate? |
| 4 | Multi-head attention | 04_heads_specialize.ipynb | Do heads learn different things? |
| 5 | Positional encodings | 05_positional_bakeoff.ipynb | Which encodings extrapolate? |
| 6 | Induction heads | 06_induction_heads.ipynb | Can you watch a circuit form? |
| 7 | Attention sinks | 07_attention_sinks.ipynb | Why does token 0 hoard attention? |
| 8 | Logit lens & patching | 08_logit_lens.ipynb | How do predictions refine by layer? |

Chapters 4+ are stubs — we'll fill them as you progress.

## Setup

```bash
cd ~/attention-lab
uv venv
uv pip install -e ".[dev]"
uv run jupyter lab   # or open notebooks in VS Code
```

## Conventions

- Every notebook starts with a markdown cell: **Question**, **Hypothesis**, **Method**.
- Every notebook ends with a markdown cell: **Finding** (2–3 sentences).
- Copy findings into `learnings.md` with date + notebook link.
