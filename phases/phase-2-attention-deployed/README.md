# Phase II — Attention deployed

**Top-level question:** Attention is one piece of a transformer block. What wraps it (masking, heads, positional info, norms, residuals, FFN), why, and how do those pieces behave differently at training vs. inference?

See [CURRICULUM.md](../../CURRICULUM.md#phase-ii--attention-deployed-how-is-attention-actually-used-in-a-block).

## Chapters

| # | Title | Type | Status |
|---|---|---|---|
| 4 | Causal masking and attention at inference | T+N | new |
| 5 | Multi-head attention | T+N | stub |
| 6 | Positional encodings | T+N | stub |
| 7 | Layer norm and residuals | T+N | new |
| 8 | The feedforward block | T+N | new |
| 9 | Stacking: the residual stream view | T+N | new |

**Experiment hook:** Ablate one component at a time; predict the training-loss consequence before running.

_Empty — no chapters drafted yet._
