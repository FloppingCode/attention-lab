# Phase I — Attention mechanics

**Top-level question:** Given $\text{softmax}(QK^\top / \sqrt{d_k}) V$, what does each component do, why is it there, and what are the sharp mathematical facts about it?

See [CURRICULUM.md](../../CURRICULUM.md#phase-i--attention-mechanics-what-is-one-attention-computation-doing).

## Chapters

| # | Title | Type | Status |
|---|---|---|---|
| 1 | Scaled dot-product attention | T+N | ✓ |
| 2 | Softmax & temperature | T+N | ✓ |
| 3 | Why $\sqrt{d_k}$? | T+N | ✓ |

- Theory: [`theory/`](./theory/)
- Notebooks: [`notebooks/`](./notebooks/)

**Experiment hook:** Given a sequence and chosen $W_Q, W_K, W_V$, predict the attention matrix pattern before running it.
