# Phase II — Attention deployed

**Top-level question:** Attention is one piece of a transformer block. What wraps it (masking, heads, positional info, norms, residuals, FFN), why, and how do those pieces behave differently at training vs. inference?

See [CURRICULUM.md](../../CURRICULUM.md#phase-ii--attention-deployed-how-is-attention-actually-used-in-a-block).

## Chapters

| # | Title | Theory | Notebook | Type | Ord |
|---|---|---|---|---|---|
| 4 | Causal masking and attention at inference | [04_causal_masking.md](theory/04_causal_masking.md) | [04_causal_masking.ipynb](notebooks/04_causal_masking.ipynb) | T+N | TF |
| 5 | Multi-head attention | [05_multi_head.md](theory/05_multi_head.md) | [05_heads_specialize.ipynb](notebooks/05_heads_specialize.ipynb) | T+N | M |
| 6 | Positional encodings | [06_positional_encodings.md](theory/06_positional_encodings.md) | [06_positional_bakeoff.ipynb](notebooks/06_positional_bakeoff.ipynb) | T+N | TF |
| 7 | Layer norm and residuals | [07_layer_norm_residuals.md](theory/07_layer_norm_residuals.md) | [07_norm_and_residuals.ipynb](notebooks/07_norm_and_residuals.ipynb) | T+N | TF |
| 8 | The feedforward block | [08_feedforward.md](theory/08_feedforward.md) | [08_ffn_capacity.ipynb](notebooks/08_ffn_capacity.ipynb) | T+N | TF |
| 9 | Stacking: the residual stream view | [09_residual_stream.md](theory/09_residual_stream.md) | [09_residual_stream.ipynb](notebooks/09_residual_stream.ipynb) | T+N | M |

**Experiment hook:** Ablate one component at a time; predict the training-loss consequence before running.

## Recommended reading order

Chapters can be read independently after Phase I, but the natural progression is:

1. **Ch 4 (causal masking)** — extends Ch. 1's permutation-equivariance discussion to the autoregressive setting; sets up the train/inference asymmetry.
2. **Ch 5 (multi-head)** — orthogonal to Ch. 4; introduces head specialization.
3. **Ch 6 (positional)** — independent of 4 and 5; positional encoding is a separate axis of design.
4. **Ch 7 (LN + residuals)** — load-bearing for everything that follows. Pre-norm vs. post-norm is the single biggest stability lesson.
5. **Ch 8 (FFN)** — depends on Ch. 7 (uses the residual + LN scaffolding). Establishes the FFN-as-key-value-memory view.
6. **Ch 9 (residual stream)** — the synthesis chapter. Reframes the entire transformer as a running sum, and motivates the mechanistic-interpretability tools in Phase IV.
