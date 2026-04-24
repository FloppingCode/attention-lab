# Phase VI — Flow matching and diffusion

**Top-level question:** What are flow matching and diffusion models mathematically, why is the training objective what it is, and what does the transformer contribute vs. what does the SDE/ODE structure contribute?

See [CURRICULUM.md](../../CURRICULUM.md#phase-vi--flow-matching-and-diffusion-the-research-layer).

## Chapters

### VI.a — Forward/reverse structure
| # | Title | Type | Status |
|---|---|---|---|
| 21 | Forward SDEs and noise schedules | T | new |
| 22 | The reverse-time theorem | T | new |

### VI.b — Training objectives
| # | Title | Type | Status |
|---|---|---|---|
| 23 | Score matching and its variants | T+N | new |
| 24 | Flow matching: regression, not score | T+N | new |
| 25 | Guidance: classifier and classifier-free | T+N | new |

### VI.c — Sampling and architecture
| # | Title | Type | Status |
|---|---|---|---|
| 26 | Sampler zoo: ODE vs. SDE vs. distillation | E | new |
| 27 | The network's role vs. the dynamics' role | T+N | new |
| 28 | Transformers as diffusion backbones | T+N | new |

**Experiment hook:** Reproduce a minimal diffusion / flow-matching result on a toy 2D distribution. Predict failure modes under sampler, schedule, and network-capacity ablations before running.

_Empty — no chapters drafted yet._
