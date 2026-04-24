# Curriculum — attention-lab

Staged curriculum from attention mechanics to the reader's research target (flow matching and diffusion models on manifolds). Each phase answers one top-level question. Each chapter declares: **core question**, **type** (theory-only / theory+notebook / experiment-led), **ordering** (theory-first / notebook-first / mixed), **status**, **depends on**.

Governed by [`LEARNING_SPEC.md`](./LEARNING_SPEC.md) and the global `AI-applied-math-tutor` skill. Every chapter here must satisfy the chapter contract in §2 of the spec.

**Status markers**
- `✓` — exists in current repo (may still need audit against the new spec)
- `stub` — placeholder in README, not written
- `new` — added by this curriculum
- `cut` — removed from prior curriculum

**Type key:** `T` = theory-only · `T+N` = theory + notebook · `E` = experiment-led
**Ordering key:** `TF` = theory-first · `NF` = notebook-first · `M` = mixed

---

## Phase 0 — Conventions

**Top-level question:** What conventions and notation does this repo use?

| # | Title | Type | Ord | Status | Deps |
|---|---|---|---|---|---|
| 0 | Conventions & notation | T | — | new | — |

Short reference. Index and shape conventions ($n$ tokens, $d$ embed, $d_k$ key, $h$ heads; rows are tokens), norms and inner products, notation for train / val / inference regimes, solution-file conventions. Replaces the cut linear-algebra primer.

---

## Phase I — Attention mechanics: what is one attention computation doing?

**Top-level question:** Given $\text{softmax}(QK^\top / \sqrt{d_k}) V$, what does each component do, why is it there, and what are the sharp mathematical facts about it?

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 1 | Scaled dot-product attention | What is $QK^\top$ measuring, and what does softmax-then-mix compute? | T+N | TF | ✓ | 0 |
| 2 | Softmax & temperature | How does softmax concentrate mass, and how does temperature interpolate between argmax and uniform? | T+N | TF | ✓ | 1 |
| 3 | Why $\sqrt{d_k}$? | Why does unscaled attention saturate as $d_k$ grows, and why is $\sqrt{d_k}$ the right correction? | T+N | TF | ✓ | 1, 2 |

**Phase-I experiment hook:** Given a sequence and chosen $W_Q, W_K, W_V$, predict the attention matrix pattern before running it.

---

## Phase II — Attention deployed: how is attention actually used in a block?

**Top-level question:** Attention is one piece of a transformer block. What wraps it (masking, heads, positional info, norms, residuals, FFN), why, and how do those pieces behave differently at training vs. inference?

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 4 | Causal masking and attention at inference | What does the causal mask enforce, and how does attention differ between training (parallel) and autoregressive inference (sequential, KV-cached)? | T+N | TF | new | 1 |
| 5 | Multi-head attention | What do multiple heads buy that a single larger head cannot? | T+N | M | stub | 1, 3 |
| 6 | Positional encodings | How does a permutation-equivariant mechanism encode position, and which encodings extrapolate? | T+N | TF | stub | 1 |
| 7 | Layer norm and residuals | What do LN and residual streams do to gradients and activations, and why pre-norm vs. post-norm? | T+N | TF | new | 1 |
| 8 | The feedforward block | What does the per-token MLP do that attention can't, and why is it wider than $d$? | T+N | TF | new | 1, 7 |
| 9 | Stacking: the residual stream view | How do layers interact along the residual stream, and why is "the residual stream is the communication channel" load-bearing? | T+N | M | new | 7, 8 |

**Phase-II experiment hook:** Ablate one component at a time (remove LN, swap pre/post-norm, halve FFN width); predict the training-loss consequence before running.

---

## Phase III — Training dynamics and empirical maturity

**Top-level question:** How do you read a loss curve, design an ablation, and tell a real effect from a seed artifact?

Chapters here lean heavier on the empirical loop than on rigorous cores. Most are experiment-led.

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 10 | Reading a loss curve | What do characteristic loss-curve shapes (spikes, plateaus, divergence, grokking hints) indicate about the underlying dynamics? | E | NF | new | II |
| 11 | Initialization and signal propagation | Why does init scale matter, and what goes wrong at train-time when the variance budget is broken? | T+N | TF | new | 7 |
| 12 | Optimizers for transformers | Why AdamW and not SGD — what does Adam's preconditioning do for attention's geometry? Where do LR schedules come from? | T+N | M | new | 10 |
| 13 | Ablation design | How do you design an ablation that actually isolates a mechanism, and how do you read a noisy comparison? | E | NF | new | 10 |
| 14 | Scaling laws and what they imply | What do empirical scaling laws assert, what are the caveats, and what do they imply for experimental design? | T | — | new | 10, 12 |

**Phase-III experiment hook:** Given a deliberately buggy training setup, predict from the loss curve what is wrong before checking the code.

---

## Phase IV — Interpretability: how do you see the mechanism working?

**Top-level question:** Given a trained transformer, how do you read its internals as falsifiable claims about learned circuits?

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 15 | Induction heads | Can you watch an in-context-learning circuit form, and what does its phase transition look like? | E | NF | stub | 9, 13 |
| 16 | Attention sinks | Why does token 0 hoard attention mass, and what does this say about softmax's zero-option problem? | E | NF | stub | 2, 9 |
| 17 | Logit lens and activation patching | How do predictions refine by layer, and how do you causally isolate where a behavior lives? | T+N | M | stub | 9 |

**Phase-IV experiment hook:** Predict which layer / head / position a patch will localize a behavior to, before running the patch.

---

## Phase V — Attention as a substrate for generative modeling

**Top-level question:** How does a transformer parameterize a distribution, and what does "generation" mean formally across autoregressive, masked, and continuous-time regimes?

Bridge to Phase VI. Deliberately light on new transformer internals; heavy on linking to the reader's existing fluency in measure theory and SDEs.

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 18 | Autoregressive transformers as distributions | What measure does a decoder-only transformer define over sequences, and what divergence does cross-entropy minimize? | T | — | new | 4, 9 |
| 19 | Score and velocity parameterizations | What does the network represent in a score-based or flow-matching model — score field, velocity field, or denoiser? How does conditioning enter? | T | — | new | 9 |
| 20 | Sampling is inference with dynamics | How does the sampler (Langevin, probability-flow ODE, CFM integration) interact with the trained network? Where do the engineering failure modes live? | T+N | TF | new | 19 |

---

## Phase VI — Flow matching and diffusion (the research layer)

**Top-level question:** What are flow matching and diffusion models mathematically, why is the training objective what it is, and what does the transformer contribute vs. what does the SDE/ODE structure contribute?

This is where SDE / measure-theory / optimization fluency gets used at strength. Expanded from the earlier draft — each sub-topic deserves its own chapter.

### VI.a — The forward/reverse structure

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 21 | Forward SDEs and noise schedules | What does the forward process destroy, and how do VE / VP / sub-VP / rectified-flow schedules differ structurally? | T | — | new | 19 |
| 22 | The reverse-time theorem | Anderson's reverse-time SDE and the probability-flow ODE: what does each say, and what does the network have to approximate? | T | — | new | 21 |

### VI.b — Training objectives

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 23 | Score matching and its variants | How does DSM approximate the intractable score, and what's the bias–variance story across variants (sliced, target, $v$-parameterization)? | T+N | TF | new | 22 |
| 24 | Flow matching: regression, not score | What is the flow-matching objective, and why is conditional flow matching the workable form? | T+N | TF | new | 22 |
| 25 | Guidance: classifier and classifier-free | Where does guidance come from formally, and why does CFG work despite its heuristic origin? | T+N | M | new | 23, 24 |

### VI.c — Sampling and architecture

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 26 | Sampler zoo: ODE vs. SDE vs. distillation | How do Euler, Heun, DPM-Solver, and their stochastic counterparts compare in compute / quality / mode coverage? | E | NF | new | 22, 24 |
| 27 | The network's role vs. the dynamics' role | Disentangle what the data manifold, the stochastic process, and the network each contribute. What would a bad network *still* get right thanks to the dynamics? | T+N | M | new | 23, 24, 26 |
| 28 | Transformers as diffusion backbones | DiT and variants: what does attention buy over U-Net, and where does the tokenization boundary sit? | T+N | M | new | II, 27 |

**Phase-VI experiment hook:** Reproduce a minimal diffusion / flow-matching result on a toy 2D distribution. Predict failure modes under sampler, schedule, and network-capacity ablations before running.

---

## Phase VII — Geometric and manifold-valued mechanisms

**Top-level question:** What breaks when the data, the latents, or the attention mechanism lives on a non-Euclidean manifold, and what are the principled fixes?

Expanded with proper subdivision. The three problems are distinct.

### VII.a — Data on manifolds

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 29 | Brownian motion and the heat equation on Riemannian manifolds | What's the right analog of the forward SDE when data lives on a manifold, and what role does the Laplace–Beltrami operator play? | T | — | new | 21 |
| 30 | Diffusion and flow matching on manifolds | Riemannian flow matching, stereographic CFM, geodesic interpolation: how do the Euclidean training objectives generalize? | T+N | M | new | 24, 29 |
| 31 | Sampling on manifolds | Exponential-map integrators, retraction-based samplers, and their stability under step-size choice. | E | NF | new | 30 |

### VII.b — Attention on non-Euclidean key spaces

| # | Title | Core question | Type | Ord | Status | Deps |
|---|---|---|---|---|---|---|
| 32 | Attention beyond inner products | What replaces $QK^\top$ when the key space is Riemannian? What are the geodesic-kernel, log-map, and bilinear-on-tangent-space options, and where does each break? | T+N | M | new | 1, 5 |
| 33 | Equivariant attention | How do you build attention that respects a group action (SO(3), permutations beyond tokens, gauge symmetries)? What does equivariance buy, and what does it cost? | T+N | M | new | 32 |

### VII.c — Open problems

| # | Title | Content | Status |
|---|---|---|---|
| 34 | Research frontier: questions and spike ideas | Living document. Each entry: a question the reader is positioned to attack, a short literature pointer, and a concrete spike experiment. Not a chapter in the normal sense. | new |

---

## Cut / deprioritized

| Was | Status | Reason |
|---|---|---|
| `00_linear_algebra_primer.md` | **cut** | Reader has measure-theoretic / functional-analytic fluency. Replaced by §0 Conventions. |

---

## Chapter-type summary

| Type | Count | Phases |
|---|---|---|
| Theory-only (T) | 9 | 0, 14, 18, 19, 21, 22, 29 (+ 34 as a living doc) |
| Theory + notebook (T+N) | 18 | I, II, III, IV, V (part), VI, VII |
| Experiment-led (E) | 7 | III, IV, VI.c, VII.a |

**Rough totals:** 34 chapters across 8 phases. Phases I–II are ~1–2 weeks of engaged work each. Phase III is the training-dynamics bottleneck — budget a month. Phase VI is the research depth — expect several months, with chapters 23–28 each deserving multiple passes.

---

## What to do before writing any new chapter

1. Declare type and ordering in the chapter header.
2. Write the core question in one sentence.
3. Write the **storyline** (spec §2) first — can you narrate the mechanism end-to-end before writing any math?
4. Write two exercises — one mechanism-dissection, one predict-then-verify. If you can't, the chapter isn't ready.
5. Only then draft the rigorous core and numerical example.
6. For a notebook-first chapter: draft the **Question, naive Hypothesis, and Prediction** first. If the prediction is vague, the notebook isn't ready.

## Revision policy

Directional, not frozen. Phases VI and VII will likely grow as you push into them. Revisions must preserve the phase structure and the dependency graph. Log curriculum changes in `learnings.md`.
