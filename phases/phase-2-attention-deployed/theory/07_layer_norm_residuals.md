# Chapter 7 — Layer norm and residuals

```
Type: theory + notebook
Ordering: theory-first
Depends on: 1 (attention)
```

## 7.1 Motivation

A transformer block is not just attention. Around the attention (and around the FFN, Ch. 8) sits a pair of layer-normalization-plus-residual operations. They look like trivial scaffolding — "just normalize, just add" — and that view is wrong. Layer norm controls the *variance budget* through the depth of the network; residuals are the *communication channel* that gives every layer access to every prior representation. Together they decide whether a deep transformer trains at all and whether its representations remain interpretable. The placement of layer norm relative to the residual (pre-norm vs. post-norm) was the single biggest stability lesson of the post-2018 transformer era.

## 7.2 Storyline

**Layer norm.** For input $x \in \mathbb{R}^d$, layer norm computes
$$\text{LN}(x) = \gamma \odot \frac{x - \mu(x)}{\sigma(x)} + \beta,$$
with $\mu(x), \sigma(x)$ the mean and standard deviation across the $d$ feature coordinates of a single token, and $\gamma, \beta \in \mathbb{R}^d$ learned. The point is: each token's representation is rescaled to have zero mean, unit variance per token, then re-affine-transformed. This stabilizes the *scale* of activations and gradients independently of batch size — unlike batch norm, which mixes statistics across tokens.

**Residual connections.** Each sub-layer (attention, FFN) is wrapped: $x_{\text{out}} = x + \text{SubLayer}(x)$. The unmodified input is passed forward in addition to the sub-layer's output. Mathematically: the network output is the sum of contributions from many depths, not the composition of many depths.

**Pre-norm vs. post-norm.** Two placements:

- **Post-norm** (original 2017 transformer): $x_{\text{out}} = \text{LN}(x + \text{SubLayer}(x))$. The residual stream is normalized after addition.
- **Pre-norm** (modern default): $x_{\text{out}} = x + \text{SubLayer}(\text{LN}(x))$. The sub-layer sees a normalized input, but the residual stream itself is *not* normalized after addition.

The difference is structural and load-bearing. **Pre-norm is dramatically more stable for deep training** — post-norm transformers often need careful warmup to avoid divergence, while pre-norm trains stably to hundreds of layers. Empirical observation 7.4 below.

**The residual stream as a communication channel.** Because each block adds (rather than transforms) its output, the input embedding flows through to the output, bumped at each layer by a small contribution. Mechanistic interpretability (Chapter 17) builds on this: the *residual stream* is a vector space in which each layer reads from and writes to specific subspaces. Heads and FFN blocks become "writers" to particular features in the residual stream; downstream layers become "readers." This decomposition only works because of the additive (residual) structure.

**Train / val / inference.** Layer norm has no train/eval distinction (unlike batch norm). It computes the same statistics in all regimes — per-token mean and std. This is part of why transformers behave consistently across batch sizes and deployment regimes. The $\gamma, \beta$ parameters are learned at train time and frozen thereafter.

**Aside on RMSNorm.** A simplified variant: $\text{RMSNorm}(x) = \gamma \odot x / \text{RMS}(x)$, with $\text{RMS}(x) = \sqrt{\tfrac{1}{d} \sum_i x_i^2}$. No mean centering, no $\beta$. Cheaper (fewer ops, no need to compute mean), and empirically similar to LayerNorm for transformers — most modern LLMs use RMSNorm.

## 7.3 Rigorous core

**Definition 7.1 (Layer normalization).** Let $x \in \mathbb{R}^d$. Define $\mu(x) = \tfrac{1}{d} \sum_i x_i$, $\sigma^2(x) = \tfrac{1}{d} \sum_i (x_i - \mu(x))^2$. Then
$$\text{LN}(x) = \gamma \odot \frac{x - \mu(x) \mathbf{1}}{\sqrt{\sigma^2(x) + \epsilon}} + \beta,$$
with $\gamma, \beta \in \mathbb{R}^d$ learned and $\epsilon > 0$ a small numerical-stability constant.

**Proposition 7.2 (LN invariances).** Layer norm is invariant to (a) shifts in the input $x \to x + c \mathbf{1}$ for any scalar $c$, and (b) positive rescalings $x \to \lambda x$ for $\lambda > 0$ when $\beta = 0$. (When $\beta \ne 0$, scale-invariance is broken at the affine output.)

**Proof.** Shift: $x + c \mathbf{1}$ has mean $\mu(x) + c$ and the same variance, so $(x + c\mathbf{1}) - (\mu(x) + c) \mathbf{1} = x - \mu(x) \mathbf{1}$. Rescale: $\lambda x$ has mean $\lambda \mu(x)$ and variance $\lambda^2 \sigma^2(x)$, so the normalization recovers the same unit vector. $\beta$ then adds a fixed translation that is unaffected by the input. $\blacksquare$

**Proposition 7.3 (Residual gradient flow).** For a stack of $L$ residual blocks $x_{\ell+1} = x_\ell + f_\ell(x_\ell)$, the gradient of the loss $\mathcal{L}$ with respect to $x_0$ contains a term $\partial \mathcal{L} / \partial x_L$ — i.e., a *direct* gradient path from output to input independent of the depth $L$. Combined with the chain-rule terms from the $f_\ell$'s, this prevents the vanishing-gradient pathology of plain feed-forward stacks.

**Proof.** $\partial x_{\ell+1} / \partial x_\ell = I + \partial f_\ell / \partial x_\ell$. By the chain rule, $\partial \mathcal{L} / \partial x_0 = \prod_{\ell=0}^{L-1} (I + J_\ell) \cdot \partial \mathcal{L} / \partial x_L$, where $J_\ell = \partial f_\ell / \partial x_\ell$. Expanding the product, the leading term (all $I$'s) is the identity — gradient passes through unchanged. Other terms are products of various $J_\ell$'s. As long as the $J_\ell$ are reasonable (not all near zero or all near $-I$), the gradient survives. $\blacksquare$

**Empirical observation 7.4 (Pre-norm vs. post-norm stability).** Pre-norm transformers train stably without learning-rate warmup; post-norm transformers diverge without it, especially as depth grows. The reason is that the residual stream in post-norm passes through LN at every layer, repeatedly rescaling the gradient signal; in pre-norm, the residual stream is not normalized, so the gradient through the residual path is preserved exactly (Proposition 7.3 applies cleanly).

**Empirical observation 7.5 (Residual-stream norm growth).** In a trained pre-norm transformer, the norm of the residual stream tends to grow monotonically with depth — each layer adds a contribution. The norm at the final layer can be much larger than at the embedding. This is healthy if controlled by output LN (or RMSNorm) before the unembedding; without that, the output logits have unbounded scale.

**Proposition 7.6 (LayerNorm Jacobian — projection structure).** The Jacobian of $\text{LN}(x)$ at $x$ (with $\gamma = \mathbf{1}, \beta = 0$ for simplicity) is
$$\frac{\partial \text{LN}_i}{\partial x_j} = \frac{1}{\sigma(x)} \left[ \delta_{ij} - \frac{1}{d} - \frac{(x_i - \mu)(x_j - \mu)}{d \sigma^2(x)} \right].$$
This Jacobian has rank $d - 2$: there are two null directions, the constant direction $\mathbf{1}$ (LN is shift-invariant) and the direction along $x - \mu \mathbf{1}$ (LN is scale-invariant per token).

**Proof.** Direct computation. $\blacksquare$

**Corollary 7.6.1.** Two degrees of freedom per token are killed by layer norm: the mean and the scale. Information in those directions is lost. This is intentional — LN's job is to remove scale and offset variability — but it is a real loss, and the residual connection compensates by carrying the un-normalized input forward.

## 7.4 Numerical example — LN by hand on $d = 4$

Take $x = (1, 3, 5, 7)$. Then $\mu = 4$, $\sigma^2 = ((1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2)/4 = (9 + 1 + 1 + 9)/4 = 5$, $\sigma = \sqrt{5} \approx 2.236$. Centered: $(-3, -1, 1, 3)$. Normalized: $(-3, -1, 1, 3) / \sqrt{5} \approx (-1.342, -0.447, 0.447, 1.342)$.

With $\gamma = \mathbf{1}, \beta = \mathbf{0}$ that is the LN output. Verify: mean $= 0$, variance $= 1$. ✓

Now check shift invariance: $\text{LN}(x + 100) = \text{LN}(x)$ (Prop 7.2(a)). And rescaling: $\text{LN}(2 x) = \text{LN}(x)$ when $\beta = 0$ (Prop 7.2(b)). Both hold by direct computation.

## 7.5 Mechanics check

- **LN is per-token, not per-batch.** Statistics computed over the $d$ feature axis of *one* token at a time. No cross-token mixing — preserves the per-token semantics that attention relies on.
- **$\gamma, \beta$ are per-coordinate.** Each of the $d$ coordinates has its own scale and shift parameter. Total $2d$ extra parameters per LN.
- **Pre-norm leaves the residual stream alone.** The sub-layer sees a normalized input; the residual stream itself is just the running sum of un-normalized contributions. This is what makes Prop 7.3 land cleanly.
- **Post-norm normalizes the residual stream.** Repeated LNs on the residual stream eventually flatten its scale information. This is why post-norm transformers need warmup — early gradients can blow up before LN's scale-killing has anything to compress.
- **Output LN is non-optional.** Without a final LN before the unembedding, residual-stream norm growth makes output logits unbounded. Modern stacks always have a final LN/RMSNorm.
- **$\epsilon$ in the denominator.** Numerical stability for the case $\sigma(x) \to 0$ (input is constant). $\epsilon = 10^{-5}$ or $10^{-6}$ typical. Doesn't affect the math.

## 7.6 Exercises

**7.1 ★★** *[Mechanism dissection — LN's null space.]* Prop 7.6 says LN's Jacobian has rank $d - 2$. Explicitly identify the two null directions and verify via direct calculation that LN is constant along each. State a precise version of the claim "LN destroys two degrees of freedom per token, and the residual carries them forward."

**7.2 ★★** *[Failure-mode construction — post-norm at depth.]* Construct a stack of $L$ post-norm blocks where the gradient from the loss with respect to the input becomes vanishingly small as $L \to \infty$ for any non-trivial sub-layer. Quantify: at what depth does the gradient norm drop below $10^{-6}$ for typical $f_\ell$ Jacobians? Then redo the calculation for pre-norm and show the gradient *does not* vanish.

**7.3 ★★★** *[Theorem about the numerics — RMSNorm vs. LayerNorm bias term.]* RMSNorm differs from LN by removing mean centering: $\text{RMSNorm}(x) = \gamma \odot x / \text{RMS}(x)$.

(a) Compute the Jacobian of RMSNorm and find its null space. How many degrees of freedom does it kill?

(b) Show that the difference $\text{LN}(x) - \text{RMSNorm}(x)$ is a function of $\mu(x)$ alone (modulo $\gamma, \beta$). Conclude: LN and RMSNorm differ only in how they treat the mean direction.

(c) Empirically, RMSNorm performs comparably to LN in transformers despite this difference. Hypothesize: why does the mean direction matter so little after a few transformer layers? Relate to the residual stream's tendency to develop a "DC component" that subsequent layers learn to ignore.

**7.4 ★★★** *[Predict-then-verify — pre-norm vs. post-norm at depth.]* The notebook will train two transformers — one pre-norm, one post-norm — to varying depths $L \in \{2, 4, 8, 16\}$ on a synthetic task. Predict, before running:

(a) For each depth, predict whether each variant trains stably to convergence. Justify from Prop 7.3 and Empirical obs 7.4.

(b) Where post-norm fails, predict the failure mode: NaN, slow convergence, divergence after some steps?

(c) Predict the ratio of training-loss-at-step-100 between pre-norm and post-norm at $L = 16$. Order of magnitude.

(d) Bonus prediction: does adding a learning-rate warmup of 100 steps fix post-norm at $L = 16$?

State confidence per part.

## 7.7 Before the notebook (`07_norm_and_residuals.ipynb`)

Write these predictions before running.

1. **LN by hand vs. PyTorch.** Should match exactly. Predict any source of deviation.
2. **LN invariances.** Predict the max-abs difference between $\text{LN}(x)$, $\text{LN}(x + 100)$, and $\text{LN}(2x)$ (with $\beta = 0$).
3. **Residual gradient flow.** Build a stack of $L = 32$ residual blocks with random Jacobians. Predict the gradient norm at the input as a function of $L$, with and without residuals. Order of magnitude.
4. **Pre-norm vs. post-norm at $L = 16$.** Loss-at-step-100 ratio prediction.
5. **Output norm growth in pre-norm.** For a 16-layer pre-norm transformer at init, predict the ratio $\|x_{16}\| / \|x_0\|$. Constant? Polynomial in $L$? Exponential?

For each: direction, magnitude, confidence.
