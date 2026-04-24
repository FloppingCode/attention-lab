# Chapter 3 — Solutions

Resist reading until you have made a genuine attempt.

---

## 3.1 (★★) General-variance scaling

**Setup recap.** $q, k \in \mathbb{R}^{d_k}$ with independent entries, $\text{Var}(q_i) = \sigma_q^2$, $\text{Var}(k_i) = \sigma_k^2$, $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$, $q \perp k$.

**Derivation.** Each summand $q_i k_i$ has $\mathbb{E}[q_i k_i] = 0$ and $\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] = \sigma_q^2 \sigma_k^2$ (independence of $q_i, k_i$ makes the product's second moment factor). Summing $d_k$ independent such terms:
$$\text{Var}(\langle q, k \rangle) = d_k \, \sigma_q^2 \sigma_k^2.$$
To obtain unit variance, divide by $c = \sqrt{d_k}\, \sigma_q \sigma_k$. *Why each step:* the first step uses independence across indices (making variances add); the second uses independence of $q_i$ from $k_i$ (making the second moment factor as a product of second moments).

**Takeaway.** The $\sqrt{d_k}$ scaling is *only correct* when $\sigma_q = \sigma_k = 1$. If $W_Q, W_K$ are initialized with, say, Xavier scaling $\sigma = 1/\sqrt{d_{\text{model}}}$, then $\sigma_q = \sigma_k = 1/\sqrt{d_{\text{model}}}$ and the "correct" constant is $\sqrt{d_k}/d_{\text{model}}$, not $\sqrt{d_k}$. In practice, modern initialization (e.g., scaled Xavier / He with the pre-LN factored in) aims to make $\sigma_q \approx \sigma_k \approx 1$ at the post-projection point, so $\sqrt{d_k}$ is approximately right *by convention*, not automatically. This is a small but real source of confusion when porting init schemes.

---

## 3.2 (★★) Dependent $q, k$ from a shared input

**Setup.** $q = W_Q x$, $k = W_K x$ with $x \in \mathbb{R}^d$ having i.i.d. unit-variance zero-mean entries; $W_Q, W_K \in \mathbb{R}^{d_k \times d}$.

**Derivation.** Write the dot product as a quadratic form in $x$:
$$\langle q, k \rangle = x^\top W_Q^\top W_K x = x^\top M x, \qquad M := W_Q^\top W_K \in \mathbb{R}^{d \times d}.$$
For $x \sim (0, I_d)$ with independent entries, the variance of $x^\top M x$ is (standard result)
$$\text{Var}(x^\top M x) = \|M\|_F^2 + \text{tr}(M^2) + (\kappa_4 - 3)\sum_i M_{ii}^2,$$
where $\kappa_4$ is the fourth moment of $x_i$ (for $\mathcal{N}(0,1)$, $\kappa_4 = 3$ and the last term drops). For Gaussian $x$:
$$\boxed{\text{Var}(\langle q, k \rangle) = \|W_Q^\top W_K\|_F^2 + \text{tr}\!\left((W_Q^\top W_K)^2\right).}$$

**Does it still scale like $d_k$?** Not necessarily. The answer is dictated by the structure of $M = W_Q^\top W_K$, which is $d \times d$ and rank at most $d_k$.

**Constructions.**

- **Variance $\Theta(d_k^2)$ (much larger than $d_k$).** Take $W_Q = W_K = \sqrt{d_k}\, I_{d_k, d}$ (pad to $d_k \times d$ with zeros outside the top-left $d_k \times d_k$ block, and scale so each nonzero diagonal is $\sqrt{d_k}$). Then $M = d_k \cdot P$ where $P$ projects onto the first $d_k$ coords, giving $\|M\|_F^2 = d_k^2 \cdot d_k = d_k^3$ and $\text{tr}(M^2) = d_k^2 \cdot d_k = d_k^3$. Variance $= 2 d_k^3$. The point: coupling $W_Q$ and $W_K$ (here, equality) makes $\langle q, k \rangle = \|W x\|^2$ essentially — a nonnegative quadratic form with huge variance.

- **Variance $O(1)$ (much smaller than $d_k$).** Take $W_Q, W_K$ with orthogonal row spaces: $W_Q W_K^\top = 0$, equivalently $W_K W_Q^\top = 0$. Then $M^\top = W_K^\top W_Q$ satisfies $M^\top$-column space $\perp$ $W_K$-row space... cleanest version: pick $W_Q$ supported on the first $d_k$ coords of $x$ and $W_K$ on the last $d_k$ coords (assume $d \ge 2 d_k$). Then $q$ is a function of $x_{1:d_k}$, $k$ of $x_{d_k+1:2d_k}$. These are independent (since $x$ has independent entries), so we are back in the Lemma 3.1 regime, variance $= d_k \sigma_q^2 \sigma_k^2$.

**Structural feature.** The singular-value structure of $M = W_Q^\top W_K$ controls everything. Aligned $W_Q, W_K$ (large overlap between the row span of $W_K$ and the column span of $W_Q$, so $M$ has large singular values) produce variance much larger than $d_k$. Orthogonal $W_Q, W_K$ ($M$ small) recover the i.i.d.-like regime. *Why this matters:* at initialization, $W_Q$ and $W_K$ are drawn independently, so their $M$ is roughly a product of independent Gaussian matrices with $\|M\|_F^2 \approx d_k^2 / d$, close to the i.i.d. regime. After training, they become aligned in meaningful directions — which is the empirical observation from §3.3 that variance can drift away from the Lemma 3.1 prediction.

---

## 3.3 (★★★) QK-norm

**(a) Deterministic bound.** For any nonzero $q, k \in \mathbb{R}^{d_k}$, $\hat q := q / \|q\|_2$ and $\hat k := k / \|k\|_2$ are unit vectors. By Cauchy–Schwarz, $|\hat q^\top \hat k| \le \|\hat q\|_2 \|\hat k\|_2 = 1$, with equality iff $\hat q = \pm \hat k$. Hence $\hat q^\top \hat k \in [-1, 1]$ regardless of the distribution of $q, k$.

**(b) Bound on pre-softmax scores.** Each entry of $\gamma \hat Q \hat K^\top$ lies in $[-\gamma, \gamma]$ deterministically. Contrast: under the i.i.d. hypothesis, $\text{Var}(\langle q, k \rangle / \sqrt{d_k}) = 1$ only *on expectation*. Individual realizations can be much larger — for instance, under Gaussian entries the score is $\mathcal{N}(0, 1)$ so $P(|s| > 4) \approx 6 \times 10^{-5}$ per entry, and across an $n \times n$ attention matrix with $n = 1024$ there are $\approx 60$ entries exceeding that per forward pass. QK-norm rules those out structurally.

**(c) $\gamma$ is temperature.** In Chapter 2, the softmax temperature $\tau$ appears as $\text{softmax}(s / \tau)$: small $\tau$ concentrates, large $\tau$ smooths. In QK-norm, $\text{softmax}(\gamma \hat Q \hat K^\top)$: large $\gamma$ concentrates (all $[-\gamma, \gamma]$-bounded scores spread further), small $\gamma$ smooths. So $\gamma$ plays the role of *inverse temperature*: $\gamma \leftrightarrow 1 / \tau$. Training can learn it to reach any concentration level.

**(d) Robustness vs. expressivity.**

- **QK-norm strictly more robust:** under distribution shift in $q, k$ scale (e.g., layer-norm stats drift during training, activations blow up under precision loss in fp16), $\sqrt{d_k}$-scaling passes the scale drift through to the softmax input and can saturate. QK-norm's bound $[-\gamma, \gamma]$ is scale-invariant by construction.
- **QK-norm strictly less expressive:** the magnitude of $q$ and $k$ is discarded. If the model wants to encode *confidence* about a match via the magnitude of $q$ (a token that strongly wants to attend uses a larger-norm query than one that is indifferent), QK-norm kills that channel. Only the directional geometry is preserved. With plain $\sqrt{d_k}$-scaled attention, $\|q\| \|k\| \cos\theta$ is passed to softmax, so magnitude carries information.

---

## 3.4 (★★★) Predict-then-verify — unscaled attention across $d_k$

No "solution" in the normal sense — this exercise is the setup for the notebook. A defensible prediction grid, with justifications, looks like:

| $d_k$ | $P(\text{a: plateau})$ | $P(\text{b: worse optimum})$ | $P(\text{c: fine})$ | $P(\text{d: diverge})$ | Reason |
|---|---|---|---|---|---|
| 4 | 0.05 | 0.1 | 0.80 | 0.05 | Score std $\approx 2$, softmax not yet saturated; Jacobian still healthy. |
| 16 | 0.2 | 0.35 | 0.40 | 0.05 | Score std $= 4$; softmax partial saturation, gradient noticeably reduced but not dead. |
| 64 | 0.6 | 0.3 | 0.05 | 0.05 | Score std $= 8$; Corollary 3.3.1 applies — expected Jacobian near zero. Expect plateau or very slow descent. |
| 256 | 0.85 | 0.1 | 0.02 | 0.03 | Score std $= 16$; softmax essentially one-hot on random init; gradient zero to numerical precision. |

Overall confidence: **medium.** The theory argues strongly for (a) at large $d_k$, but real training has warm-up, Adam adaptive LR, and numerical stabilization in softmax (log-sum-exp) that may delay or soften the failure. The relative ordering across $d_k$ is higher confidence than the absolute outcome at any single $d_k$. Divergence (d) is unlikely under a stable optimizer — more likely at the data side (nan propagation) than the attention side.

*What makes this a good exercise:* it forces you to commit to a falsifiable grid, not a hedge. If the notebook shows (say) that $d_k = 64$ trains fine, you learn that log-sum-exp + Adam mask the scaling issue more than Corollary 3.3.1 suggests — a useful update to the mental model, not a failure.
