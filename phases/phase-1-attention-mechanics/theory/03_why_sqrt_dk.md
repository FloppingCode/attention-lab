# Chapter 3 — Why $\sqrt{d_k}$?

```
Type: theory + notebook
Ordering: theory-first
Depends on: 1 (scaled dot-product attention), 2 (softmax & temperature)
```

## 3.1 Motivation

The scaling factor $1/\sqrt{d_k}$ in scaled dot-product attention is not cosmetic. It is the factor that keeps pre-softmax scores of order one as the head dimension grows. Without it, softmax saturates at initialization and gradients die. Why $\sqrt{d_k}$ specifically — not $d_k$, not $\log d_k$, not a learned scalar — is a variance argument followed by a gradient argument.

## 3.2 Storyline

The mechanism is a three-link chain.

1. **Variance in, variance out.** The inner product of two independent random vectors with i.i.d. unit-variance entries has variance equal to the dimension. Raw scores $s_{ij} = \langle q_i, k_j \rangle$ therefore have standard deviation $\sqrt{d_k}$, which grows unboundedly with head dimension. For $d_k = 128$, scores live on the scale of $\pm 11$.

2. **Softmax concentrates under large inputs.** From Chapter 2, softmax with a winner of margin $\Delta$ gives $p_{\text{win}} = e^\Delta / (e^\Delta + (n-1))$; once $\Delta \gg \log(n-1)$, $p_{\text{win}} \to 1$ and the distribution is effectively one-hot. Large-scale inputs push softmax into this saturated regime.

3. **Saturated softmax has zero Jacobian.** $\partial p_i / \partial s_j = p_i(\delta_{ij} - p_j)$. When $p$ is near one-hot every entry of this Jacobian is near zero. No gradient flows back to $Q, K$, so attention cannot learn.

Dividing by $\sqrt{d_k}$ rescales the first link so pre-softmax inputs have unit variance regardless of head dimension. Softmax then sees inputs of moderate magnitude and the gradient survives.

**Regime notes.** The argument is sharpest *at initialization*. After training, learned $W_Q, W_K$ restructure the score distribution in ways the i.i.d. hypothesis does not capture (see §3.3, Empirical observation 3.4). Measured pre-softmax score variance in a trained model is typically below one even with the $\sqrt{d_k}$ factor present; the factor's job is to keep *early* training healthy, not to maintain unit variance throughout. The factor is a fixed constant — no train / val / inference distinction, no interaction with dropout or masking.

## 3.3 Rigorous core

**Lemma 3.1 (Variance of an inner product).** Let $q, k \in \mathbb{R}^{d_k}$ have independent components $q_i, k_i$ with $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$ and $\text{Var}(q_i) = \text{Var}(k_i) = 1$, and $q \perp k$ as random vectors. Then
$$\mathbb{E}[\langle q, k \rangle] = 0, \qquad \text{Var}(\langle q, k \rangle) = d_k.$$
*Sketch: mean by independence; variance by independence across $i$ and the factoring $\mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$. Generalized in Exercise 3.1.*

**Proposition 3.2 (The $\sqrt{d_k}$ correction).** Under the hypotheses of Lemma 3.1, $\text{Var}(\langle q, k \rangle / \sqrt{d_k}) = 1$.

**Lemma 3.3 (Jacobian vanishing under saturation).** Let $p = \text{softmax}(s)$ for $s \in \mathbb{R}^n$. Let $J_{ij} = \partial p_i / \partial s_j = p_i(\delta_{ij} - p_j)$. If $p \to e_k$ (one-hot at index $k$), then $\|J\|_F \to 0$. *Sketch: case-split on $i = k$ vs. $i \ne k$ — every entry vanishes individually.*

**Corollary 3.3.1.** Pre-softmax scores with standard deviation $\sqrt{d_k}$ place softmax in the saturating regime (Chapter 2, §2.4); combined with Lemma 3.3, this means gradients at $W_Q, W_K$ through an unscaled attention layer vanish in expectation as $d_k \to \infty$ at initialization.

**Empirical observation 3.4 (The i.i.d. hypothesis breaks post-training).** In a trained model the entries of $q, k$ are not i.i.d. — learned $W_Q, W_K$ concentrate mass in specific directions dictated by the task. Pre-softmax score variance in trained attention is typically *below* one even with the $\sqrt{d_k}$ factor present. The scaling argument is sharp at init; after training it is conservative. This motivates alternatives like QK-norm (Exercise 3.3) that do not depend on the i.i.d. hypothesis.

## 3.4 Numerical example — variance scaling is a distributional fact

A single hand computation of $\langle q, k \rangle$ for fixed $q, k$ doesn't surface what Lemma 3.1 is saying — the lemma is about the *distribution* of the inner product across i.i.d. samples, not any particular value. The minimal experiment that reveals the mechanism is a Monte-Carlo estimate of the variance across $d_k$:

```python
import numpy as np

M = 100_000
for d_k in [1, 4, 16, 64, 256, 1024]:
    q = np.random.randn(M, d_k)
    k = np.random.randn(M, d_k)
    s = np.sum(q * k, axis=1)
    print(f"d_k={d_k:>5}  var(s) ≈ {s.var():8.2f}   var(s/√d_k) ≈ {(s / np.sqrt(d_k)).var():5.3f}")
```

Expected: the first column tracks $d_k$ linearly, the second stays near $1$. The notebook sweeps this across several random seeds and measures the full distribution, not just the variance.

## 3.5 Mechanics check

- **Scale by $1/\sqrt{d_k}$** — keeps pre-softmax score variance at $O(1)$ regardless of head dimension (Lemma 3.1 + Proposition 3.2). Without this, softmax saturates at init and gradients die.
- **Why $\sqrt{d_k}$, not $d_k$** — we want *standard deviation* $O(1)$, not variance $O(1/d_k)$. Dividing by $d_k$ shrinks scores too hard and collapses softmax toward uniform (temperature $\to \infty$ in Chapter 2's sense), which is also uninformative.
- **Why not a learned scalar at init** — a learned scalar is initialized somewhere; starting far from $1/\sqrt{d_k}$ creates the saturation problem this factor is designed to prevent. Learned scalars are a *post-init* tool (see QK-norm, Exercise 3.3).
- **Interaction with softmax, not with masking or dropout** — the factor sits between $QK^\top$ and softmax; it does not interact with the additive mask (which is applied to the scaled scores) or with attention dropout (which is applied to the softmax output). No train / val / inference asymmetry.

## 3.6 Exercises

**3.1 ★★** *[Mechanism dissection — general variances.]* Let entries of $q, k$ be i.i.d. with $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$, $\text{Var}(q_i) = \sigma_q^2$, $\text{Var}(k_i) = \sigma_k^2$. Derive the scaling factor $c(d_k, \sigma_q, \sigma_k)$ such that $\langle q, k \rangle / c$ has unit variance. What does this say about the common convention of initializing both $W_Q$ and $W_K$ at the same scale?

**3.2 ★★** *[Failure-mode construction — dependent $q$ and $k$.]* Suppose $q = W_Q x$ and $k = W_K x$ for the *same* input $x \in \mathbb{R}^d$, so $q$ and $k$ are not independent. Let $x$ have i.i.d. unit-variance entries. Derive $\text{Var}(\langle q, k \rangle)$ in terms of $W_Q, W_K$. Does it still scale like $d_k$? Construct explicit $W_Q, W_K$ for which the variance is $\Theta(d_k^2)$ (much larger than $d_k$), and others for which it is $O(1)$ (much smaller). What structural feature of $W_Q^\top W_K$ controls the answer?

**3.3 ★★★** *[Theorem about the numerics — QK-norm.]* QK-norm replaces $QK^\top / \sqrt{d_k}$ with $\gamma \cdot \hat Q \hat K^\top$, where $\hat q_i = q_i / \|q_i\|_2$ row-wise (similarly $\hat k_j$), and $\gamma \in \mathbb{R}$ is learned.

(a) Prove that $\hat q^\top \hat k \in [-1, 1]$ *deterministically*, independent of any distributional assumption on $q, k$.
(b) Conclude that pre-softmax scores are bounded by $\pm \gamma$, and contrast this with the $\sqrt{d_k}$ guarantee (which is an expectation, not a bound).
(c) In Chapter 2's language, what quantity does $\gamma$ play the role of? State the analogy precisely.
(d) Give one situation in which QK-norm is strictly more robust than $\sqrt{d_k}$-scaling, and one in which it is strictly less expressive.

**3.4 ★★★** *[Predict-then-verify — unscaled attention across $d_k$.]* The notebook will train a 1-layer transformer with *unscaled* attention on a simple copy task, sweeping $d_k \in \{4, 16, 64, 256\}$. For each $d_k$, predict *before running* which of the following occurs, and justify from Lemmas 3.1 and 3.3 and Corollary 3.3.1:

(a) training loss fails to decrease at all (plateau at the init-entropy floor);
(b) training loss decreases slowly and converges to a worse optimum than the scaled baseline;
(c) training loss decreases normally (the unscaled model learns fine);
(d) training loss diverges (blows up) after some number of steps.

Write your prediction as a $4 \times 4$ table (rows: $d_k$, columns: outcomes (a)–(d), entries: your assigned probability summing to 1 per row) with a one-line reason per row. State your overall confidence (low / medium / high, with reason).

## 3.7 Before the notebook (`03_variance_scaling.ipynb`)

Write down the following predictions before opening the notebook. The notebook's purpose is to test them.

1. **Empirical variance of unscaled scores** as a function of $d_k \in \{1, 4, 16, 64, 256, 1024\}$ — shape, scaling exponent, numeric values at $d_k = 1$ and $d_k = 1024$.
2. **Empirical variance of scaled scores** — shape, scaling exponent. State the expected deviation from 1 (concentration rate in $M$, the number of Monte-Carlo samples).
3. **Softmax entropy** of a single row as a function of $d_k$, for unscaled vs. scaled inputs, with $n = 32$ keys. State the limit of each as $d_k \to \infty$ in one word.
4. **Max / mean softmax ratio** in a single row, as a function of $d_k$, for unscaled vs. scaled. Direction and limit.
5. **Frobenius norm of the softmax Jacobian** (evaluated at a random score vector) as a function of $d_k$, unscaled vs. scaled. Direction and the approximate rate of decay in the unscaled case (polynomial or exponential in $d_k$?).

For each prediction: direction, order-of-magnitude estimate, confidence level with a one-line reason. The notebook confirms or refutes each in its Finding section.
