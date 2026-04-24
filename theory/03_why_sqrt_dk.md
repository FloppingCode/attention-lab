# Chapter 3 — Why $\sqrt{d_k}$?

## 3.1 Motivation

The "scaled" in scaled dot-product attention refers to dividing by $\sqrt{d_k}$ before softmax:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V.$$

Why this specific factor? Not $d_k$, not $\log d_k$, not 1. The answer is a direct consequence of how variance accumulates in inner products.

## 3.2 The variance argument

**Setup.** Assume the entries of $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ are i.i.d. with mean $0$ and variance $1$, and that $\mathbf{q} \perp \mathbf{k}$ (independent as random vectors).

**Claim.** $\mathbb{E}[\langle \mathbf{q}, \mathbf{k} \rangle] = 0$ and $\text{Var}(\langle \mathbf{q}, \mathbf{k} \rangle) = d_k$.

*Proof.*
$$\mathbb{E}\!\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_i \mathbb{E}[q_i] \mathbb{E}[k_i] = 0.$$
For the variance, because the $q_i k_i$ terms are independent (different $i$'s) with mean zero:
$$\text{Var}\!\left( \sum_i q_i k_i \right) = \sum_i \text{Var}(q_i k_i) = \sum_i \mathbb{E}[q_i^2 k_i^2] = \sum_i \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = d_k \cdot 1 \cdot 1 = d_k. \quad \blacksquare$$

**Consequence.** If we feed raw scores $s_{ij} = \langle \mathbf{q}_i, \mathbf{k}_j \rangle$ into softmax, their standard deviation is $\sqrt{d_k}$. For modern transformers ($d_k = 64, 128$), scores have std 8–11. That is a *huge* input to softmax — it drives the distribution toward one-hot (see Ch. 2, Exercise 2.5) and shrinks gradients.

## 3.3 The fix

Divide by $\sqrt{d_k}$:
$$\text{Var}\!\left( \frac{\langle \mathbf{q}, \mathbf{k} \rangle}{\sqrt{d_k}} \right) = \frac{d_k}{d_k} = 1.$$

Now scores have unit variance regardless of head dimension. Softmax receives inputs of moderate magnitude, gradients flow.

## 3.4 Why softmax saturates on large inputs

Recall Exercise 2.5: with a single "winner" of margin $\Delta$ over $n-1$ ties, $p_{\text{win}} = \frac{e^\Delta}{e^\Delta + (n-1)}$. Once $\Delta \gg \log(n-1)$, $p_{\text{win}} \to 1$ and the distribution collapses.

Now consider gradients. From Prop 2.2:
$$\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j).$$

If $p_i \to 1$ and all others $\to 0$, then $\partial p_i / \partial s_j \to 0$ for $j \ne i$, and $\partial p_i / \partial s_i = p_i(1 - p_i) \to 0$. **All gradients vanish.** Saturated softmax = dead attention during training.

Scaling by $\sqrt{d_k}$ keeps pre-softmax magnitudes in the regime where gradients are healthy.

## 3.5 Caveat: the i.i.d. assumption

The $\sqrt{d_k}$ derivation assumes $\mathbf{q}, \mathbf{k}$ have i.i.d. unit-variance entries. After training, this is *not exactly true* — learned projections $W_Q, W_K$ concentrate variance in specific directions. Empirically, trained attention scores often have std *less* than 1 with the $\sqrt{d_k}$ factor, which is fine. The factor is right at initialization, where training must not immediately saturate.

Some recent architectures replace $\sqrt{d_k}$ with a *learned* scalar, or use **QK-normalization** (L2-normalize $Q$ and $K$ row-wise, then multiply by a learned scalar). These address the same concern with fewer assumptions.

## 3.6 Exercises

**3.1 ★** You have $d_k = 64$ and use unscaled attention at initialization. Pre-softmax scores have std $\approx 8$. If the "winner" token's score exceeds its rivals' scores by $3\sigma$, how close is its softmax probability to 1? *(Use Exercise 2.5 with $\Delta = 24$.)*

**3.2 ★★** Suppose entries of $\mathbf{q}, \mathbf{k}$ are i.i.d. with variance $\sigma_q^2, \sigma_k^2$ respectively (not necessarily 1). Derive the correct scaling factor. Answer in terms of $d_k, \sigma_q, \sigma_k$.

**3.3 ★★** Suppose $\mathbf{q} = W_Q \mathbf{x}$ and $\mathbf{k} = W_K \mathbf{x}$ with **the same** input $\mathbf{x}$ (so $\mathbf{q}$ and $\mathbf{k}$ are *not* independent). Does $\text{Var}(\langle \mathbf{q}, \mathbf{k} \rangle) = d_k$ still hold? If not, what changes?

**3.4 ★★★** Consider **QK-norm**: replace $QK^\top/\sqrt{d_k}$ with $\gamma \cdot \hat Q \hat K^\top$, where $\hat Q_{i,:} = Q_{i,:} / \|Q_{i,:}\|$ and similarly for $\hat K$, and $\gamma$ is a learned scalar. Argue why QK-norm is more robust to shifts in input distribution than plain $\sqrt{d_k}$-scaled attention. What does $\gamma$ control? *Hint: think about Ch. 2 temperature.*

**3.5 ★★★** Run the following thought experiment. Train a 1-layer transformer with *unscaled* attention on a simple copy task. Predict — before reading any literature — which of these fails first:
(a) loss blows up,
(b) loss plateaus immediately,
(c) loss converges but to a worse optimum than scaled attention,
(d) training works fine for small $d_k$ but breaks once $d_k \gtrsim 64$.
Justify your prediction from the theory above. *(You will test this in notebook 03.)*

## 3.7 Before the notebook (`03_variance_scaling.ipynb`)

The notebook will:

1. Sample $Q, K$ with i.i.d. normal entries for $d_k \in \{1, 4, 16, 64, 256, 1024\}$.
2. Compute pre-softmax score histograms, with and without scaling. Verify empirically that $\text{Var}(QK^\top_{ij}) = d_k$.
3. Plot softmax entropy vs $d_k$ for scaled and unscaled attention.
4. Compute the magnitude of the gradient $\partial \mathbf{p} / \partial \mathbf{s}$ in both regimes.

Predict each plot shape before running the code. Write your predictions in the scratch cell at the top of the notebook.
