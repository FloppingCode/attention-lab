# Chapter 2 — Softmax and Temperature

## 2.1 Motivation

Attention scores $s_{ij}$ are arbitrary real numbers. We need to convert them into a probability distribution: non-negative, summing to one, **differentiable** w.r.t. the scores, and preserving the *ordering* of the scores. Softmax is the canonical solution.

## 2.2 Definition

**Definition 2.1 (Softmax).** For $\mathbf{s} \in \mathbb{R}^n$,
$$\text{softmax}(\mathbf{s})_i = \frac{e^{s_i}}{\sum_{j=1}^{n} e^{s_j}}.$$

**Definition 2.2 (Softmax with temperature $T > 0$).**
$$\text{softmax}_T(\mathbf{s})_i = \frac{e^{s_i / T}}{\sum_j e^{s_j / T}}.$$

As $T \to 0^+$, softmax approaches a one-hot (hard argmax). As $T \to \infty$, it approaches uniform.

## 2.3 Core properties

**Proposition 2.1 (Shift invariance).** For any constant $c$, $\text{softmax}(\mathbf{s} + c\mathbf{1}) = \text{softmax}(\mathbf{s})$.

*Proof.* $e^{s_i + c} / \sum_j e^{s_j + c} = e^c e^{s_i} / (e^c \sum_j e^{s_j}) = \text{softmax}(\mathbf{s})_i$. ∎

*Practical consequence.* Subtracting $\max_i s_i$ before exponentiating is safe and prevents overflow. This is how softmax is implemented in practice.

**Proposition 2.2 (Jacobian).** Let $\mathbf{p} = \text{softmax}(\mathbf{s})$. Then
$$\frac{\partial p_i}{\partial s_j} = p_i (\delta_{ij} - p_j),$$
where $\delta_{ij}$ is the Kronecker delta.

**Proposition 2.3 (Entropy bounds).** For $\mathbf{p} = \text{softmax}(\mathbf{s})$,
$$0 \le H(\mathbf{p}) = -\sum_i p_i \log p_i \le \log n.$$
The upper bound is achieved iff all $s_i$ are equal (uniform distribution). The lower bound is approached as one $s_i$ becomes much larger than the rest.

## 2.4 Temperature as information control

Think of temperature as controlling **how confidently** attention concentrates:

- Low $T$ → sharp distribution → selective retrieval, low entropy.
- High $T$ → flat distribution → averaging over many sources, high entropy.

When you see "attention saturates" or "attention becomes one-hot," the underlying cause is always that the effective pre-softmax scores have large magnitude. Scaling (Chapter 3) is temperature control in disguise.

**Proposition 2.4.** $\text{softmax}(\mathbf{s}/T) = \text{softmax}(\mathbf{s})$ with an effective temperature of $T$ — equivalently, dividing the *logits* by $T$ and dividing the *scores* by $T$ are the same operation. In attention, the $1/\sqrt{d_k}$ scaling is a temperature of $\sqrt{d_k}$.

## 2.5 Exercises

**2.1 ★** Compute $\text{softmax}((1, 2, 3))$ by hand (to 3 decimals). Then compute $\text{softmax}((101, 102, 103))$. Do they match? Which proposition did you just verify?

**2.2 ★** Show that $\sum_i \frac{\partial p_i}{\partial s_j} = 0$ for all $j$. Interpret: what does this say about how "probability mass" responds to a perturbation in one logit?

**2.3 ★★ (Prove Proposition 2.2.)** Use the quotient rule. Case-split on $i = j$ vs $i \ne j$.

**2.4 ★★ (Prove Proposition 2.3.)** Upper bound: use Jensen's inequality applied to $-\log$, or Lagrange multipliers to maximize entropy subject to $\sum p_i = 1$. Lower bound: entropy is non-negative for any probability distribution — why?

**2.5 ★★** Let $\mathbf{s} = (0, \Delta, 0, 0, \dots, 0) \in \mathbb{R}^n$. Derive a closed form for $p_1 = \text{softmax}(\mathbf{s})_1$ (the "winning" entry) as a function of $\Delta$ and $n$. Plot (mentally or on paper) $p_1$ against $\Delta$ for $n = 10, 100, 1000$. What does this say about how much margin a "correct" token needs to win in long-context attention?

**2.6 ★★★** Define the **log-sum-exp** function $\text{LSE}(\mathbf{s}) = \log \sum_i e^{s_i}$. Show:
(a) $\nabla \text{LSE}(\mathbf{s}) = \text{softmax}(\mathbf{s})$.
(b) $\max_i s_i \le \text{LSE}(\mathbf{s}) \le \max_i s_i + \log n$.
This is why LSE is called a "soft max" — it is a smooth upper bound on the hard max.

**2.7 ★★★** In attention, if we replace $\text{softmax}$ with a hard argmax, gradients vanish almost everywhere. Argue why. Then: name at least two ways the literature has tried to get discrete-selection attention while preserving gradients (e.g., Gumbel-softmax, straight-through estimators). You don't need to prove they work — just name and sketch.

## 2.6 Before the notebook (`02_softmax_temperature.ipynb`)

Predict:

1. A plot of entropy $H(\text{softmax}(\mathbf{s}/T))$ as $T$ ranges over $[0.01, 100]$ for a fixed random $\mathbf{s} \in \mathbb{R}^{50}$. Sketch it. Where is it high/low?
2. For random $\mathbf{s} \sim \mathcal{N}(0, \sigma^2 I_{n})$, how should the expected entropy of $\text{softmax}(\mathbf{s})$ depend on $\sigma$ and $n$? Which limit (large $\sigma$, large $n$) reduces entropy faster?
