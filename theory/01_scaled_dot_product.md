# Chapter 1 — Scaled Dot-Product Attention

> *"Attention is a differentiable, content-based lookup."*

## 1.1 Motivation

Given a sequence of token representations $\mathbf{x}_1, \dots, \mathbf{x}_n$, we want each token to **reach into the sequence** and pull information from other tokens it finds relevant. Not based on position (that's convolution), but based on **content**.

We need three things:

1. A way for token $i$ to describe *what it's looking for* — the **query** $\mathbf{q}_i$.
2. A way for token $j$ to describe *what it offers as an index* — the **key** $\mathbf{k}_j$.
3. A way for token $j$ to describe *what it offers as content* — the **value** $\mathbf{v}_j$.

Attention is how we combine these into a new representation of token $i$.

## 1.2 Definition

**Definition 1.1 (Scaled dot-product attention).** Let $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$. Define
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V.$$

Here $\text{softmax}$ is applied row-wise, so the output is in $\mathbb{R}^{n \times d_v}$.

In self-attention, $Q$, $K$, $V$ all come from the same input sequence $X \in \mathbb{R}^{n \times d}$ via learned linear projections:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V,$$
with $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$.

## 1.3 Unpacking the formula

Let's trace what happens to a single query $\mathbf{q}_i$ (row $i$ of $Q$).

1. **Score**: $s_{ij} = \langle \mathbf{q}_i, \mathbf{k}_j \rangle$. A scalar measuring how much token $j$ matches what token $i$ is looking for. The matrix form $QK^\top \in \mathbb{R}^{n \times m}$ collects all of these at once (Fact 1 from Chapter 0).

2. **Scale**: $\tilde{s}_{ij} = s_{ij} / \sqrt{d_k}$. This keeps the variance of scores stable as $d_k$ grows. Why specifically $\sqrt{d_k}$? That's Chapter 3.

3. **Normalize**: $\alpha_{ij} = \text{softmax}_j(\tilde{s}_{i, :})_j = \dfrac{e^{\tilde{s}_{ij}}}{\sum_{j'} e^{\tilde{s}_{ij'}}}$. Now $\boldsymbol{\alpha}_i$ is a probability distribution over the $m$ source tokens.

4. **Mix**: $\mathbf{o}_i = \sum_j \alpha_{ij} \mathbf{v}_j$. A convex combination of the value vectors, weighted by how much token $i$ attended to each.

That's the whole thing. Each output is a weighted average of values, with weights computed from Q-K similarity.

## 1.4 Worked example

Let $d_k = d_v = 2$, $n = m = 2$. Suppose
$$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}, \quad V = \begin{pmatrix} 10 & 0 \\ 0 & 10 \end{pmatrix}.$$

**Step 1.** $Q K^\top = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$.

**Step 2.** Divide by $\sqrt{2}$: $\begin{pmatrix} 0.707 & 0.707 \\ 0 & 0.707 \end{pmatrix}$.

**Step 3.** Row-wise softmax:
- Row 1: $[e^{0.707}, e^{0.707}]/Z_1 = [0.5, 0.5]$. (Equal — query 1 finds both keys equally similar after scaling.)
- Row 2: $[e^0, e^{0.707}]/Z_2 \approx [0.330, 0.670]$.

**Step 4.** Multiply by $V$:
- Output 1: $0.5 \cdot (10, 0) + 0.5 \cdot (0, 10) = (5, 5)$.
- Output 2: $0.330 \cdot (10, 0) + 0.670 \cdot (0, 10) \approx (3.30, 6.70)$.

## 1.5 Properties to prove

**Property 1 (Permutation equivariance over keys/values).** If we permute the rows of $K$ and $V$ by the same permutation $\pi$, the output does not change. That is, attention has no built-in notion of position — positional information must be injected separately (Chapter 5).

**Property 2 (Convex combination).** Each output row $\mathbf{o}_i$ lies in the convex hull of the rows of $V$. No output can have a coordinate larger than the max value coordinate or smaller than the min.

**Property 3 (Differentiability).** $\text{Attention}(Q, K, V)$ is differentiable in $Q$, $K$, and $V$ everywhere, because softmax is smooth.

## 1.6 Exercises

**1.1 ★** Redo the worked example with $V = \begin{pmatrix} 10 & 0 \\ 0 & 10 \end{pmatrix}$ replaced by $V = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$. Explain the output in one sentence.

**1.2 ★** Compute $\text{Attention}(Q, K, V)$ for $Q = K = V = I_3$ (the $3 \times 3$ identity). You should get a matrix that is *almost* the identity. Which entries differ, and why?

**1.3 ★★ (Prove Property 1.)** Let $P$ be an $m \times m$ permutation matrix. Show that
$$\text{Attention}(Q, P K, P V) = \text{Attention}(Q, K, V).$$
*Hint: $P^\top P = I$, and softmax applied to a permuted row gives a permuted output.*

**1.4 ★★ (Prove Property 2.)** Use the fact that $\alpha_{ij} \ge 0$ and $\sum_j \alpha_{ij} = 1$.

**1.5 ★★** Show that attention with $d_k = 1$ and $V = $ (a column of $n$ distinct scalars) reduces to a **soft-argmax** lookup. Under what condition on the scores does the output approach a *hard* argmax (i.e., exactly one $\alpha_{ij} \to 1$)?

**1.6 ★★★** Suppose you want attention to behave like a **hard lookup**: each query selects exactly one key deterministically. Can you achieve this with the current formulation by adjusting a hyperparameter? What breaks when you push it to the limit? (You are previewing the temperature discussion in Chapter 2.)

**1.7 ★★★** Show that if $W_Q$ and $W_K$ are the *same* linear map (tied weights), self-attention becomes symmetric: $s_{ij} = s_{ji}$. Does this help or hurt expressivity? Construct a toy task where untied $W_Q, W_K$ is strictly more expressive.

## 1.7 Before the notebook

Write down your predictions for `notebooks/01_attention_by_hand.ipynb`:

1. If $\mathbf{q}_i = \mathbf{k}_j$ for some specific $j$ and all other keys are orthogonal, what does the attention distribution look like? Draw it as a bar chart.
2. If we scale $Q$ by a large constant $\lambda \to \infty$, what happens to the output?
3. If all keys are equal, what does the output equal? (No computation needed — reason from the definition.)
