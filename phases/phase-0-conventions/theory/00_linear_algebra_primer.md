# Chapter 0 — Linear Algebra Primer

A diagnostic, not a course. If these exercises feel easy, skip to Chapter 1. If they feel hard, go through *Linear Algebra Done Right* (Axler) Ch. 1–6 or *Mathematics for Machine Learning* Ch. 2–4 before continuing.

## 0.1 Notation we will use

- Vectors: lowercase bold, $\mathbf{x} \in \mathbb{R}^d$. Treated as **column vectors** by default.
- Matrices: uppercase, $A \in \mathbb{R}^{m \times n}$.
- Inner product: $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = \sum_i x_i y_i$.
- Euclidean norm: $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$.
- A row of matrix $A$: $A_{i,:}$. A column: $A_{:,j}$.

## 0.2 Facts to have at your fingertips

**Fact 1.** For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the entry $(AB)_{ij}$ equals the inner product of row $i$ of $A$ with column $j$ of $B$:
$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj} = \langle A_{i,:}, B_{:,j} \rangle.$$

This fact is the entire reason attention works the way it does. Internalize it.

**Fact 2.** $\langle \mathbf{x}, \mathbf{y} \rangle = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$ where $\theta$ is the angle between $\mathbf{x}$ and $\mathbf{y}$. So inner products measure directional similarity, scaled by magnitude.

**Fact 3.** If $X_1, \dots, X_n$ are i.i.d. with mean $0$ and variance $\sigma^2$, then $\sum_i X_i$ has variance $n\sigma^2$, and $\frac{1}{\sqrt{n}} \sum_i X_i$ has variance $\sigma^2$. (This is the reason for the $\sqrt{d_k}$ scaling in Chapter 3.)

## 0.3 Exercises

**0.1 ★** Let $\mathbf{q} = (1, 2, 0)^\top$ and let $K$ have rows $\mathbf{k}_1 = (1,0,1)$, $\mathbf{k}_2 = (0,2,0)$, $\mathbf{k}_3 = (-1, 1, 1)$. Compute $K\mathbf{q}$ by hand. Which row of $K$ is "most similar" to $\mathbf{q}$ under the inner product?

**0.2 ★** Show that if $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ have i.i.d. entries with mean $0$ and variance $1$, then $\mathbb{E}[\langle \mathbf{x}, \mathbf{y} \rangle] = 0$ and $\text{Var}(\langle \mathbf{x}, \mathbf{y} \rangle) = d$. *Hint: linearity of expectation, independence.*

**0.3 ★★** Consequence of 0.2: if we want the inner product $\langle \mathbf{x}, \mathbf{y} \rangle$ to have variance $1$ regardless of $d$, by what scalar should we divide it? Record your answer — it returns in Chapter 3.

**0.4 ★★** Let $A \in \mathbb{R}^{n \times d}$ and $\mathbf{q} \in \mathbb{R}^d$. Show that $A \mathbf{q} \in \mathbb{R}^n$ is the vector whose $i$-th entry is $\langle A_{i,:}, \mathbf{q} \rangle$. *This is just Fact 1 specialized — write it out carefully.*

**0.5 ★★★** Suppose we want a row-stochastic matrix $P \in \mathbb{R}^{n \times n}$ (each row sums to 1, entries in $[0,1]$) such that $P \mathbf{v}$ produces a "weighted average" of the entries of $\mathbf{v}$ according to row $i$'s weights. Given arbitrary scores $S \in \mathbb{R}^{n \times n}$, write a formula that produces such a $P$ from $S$. What properties should the formula satisfy? (You are deriving softmax. We will formalize it in Chapter 2.)

## 0.4 Before Chapter 1

Make sure you can, without notes:

- Compute a $3 \times 3$ matrix-vector product.
- State what $\mathbf{x}^\top \mathbf{y}$ equals geometrically.
- Explain *in one sentence* why summing $d$ independent unit-variance terms gives variance $d$.
