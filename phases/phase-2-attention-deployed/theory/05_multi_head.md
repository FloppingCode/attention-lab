# Chapter 5 — Multi-head attention

```
Type: theory + notebook
Ordering: mixed
Depends on: 1 (scaled dot-product), 3 (√d_k scaling)
```

## 5.1 Motivation

A single attention head produces output rows in the convex hull of linear-projected value rows (Corollary 1.3.1) — one mixing per position, weighted by one similarity. Multi-head attention runs $h$ independent heads in parallel, each with its own $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ at dimension $d_k = d_v = d_{\text{model}} / h$, and concatenates the results. The claim — and it *is* a claim, not a theorem — is that different heads specialize to different attention patterns, giving the block access to multiple attention shapes per position at no additional FLOP cost compared to a single head of the same total dimension.

## 5.2 Storyline

**Design.** Given input $X \in \mathbb{R}^{n \times d_{\text{model}}}$:
1. Project to $h$ heads each at dimension $d_h := d_{\text{model}} / h$: $Q_i = X W_Q^{(i)}$, etc., with $W_Q^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_h}$.
2. Run scaled dot-product attention independently in each head: $\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$.
3. Concatenate along the feature axis: $H = [\text{head}_1 | \cdots | \text{head}_h] \in \mathbb{R}^{n \times d_{\text{model}}}$.
4. Output projection: $\text{MHA}(X) = H W_O$ with $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

**Why this is not equivalent to one big head.** A single head with dimension $d_{\text{model}}$ produces one $n \times n$ attention matrix — one mixing pattern. Multi-head with $h$ heads produces $h$ different $n \times n$ attention matrices, each over different subspaces of the representation. The output is a concatenation of $h$ independent convex combinations, one per head. This breaks the single-mixing-pattern constraint of Corollary 1.3.1 at the block level.

**Why the FLOP count is unchanged.** Total compute in the attention step: $h$ heads $\times$ $O(n^2 d_h)$ per head = $O(n^2 d_{\text{model}})$. A single head of dimension $d_{\text{model}}$ would also cost $O(n^2 d_{\text{model}})$. Splitting into heads is a *reshuffling* of the same FLOP budget, not an increase.

**What each head learns.** Empirically (and this is **Empirical observation 5.4** below, not a theorem), heads specialize: some attend to previous tokens (positional), some to syntactically-similar tokens, some to semantically-similar tokens, some are "induction heads" (Ch. 15) that copy based on patterns, some are near-constant. The specialization is emergent from training, not enforced by architecture.

**Train / val / inference.** No operator asymmetry — the MHA forward pass is identical in all three regimes. Heads are fixed in number at architecture time. **Attention dropout**, when used, is applied independently per head during training only. At inference, all heads' KV-caches are maintained independently; total cache grows as $h \times O(n d_h) = O(n d_{\text{model}})$ — same as the single-head case.

**Regime note — head-level pruning.** At inference some deployments prune "unused" heads (those producing near-zero attention variance) for compute savings. This is a post-training intervention, not a change to the operator, and it is typically lossy.

## 5.3 Rigorous core

**Definition 5.1 (Multi-head attention).** Let $d_{\text{model}} = h \cdot d_h$. Define
$$\text{head}_i(X) = \text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}),$$
with $W_Q^{(i)}, W_K^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_h}$ and $W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_h}$. Then
$$\text{MHA}(X) = [\text{head}_1(X) | \cdots | \text{head}_h(X)] \, W_O$$
with $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

**Proposition 5.2 (MHA subsumes single-head).** For any single-head attention operator with parameters $(W_Q, W_K, W_V, W_O')$ at full dimension $d_{\text{model}}$, there exist MHA parameters realizing the same function. (Set $h = 1$ trivially.)

**Proposition 5.3 (FLOP equivalence).** MHA with $h$ heads at dimension $d_h = d_{\text{model}} / h$ and single-head attention at dimension $d_{\text{model}}$ have the same asymptotic FLOP count $O(n^2 d_{\text{model}} + n d_{\text{model}}^2)$ (attention + projection costs). The constant factors differ slightly due to the split/concat reshapes but not asymptotically.

*Sketch: per head $O(n^2 d_h)$ attention + $O(n d_{\text{model}} d_h)$ projection; summing over $h$ heads turns each $d_h$ into $d_{\text{model}}$.*

**Empirical observation 5.4 (Head specialization).** In trained transformers, heads exhibit distinct attention patterns: some attend to the previous token (distance-1 heads), some to the first token (sink heads, Ch. 16), some to grammatically-linked tokens (subject-verb heads), some copy from matching context (induction heads, Ch. 15). This specialization is not enforced by architecture and can be disrupted by pruning or re-initialization. It is the primary reason multi-head beats single-head in practice.

**Empirical observation 5.5 (Head redundancy).** In the same trained models, many heads are highly redundant — prunable without measurable loss of capability. Typical findings in the literature: 20–50% of heads in a 12-layer transformer can be pruned layerwise without affecting downstream metrics. This coexists with Empirical obs 5.4 — heads specialize, but the specialization space is lower-rank than $h$ heads would suggest.

**Proposition 5.6 (Head mixing by $W_O$).** The output projection $W_O$ can mix information freely across head outputs. Without $W_O$ (identity), heads contribute to disjoint coordinate blocks of the output; with $W_O$ learned, any linear combination of head outputs is reachable. This is why $W_O$ is architecturally non-optional.

## 5.4 Numerical example — $h = 2, d_{\text{model}} = 4, n = 3$ by hand

Take $X$ such that the two heads should naturally specialize:
$$X = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}.$$

Split the feature dim 4 into two heads of dim 2 — head 1 uses coords $(0, 1)$, head 2 uses coords $(2, 3)$ (this is what $W_Q^{(1)}, W_Q^{(2)}$ would do with the natural block-diagonal init).

**Head 1 sees** (coords 0, 1): $Q_1 = K_1 = V_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}$. Tokens 0 and 2 are identical; they attend strongly to each other and themselves.

**Head 2 sees** (coords 2, 3): $Q_2 = K_2 = V_2 = \begin{pmatrix} 1 & 0 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$. Tokens 0 and 1 are identical; they attend strongly to each other.

**Specialization.** The two heads produce *different* attention patterns from the *same* input, by looking at different subspaces. After concat + $W_O$, the block has access to both patterns at once. This is the toy version of the real empirical phenomenon.

## 5.5 Mechanics check

- **Split $\to$ attention $\to$ concat.** Logically equivalent to $h$ parallel independent attentions. Implemented in practice by a single matmul for $W_Q$ (producing $(n, h \cdot d_h)$) followed by reshape to $(h, n, d_h)$, so the "h" dimension is a batch dimension for the attention core.
- **Per-head $\sqrt{d_k}$ scaling.** Each head uses $\sqrt{d_h}$, not $\sqrt{d_{\text{model}}}$. Critical — scaling by the wrong dimension breaks Chapter 3's argument per head.
- **$W_O$ mixes heads.** Without $W_O$, each head's output would live in its own disjoint coordinate block; $W_O$ is the linear map that lets the block output be any mixture of the $h$ patterns.
- **Parameter count.** Each of $W_Q, W_K, W_V, W_O$ is $d_{\text{model}} \times d_{\text{model}}$ → total $4 d_{\text{model}}^2$ parameters. Unchanged whether $h$ is 1 or 16 (the heads just reshape the same matrix).
- **Why $h$ is not arbitrary.** At fixed $d_{\text{model}}$, larger $h$ means smaller $d_h$. Once $d_h$ is small (< 16 or so), individual heads cannot represent rich enough attention patterns — expressivity per head collapses. This is why typical choices are $d_h \in [64, 128]$.

## 5.6 Exercises

**5.1 ★★** *[Mechanism dissection — why $W_O$ is necessary.]* Consider MHA without $W_O$ (i.e., $W_O = I$). Show that each head's output contributes to a disjoint coordinate block of the output. Conclude that information flow between heads within a single block is impossible without $W_O$. Then argue why this is catastrophic for expressivity, by constructing a function that a full-rank $W_O$ achieves but $W_O = I$ cannot.

**5.2 ★★** *[Failure-mode construction — too many heads.]* Fix $d_{\text{model}} = 512$ and vary $h \in \{1, 4, 16, 64, 256, 512\}$. For each, compute $d_h = d_{\text{model}} / h$. Identify the threshold at which $d_h$ is too small for the head to express useful attention patterns. Justify: why would $d_h = 2$ be fundamentally insufficient, even for a simple retrieval task? (Hint: the attention matrix is rank-$\min(n, d_h)$ — Exercise 3.2 style analysis on $W_Q^\top W_K$.)

**5.3 ★★★** *[Theorem about the numerics — head capacity and rank.]* Let $S^{(i)} = Q_i K_i^\top / \sqrt{d_h}$ be the pre-softmax score matrix for head $i$.

(a) Prove $\text{rank}(S^{(i)}) \le d_h$.

(b) Prove that the post-softmax attention matrix $A^{(i)} = \text{softmax}(S^{(i)})$ has *full rank* $n$ generically (all $n$ eigenvalues are nonzero almost surely over random $Q_i, K_i$), despite $S^{(i)}$ being low-rank.

(c) Explain the apparent paradox: how can softmax of a low-rank matrix be full-rank? What non-linearity is doing the lifting, and what does this say about why the softmax is architecturally non-optional?

**5.4 ★★★** *[Predict-then-verify — head specialization on a synthetic task.]* The notebook will train a 1-layer 4-head transformer on a synthetic "retrieve-from-context" task: given a sequence ending in a query token, retrieve the value associated with the matching key earlier in the context. Predict, before running:

(a) Will all 4 heads end up doing the same thing (because there is only one task), or will specialization emerge (different heads attending to different structural features)?

(b) If specialization emerges, how many heads actually matter for the task? (Related: would the same task be solvable with $h = 1$ at dimension $4 d_h$?)

(c) After training, which head is most "important" by ablation (zeroing it destroys performance)? Predict the importance ranking.

(d) Pruning prediction: at what $h_{\text{retained}}$ does performance degrade to chance?

State confidence per part.

## 5.7 Before the notebook (`05_heads_specialize.ipynb`)

Write these predictions before opening the notebook.

1. **Parameter count check.** Total parameters in MHA with $h = 1$ vs. $h = 8$ at $d_{\text{model}} = 128$. Same? Different? If same, why?
2. **FLOP count scaling.** Same total FLOPs across $h \in \{1, 2, 4, 8\}$?
3. **Toy-task specialization.** On the hand-crafted 5-token sequence from Ch. 1 experiment 4, replayed with $h = 2$ heads — does each head specialize to a different signal?
4. **Rank of attention matrix.** For $h = 4$, $d_h = 8$, $n = 32$: pre-softmax score matrix rank (predict); post-softmax attention matrix rank (predict). Does softmax lift the rank?
5. **Random-head ablation.** In a $h = 8$ MHA block with random initialization, ablating 4 of 8 heads should give roughly which fraction of the unperturbed output magnitude?

For each: direction, magnitude, confidence.
