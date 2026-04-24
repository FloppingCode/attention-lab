# Chapter 4 — Causal masking and attention at inference

```
Type: theory + notebook
Ordering: theory-first
Depends on: 1 (scaled dot-product attention), 2 (softmax)
```

## 4.1 Motivation

Autoregressive generation requires that token $i$'s representation depend only on tokens $\{1, \dots, i\}$, never on tokens $> i$. Plain self-attention is permutation-equivariant over sources (Proposition 1.2) and therefore freely attends both forward and backward. The *causal mask* is the additive modification that restricts attention to the causal half-plane. It is also the modification that makes the training-vs-inference asymmetry visible: at training, attention for all $n$ positions is computed in one parallel pass; at inference, positions are generated sequentially and the key/value cache exploits the causal structure for per-token-constant work.

## 4.2 Storyline

The mask is a matrix $M \in \{0, -\infty\}^{n \times m}$ added to the pre-softmax scores:
$$\text{MaskedAttention}(Q, K, V; M) = \text{softmax}\!\left( \frac{QK^\top}{\sqrt{d_k}} + M \right) V.$$

For causal self-attention, $M_{ij} = -\infty$ for $j > i$, else $0$. After softmax, the $-\infty$ entries become $0$ and the allowed entries renormalize over the allowed positions — the operation is equivalent to running softmax on the sub-row $s_{i, 1:i}$ and zero-padding.

**Why additive, not multiplicative.** Multiplying scores by a $\{0, 1\}$-mask would also kill the forbidden entries, but after softmax they would receive $e^0 = 1$ mass each, not zero. Additive $-\infty$ composes cleanly with softmax's shift invariance (Prop 2.2): the normalizer excludes forbidden positions.

**Training (parallel).** Given an input sequence $X \in \mathbb{R}^{n \times d}$ the full attention matrix $A = \text{softmax}(QK^\top / \sqrt{d_k} + M) \in \mathbb{R}^{n \times n}$ is computed in one shot. Entries above the diagonal are zero by the mask. All $n$ output positions come out of a single matrix product. This is the central reason transformers are efficient to train: there is no sequential loop over positions.

**Inference (sequential + KV-cache).** At generation time the model emits one token per step. For step $t$, the query is only the current token's query $\mathbf{q}_t \in \mathbb{R}^{d_k}$; keys and values for all prior positions $K_{1:t}, V_{1:t}$ have been computed in previous steps and are stored in the **KV-cache**. Step $t$'s attention computes $\mathbf{o}_t = \text{softmax}(\mathbf{q}_t K_{1:t}^\top / \sqrt{d_k}) V_{1:t}$ — no mask needed (the cache contains only past positions by construction), FLOPs scale as $O(t \cdot d_k)$ per step, and memory scales as $O(t \cdot (d_k + d_v))$ per layer per head. The cache grows monotonically with generated length.

**The training/inference equivalence.** The operator computes the same thing in both regimes: the output at position $t$ depends only on $\{x_1, \dots, x_t\}$. Training just does all $t$ at once on a known sequence. Inference does them one at a time on a sequence being generated. Causal masking at training is the consistency enforcement — without it, training would learn to use future tokens and inference would diverge from the training distribution.

**Regime notes, compactly:**

| Regime | Batching | KV storage | Mask |
|---|---|---|---|
| Training (teacher-forced) | all $n$ positions at once | none (recomputed) | explicit upper-triangular $-\infty$ |
| Validation | same as training | none | same |
| Inference (autoregressive) | one query per step | cached, grows with sequence | implicit (cache contains only past) |
| Inference (prefill — processing a prompt) | full prefix in one pass | cached after | explicit for the prefix, then implicit |

## 4.3 Rigorous core

**Definition 4.1 (Causal mask).** $M^{\text{causal}} \in \mathbb{R}^{n \times n}$ with
$$M^{\text{causal}}_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i. \end{cases}$$
Equivalently: a strictly upper-triangular matrix of $-\infty$'s with zeros elsewhere.

**Proposition 4.2 (Causal equivalence).** With causal mask $M^{\text{causal}}$, the attention at position $i$ is
$$A_{i, j} = \begin{cases} \dfrac{e^{s_{ij}}}{\sum_{j' \le i} e^{s_{ij'}}} & j \le i \\ 0 & j > i, \end{cases}$$
where $s_{ij}$ is the scaled score. Equivalently, $A_{i, 1:i}$ equals $\text{softmax}(s_{i, 1:i})$ and $A_{i, i+1:n} = 0$.

**Proof.** $e^{-\infty} = 0$ under the usual convention. The normalizer therefore sums only over $j \le i$. The $e^{s_{ij}}/Z_i$ entries are standard softmax on the sub-row. $\blacksquare$

**Corollary 4.2.1 (Temporal locality).** The output row $\mathbf{o}_i$ depends only on $\{Q_i, K_{1:i}, V_{1:i}\}$ — the full sequence's future is computationally irrelevant.

**Proposition 4.3 (KV-cache equivalence).** Let the model be trained with causal masking. At inference with KV-cache, step $t$ produces $\mathbf{o}_t = \text{softmax}(\mathbf{q}_t K_{1:t}^\top / \sqrt{d_k}) V_{1:t}$. This equals the $t$-th row of $\text{MaskedAttention}(Q_{1:t}, K_{1:t}, V_{1:t}; M^{\text{causal}})$ exactly — no approximation.

**Proof.** By Corollary 4.2.1, the $t$-th row depends only on $Q_t, K_{1:t}, V_{1:t}$. KV-caching reuses the exact same $K_{1:t}, V_{1:t}$. Row indexing of the full attention matrix gives the same output. $\blacksquare$

**Corollary 4.3.1 (FLOPs and memory).** Generating a sequence of length $n$ with KV-cache costs $O(\sum_{t=1}^n t \cdot d_k) = O(n^2 d_k)$ FLOPs per layer per head for attention, with $O(n (d_k + d_v))$ peak memory. Without KV-cache (recomputing $K, V$ each step) it would cost $O(n^3 d_k)$ FLOPs.

**Empirical observation 4.4 (Attention sinks and the first token).** Even in a perfectly trained causal-masked model, the first token typically receives disproportionate attention from later queries — a phenomenon called the "attention sink" (studied in Chapter 16). The mechanism: token 1 is the only position with no distractors in its own computation (its softmax has just one key), and later queries learn to use token 1 as a "null option" for mass they don't want to place elsewhere. The causal mask does not cause this, but it *enables* it by giving the first position a structurally privileged role.

## 4.4 Numerical example — $n = 4$ by hand

Take $n = 4$, $d_k = 2$, random small-integer scores. Pre-softmax score matrix (already scaled by $1/\sqrt{d_k}$):
$$S = \begin{pmatrix} 0 & 1 & -1 & 2 \\ 1 & 0 & 2 & 1 \\ 0 & 2 & 1 & -1 \\ -1 & 1 & 0 & 2 \end{pmatrix}.$$

**Masking.** Set upper triangle to $-\infty$:
$$S + M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 1 & 0 & -\infty & -\infty \\ 0 & 2 & 1 & -\infty \\ -1 & 1 & 0 & 2 \end{pmatrix}.$$

**Row-wise softmax:**
- Row 1: single entry → $(1, 0, 0, 0)$.
- Row 2: $\text{softmax}(1, 0) \approx (0.731, 0.269)$ → $(0.731, 0.269, 0, 0)$.
- Row 3: $\text{softmax}(0, 2, 1) = (e^0, e^2, e^1) / Z_3$. $Z_3 \approx 1 + 7.389 + 2.718 = 11.107$. Row $\approx (0.090, 0.665, 0.245, 0)$.
- Row 4: $\text{softmax}(-1, 1, 0, 2)$. Shift by 2: $\text{softmax}(-3, -1, -2, 0) = (e^{-3}, e^{-1}, e^{-2}, 1) / Z_4$. $Z_4 \approx 0.050 + 0.368 + 0.135 + 1 = 1.553$. Row $\approx (0.032, 0.237, 0.087, 0.644)$.

**Check — lower-triangular structure.** All zeros above the diagonal. ✓

**Check — rows sum to 1.** Each row sums to 1 (softmax over the allowed positions). ✓

## 4.5 Mechanics check

- **Additive $-\infty$, not multiplicative $0$.** Composes with softmax's shift invariance: forbidden entries are *removed from the normalizer*, not rescaled to zero after normalization.
- **Row-wise application.** The mask acts per-query. Each row has a different allowed set, different normalizer, different softmax.
- **Training vs. inference FLOP count.** Training: $O(n^2 d_k)$ per sequence per head (matrix product $QK^\top$ dominates). Inference with KV-cache: $O(n^2 d_k)$ *total* across all generation steps (each step is $O(t d_k)$, summed). Identical asymptotic — the cache makes the cost profile match training's.
- **Memory asymmetry.** Training stores activations for backprop: $O(n^2)$ per-layer attention matrix (before softmax) and $O(n d)$ activations. Inference stores only KV-cache: $O(n (d_k + d_v))$ per layer per head. No attention matrix needs to be materialized.
- **The mask is not learned.** It is fixed by the architecture (causal for decoder-only, bidirectional for encoder-only, custom for encoder-decoder cross-attention). This is a structural choice, not a parameter.
- **Why inference doesn't need the mask.** When the KV-cache contains only past positions by construction, there are no future positions for the mask to hide. The mask is an *enforcement of a constraint that is already physically true* during generation.

## 4.6 Exercises

**4.1 ★★** *[Mechanism dissection — mask composition.]* Suppose you want a *local* causal mask: attention is causal and additionally restricted to a window of size $w$ (so position $i$ can attend to positions $\max(1, i-w+1), \dots, i$). Write down $M^{\text{local}}$ in terms of a causal mask and a local-window mask. Prove that the composition is valid (gives a well-defined softmax distribution) for all $w \ge 1$. For which $w$ does the first token $i = 1$ still have a valid attention distribution, and which doesn't?

**4.2 ★★** *[Failure-mode construction — mask leakage.]* A common implementation bug is to apply the mask *after* softmax (multiplicatively by a $\{0, 1\}$-mask). Show that this produces a distribution that does *not* sum to 1 for rows with any masked entries. Compute the row-sum in closed form for the causal mask case. Propose a (wrong but plausible) "fix" that re-normalizes after masking — and show that this fix produces *different* gradients than the correct additive mask, in a way that biases training toward ignoring the masked positions in a specific way. What's the bias?

**4.3 ★★★** *[Theorem about the numerics — KV-cache staleness.]* In practice, the KV-cache is sometimes *approximated* at inference — quantized to fp8, truncated to a sliding window, or compressed via low-rank projections. Let $\tilde{K}, \tilde{V}$ denote the approximated cache with $\|K - \tilde{K}\|_F \le \epsilon_K, \|V - \tilde{V}\|_F \le \epsilon_V$.

(a) Bound the output error $\|\mathbf{o}_t - \tilde{\mathbf{o}}_t\|_2$ in terms of $\epsilon_K, \epsilon_V$, $\|\mathbf{q}_t\|_2$, the softmax Jacobian norm, and $d_k$.

(b) Explain which of the two errors ($\epsilon_K$ or $\epsilon_V$) is amplified by softmax saturation (sharp attention distribution) and which is amplified by near-uniform attention. *Hint: $\epsilon_V$ enters linearly after softmax; $\epsilon_K$ enters through the Jacobian.*

(c) Use (b) to argue which cache component (keys or values) is more safely aggressive-quantized at long context lengths.

**4.4 ★★★** *[Predict-then-verify — attention patterns at inference.]* The notebook will load or construct a small decoder-only transformer and visualize the attention matrix at different generation steps on a prompt like "The capital of France is".

Predict, before running:

(a) Qualitatively, what should the attention pattern look like for the query at the *generated* position (the next-token prediction)? Where do you expect mass to concentrate — on the last token, on content tokens, on the first token (attention-sink)?

(b) How does the pattern change as the sequence grows? Specifically, is attention "used up" by distant context, or does later generation still attend heavily to early positions?

(c) Construct a prediction grid for three layer depths (shallow, middle, deep). Which depth should show the most semantically-targeted attention (attending to the relevant content token "Paris" precursor)? Justify from mental models of mechanistic interpretability (Chapter 15 preview).

State confidence per part with a one-line reason.

## 4.7 Before the notebook (`04_causal_masking.ipynb`)

Write these predictions before opening the notebook.

1. **Mask correctness.** The output of a correctly causal-masked attention should satisfy $A_{i,j} = 0$ for $j > i$ to floating-point precision. Predict any source of deviation.
2. **FLOP scaling, training vs. KV-cached inference.** Benchmarking $n$ from 8 to 1024, the total inference FLOPs (summed across all generation steps) should match training FLOPs asymptotically. Predict the ratio at $n = 1024$.
3. **Output equivalence.** Inference with KV-cache should produce the *exact* same output as running the full-matrix causal attention, up to floating-point precision (Prop 4.3). Predict the expected max-abs error.
4. **Attention-pattern prediction on a toy sequence.** Given five tokens where token 3 is structurally similar to tokens 1 and 2, predict the attention row for the query at position 3 and at position 5.
5. **KV-cache memory scaling.** Predict the cache size in bytes at $n = 4096, h = 16, d_k = 64, L = 32$ layers in fp16. Order of magnitude.

For each: direction, magnitude, confidence.
