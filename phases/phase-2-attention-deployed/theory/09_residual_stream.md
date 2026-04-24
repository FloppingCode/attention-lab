# Chapter 9 — Stacking: the residual stream view

```
Type: theory + notebook
Ordering: mixed
Depends on: 7 (LN + residuals), 8 (FFN)
```

## 9.1 Motivation

A transformer is *not* a deep composition. It is a *running sum*. The residual connections turn each layer into a small additive contribution to a shared vector at each token position — the **residual stream**. This reframing — due to Anthropic's mechanistic interpretability work (Elhage et al., 2021) — changes how you think about every other component. Heads and FFN blocks are *writers* to specific subspaces of the residual stream; downstream operators are *readers* of those subspaces. Once you have this picture, induction heads, attention sinks, and logit-lens-style interpretability all make immediate sense — they are statements about who writes what to which subspace at what depth.

## 9.2 Storyline

**The picture.** At each token position $t$, there is a vector $r^{(\ell)}_t \in \mathbb{R}^{d_{\text{model}}}$ — the residual stream at depth $\ell$. The recurrence is
$$r^{(\ell+1)}_t = r^{(\ell)}_t + \text{Attn}_\ell(r^{(\ell)}_{1:t}) + \text{FFN}_\ell(r^{(\ell)}_t),$$
(in the pre-norm formulation, with LN inside the sub-layers). Each layer adds a small contribution. The final residual stream is read out by an unembedding $W_U$ to logits.

**The model adds, it does not transform.** At initialization the per-layer contributions $a^{(\ell)}, m^{(\ell)}$ are small, so $r^{(L)} \approx r^{(0)} + \text{(small accumulated perturbation)}$ — the network is essentially a pass-through. Training does not learn to overwrite the input; it learns *what corrections to add* at each depth. The mental model "deep stack of compositions" is wrong; "running sum of small contributions to a shared bus" is right. This is the single reframing that makes everything else in this chapter (residuals as communication, low-rank reads/writes, mech interp) follow naturally.

**Each sub-layer reads and writes.** Attention reads from the residual streams at all positions $\le t$ (via $W_Q, W_K, W_V$), processes, and writes back to position $t$ (via $W_O$). The "reading" is a low-rank projection (rank $\le d_h$ per head); the "writing" is similarly low-rank. The FFN reads from the position-$t$ residual stream (via $W_1$) and writes back (via $W_2$); it is a sum of $d_{\text{ff}}$ rank-1 contributions (Prop 8.5).

**The residual stream as a communication bus.** Every layer can see what every previous layer wrote — because nothing is overwritten. Layer 7 can use information layer 2 wrote, *unmediated by intervening layers*, as long as the relevant subspace was preserved. This is why mech interp works: features are in *specific, identifiable subspaces*, and the layers that care about them have learned to read those subspaces directly.

**Subspace separation and feature superposition.** $d_{\text{model}}$ is too small to give every feature its own dimension — there are $\gg d_{\text{model}}$ interpretable features in a typical LLM. The model resolves this via *superposition*: features share dimensions, encoded as nearly-orthogonal directions. Reading a feature is approximately recovering one of these directions; writing is adding it. This is a separate research thread (sparse autoencoders, dictionary learning), not Phase II material — but the residual-stream picture is its foundation.

**Logit lens (Chapter 17 preview).** Apply the unembedding $W_U$ to *intermediate* residual streams $r^{(\ell)}_t$, not just the final one. The resulting logits show what next-token distribution the model would predict if generation stopped at layer $\ell$. This is the simplest possible interpretability tool, and it works *only because* of the residual-stream picture: the unembedding is meaningful at every depth because the residual stream lives in the same space throughout.

**Train / val / inference.** No regime asymmetry in the residual stream itself. The KV-cache at inference (Ch. 4) is a cached version of the keys and values that attention layers extracted from the residual stream at past positions. Nothing about the residual-stream view changes between training and inference.

## 9.3 Rigorous core

**Definition 9.1 (Residual stream).** For a pre-norm transformer with $L$ blocks, the residual stream at layer $\ell$ and token position $t$ is
$$r^{(\ell+1)}_t = r^{(\ell)}_t + a^{(\ell)}_t + m^{(\ell)}_t,$$
with $a^{(\ell)}_t = \text{Attn}_\ell(\text{LN}(r^{(\ell)}))_t$ and $m^{(\ell)}_t = \text{FFN}_\ell(\text{LN}(r^{(\ell)}_t))$. Initial state $r^{(0)}_t = \text{Embed}(\text{token}_t) + \text{PE}_t$ (positional encoding for the chosen scheme).

**Proposition 9.2 (Telescoping decomposition).** Unrolling the recurrence,
$$r^{(L)}_t = r^{(0)}_t + \sum_{\ell=0}^{L-1} a^{(\ell)}_t + \sum_{\ell=0}^{L-1} m^{(\ell)}_t.$$
The final residual stream is the sum of the input embedding and the contributions of every attention and FFN block.

*Induction on $\ell$.*

**Corollary 9.2.1 (Per-component contribution).** For any analysis tool $\Phi$ that is linear in $r^{(L)}_t$ (e.g., the unembedding $W_U$, a linear probe), $\Phi(r^{(L)}_t)$ decomposes additively over the $2L + 1$ components — embedding, $L$ attention blocks, $L$ FFN blocks.

**Proposition 9.3 (Low-rank read/write).** A single attention head with head dimension $d_h$ reads from the residual stream via $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_h}$ (rank $\le d_h$ each) and writes via $W_O^{(h)}$ (the slice of $W_O$ corresponding to head $h$, also rank $\le d_h$). The information flowing through one head is bottlenecked by $d_h \ll d_{\text{model}}$.

**Empirical observation 9.4 (Skip connections survive training).** In trained transformers, the contribution $r^{(0)}_t$ (the original token embedding) is still present and identifiable in $r^{(L)}_t$, not "forgotten." Probes trained to recover the original token from the final residual stream achieve high accuracy. This is why the unembedding can map $r^{(L)}_t$ back to token space at all — the token's identity has not been overwritten.

**Empirical observation 9.5 (Attention-head and FFN-cell features).** In trained transformers, individual heads can be characterized by what subspace of the residual stream they read from and write to. Examples (Anthropic's induction-heads paper): "previous-token heads" read positional info and write a copy of the previous token's embedding to the current position; "induction heads" read pattern-match keys and write the predicted next-token embedding. This per-head specialization is the substrate of mechanistic interpretability.

**Empirical observation 9.6 (Norm growth).** $\|r^{(\ell)}_t\|$ grows monotonically with $\ell$ in pre-norm transformers (already seen in Ch. 7 Empirical obs 7.5). The growth is approximately $O(\sqrt{L})$ at init under random-walk arguments, faster after training.

## 9.4 Numerical example

No separate hand example — the residual-stream picture is geometric and the interesting numbers come from running the notebook (logit lens across depth, layer ablation, contribution norms). The chapter's content is the structural decomposition (Prop 9.2) plus the empirical observations 9.4–9.6, all of which are verified in the notebook.

## 9.5 Mechanics check

- **The residual stream is a shared vector space at each position.** All layers operate in $\mathbb{R}^{d_{\text{model}}}$ — the same space throughout the network. This is why the unembedding $W_U$ can be applied at any depth (logit lens).
- **No layer overwrites; every layer adds.** This is the mechanism. Composition would discard prior information; addition preserves it modulo norm.
- **Each head/FFN cell is low-rank.** Reads and writes are bottlenecked by $d_h$ (heads) or $1$ (FFN cells, per Prop 8.5). No single sub-component touches the full $d_{\text{model}}$ dimensions.
- **Linear analysis composes additively.** Logit lens, linear probes, attribution methods — all decompose cleanly because the final state is a sum.
- **Superposition is possible because $d_{\text{model}} \ll \text{number of features}$.** The residual stream has finite capacity; many more features than dimensions are encoded by sharing directions in nearly-orthogonal ways. This is a separate research direction — the substrate is the residual stream view.
- **Reads and writes happen *through* LN in pre-norm.** LN's null space (shift, scale per token) means heads cannot read or write into those two directions. Whether this matters in practice is an open question.

## 9.6 Exercises

**9.1 ★★** *[Mechanism dissection — additive decomposition of the logits.]* By Prop 9.2 and Corollary 9.2.1, the final logits decompose as $W_U r^{(L)}_t = W_U r^{(0)}_t + \sum_\ell W_U a^{(\ell)}_t + \sum_\ell W_U m^{(\ell)}_t$.

(a) Write out the decomposition for a 2-layer model. There are 5 terms; identify each.

(b) For a token-prediction task, which terms would you expect to dominate at convergence? The embedding contribution? Late-layer contributions? Early-layer contributions? Justify with an interpretability argument (do early layers usually contribute much to *output* logits?).

(c) The logit-lens technique applies $W_U$ to $r^{(\ell)}$ for $\ell < L$. Argue why this is meaningful, and what failure modes you'd expect (when does the intermediate residual stream *not* look like a useful logit prediction?).

**9.2 ★★** *[Failure-mode construction — residual collapse.]* Imagine a transformer where every block writes the same vector $w \in \mathbb{R}^{d_{\text{model}}}$ regardless of input. (Far-fetched, but useful as a thought experiment.) After $L$ blocks, $r^{(L)} = r^{(0)} + L w$. As $L \to \infty$, the residual stream is dominated by $L w$ — the embedding contribution becomes negligible. State a precise version of this collapse and argue why architectures avoid it (LN before unembedding rescales the contribution; in trained models, individual block writes are not constant). Construct a residual-stream norm growth pattern that *would* cause collapse, and one that wouldn't.

**9.3 ★★★** *[Theorem about the geometry — superposition capacity.]* Suppose we want to encode $N$ features in the residual stream of dimension $d = d_{\text{model}}$, each as a unit-norm direction $v_i$, such that "reading feature $i$" via inner product $\langle v_i, r \rangle$ gives a clean signal.

(a) For $N = d$ features encoded as orthogonal directions, the reads are perfectly clean. State the maximum number of *exactly* orthogonal directions in $\mathbb{R}^d$.

(b) For $N > d$ features, exact orthogonality is impossible. Define the "interference" as $\max_{i \ne j} |\langle v_i, v_j \rangle|$. Use the Welch bound or the Johnson-Lindenstrauss lemma to give a relationship between $N$, $d$, and the achievable interference $\epsilon$.

(c) Conclude: how many features can be encoded with bounded interference $\epsilon < 0.1$ in a $d_{\text{model}} = 768$ residual stream? Compare to the number of "interpretable features" reported in the SAE literature for similar-sized models (often $> 10^4$). Where is the slack?

(d) Bonus: superposition only works for *sparsely-active* features. State the connection in one paragraph — why does sparsity in feature activation rescue capacity?

**9.4 ★★★** *[Predict-then-verify — logit-lens trajectory.]* The notebook will train a small 4-layer transformer on a next-token prediction task, then apply the logit lens at each layer. Predict, before running:

(a) At layer 0 (just the embedding), what does the logit-lens prediction look like? Random? The input token itself? Something else?

(b) At layer 1, layer 2, layer 3, layer 4: how does the prediction evolve? Does it monotonically improve toward the correct answer, or does it oscillate?

(c) For a task where the correct prediction requires multiple inference steps (e.g., $a + b = c$, where $c$ depends on integrating both $a$ and $b$), which layer would you expect to first "have" the answer in its logit-lens projection?

(d) Bonus: ablate one layer's contribution (remove $a^{(\ell)} + m^{(\ell)}$ from the sum) and see how downstream layers' logit-lens predictions change. Predict whether removing layer 1 is more disruptive than layer 3.

State confidence per part.

## 9.7 Before the notebook (`09_residual_stream.ipynb`)

The notebook's core experiment is a **2-hop lookup task**: given a sequence of `(key, value)` pairs and a query `q`, the model must output `v_2 = lookup(lookup(q))` — the value of the value of the query. This is a task where depth has to do real work; the 1-hop intermediate `v_1 = lookup(q)` is a meaningful midway answer that the logit lens can detect separately. Predictions:

1. **Telescoping decomposition at init.** For a 4-layer pre-norm transformer, predict $\|r^{(L)} - r^{(0)}\|$ vs. $L$. Linear, $\sqrt{L}$, or constant?
2. **Per-component contribution norm.** Predict relative magnitudes of $\|a^{(\ell)}\|$ vs. $\|m^{(\ell)}\|$ at init and after training. Are they comparable, or does training shift the balance?
3. **2-hop training accuracy.** Can a 4-layer model solve the 2-hop task with random distractors? Predict final accuracy. (Chance is $1/V$.)
4. **Logit-lens depth localization.** This is the key prediction. At each depth $\ell \in \{0, 1, 2, 3, 4\}$, predict (a) the accuracy of the lens-projected query prediction against the **1-hop intermediate** $v_1$, and (b) against the **2-hop answer** $v_2$. Sketch both curves. Where do they each peak? Does 1-hop accuracy precede 2-hop accuracy? Does 1-hop *decay* once 2-hop appears (which would weaken the "nothing is overwritten" mental model)?
5. **Per-layer ablation.** Removing layer $\ell$'s contribution: which layers hurt the 1-hop most? The 2-hop most? Is there a clean assignment of "this layer does hop 1, this layer does hop 2," or is the computation distributed?

For each: direction, magnitude, confidence.
