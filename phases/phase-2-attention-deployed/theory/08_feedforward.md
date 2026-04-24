# Chapter 8 — The feedforward block

```
Type: theory + notebook
Ordering: theory-first
Depends on: 1 (attention), 7 (LN + residuals)
```

## 8.1 Motivation

Each transformer block has *two* sub-layers: attention, then a per-token feedforward block (FFN). The attention sub-layer is constrained to convex combinations of linearly-projected value rows (Corollary 1.3.1) — it cannot synthesize non-linear functions of a single token's features. The FFN is the per-token non-linearity. Without it, the network is a stack of linear-then-soft-mix layers — universal-approximation-broken. Empirically, the FFN holds about two thirds of a transformer's parameters and a similar fraction of its compute. It is not a side dish.

## 8.2 Storyline

The FFN is a per-token MLP with one hidden layer:
$$\text{FFN}(x) = W_2 \, \phi(W_1 x + b_1) + b_2,$$
with $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, and $\phi$ a non-linear activation (GELU, SiLU/Swish, ReLU). Hidden dimension $d_{\text{ff}}$ is typically $4 d_{\text{model}}$ — the wide intermediate is a load-bearing design choice.

**Why per-token.** The FFN does *not* mix across positions. Each token's representation is independently mapped through the same MLP. All inter-token mixing happens in attention; all intra-token nonlinear transformation happens in the FFN. This separation of concerns is what makes the residual-stream-as-communication-channel mental model (Ch. 9) clean.

**Why wide ($d_{\text{ff}} = 4 d_{\text{model}}$).** The FFN can be viewed as a *key-value memory*: rows of $W_1$ are "keys" matched against the input, $\phi(\cdot)$ thresholds the matches, and rows of $W_2$ are "values" added to the output. Width $d_{\text{ff}}$ is the number of memory slots. Empirically, $4 \times$ is a sweet spot: enough capacity to be expressive, not so wide it dominates parameter count uncontrollably. Modern variants (gated FFN: SwiGLU, GeGLU) use slightly smaller $d_{\text{ff}} \approx 2.7 d_{\text{model}}$ for matching FLOP count.

**Why GELU/SiLU and not ReLU.** ReLU is fine; GELU and SiLU are *smoother* and have been shown empirically to train slightly faster and reach slightly better optima for transformers. The smoothness affects the per-coordinate Jacobian and therefore the conditioning of the optimization. Activation choice is one of the smaller hyperparameter levers in transformer design — meaningful but not load-bearing.

**Gated variants (SwiGLU, GeGLU).** Replace $W_1 x \mapsto \phi(W_1 x)$ with $W_3 x \odot \phi(W_1 x)$ (Hadamard product with a separate gate projection). Adds parameters but improves quality at fixed FLOP budget. The gate is the model's per-coordinate "should this hidden unit fire" decision, multiplicatively combined with the activation. Most modern LLMs use SwiGLU.

**Train / val / inference.** The FFN is identical across regimes — pure feedforward, no batch statistics, no caching. Dropout (when used) is applied between $W_1$ and $W_2$ at training only. No KV-cache needed; no parallelism subtlety. The simplest sub-layer in the whole transformer.

## 8.3 Rigorous core

**Definition 8.1 (Vanilla FFN).** $\text{FFN}(x) = W_2 \phi(W_1 x + b_1) + b_2$ for $x \in \mathbb{R}^{d_{\text{model}}}$.

**Definition 8.2 (SwiGLU FFN).** $\text{SwiGLU}(x) = W_2 (\text{SiLU}(W_1 x) \odot W_3 x)$, with $\text{SiLU}(z) = z \sigma(z) = z / (1 + e^{-z})$.

**Proposition 8.3 (FFN as universal per-token approximator).** For sufficiently wide $d_{\text{ff}}$, the FFN can approximate any continuous function $f: \mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{d_{\text{model}}}$ on a compact set to arbitrary accuracy, by the universal approximation theorem. Combined with attention (which mixes across positions), the transformer block is a universal approximator over sequences.

**Proof sketch.** Universal approximation for one-hidden-layer MLPs with non-polynomial activations is classical (Cybenko, Hornik, Pinkus). The FFN is exactly such an MLP applied per token.

**Proposition 8.4 (FFN parameter and FLOP budget).** Vanilla FFN parameters: $2 \cdot d_{\text{model}} \cdot d_{\text{ff}}$ (ignoring biases). FLOPs per token: $4 \cdot d_{\text{model}} \cdot d_{\text{ff}}$ (forward; multiply for backward). Total per-block parameters: $4 d_{\text{model}}^2$ (attention) + $2 d_{\text{model}} d_{\text{ff}}$ (FFN). At $d_{\text{ff}} = 4 d_{\text{model}}$: FFN params = $8 d_{\text{model}}^2$, attention params = $4 d_{\text{model}}^2$. **The FFN is twice the parameter count of attention per block.**

**Proposition 8.5 (FFN as key-value memory, formal version).** Write $W_1 = [k_1; \dots; k_{d_{\text{ff}}}]$ as rows (the "keys") and $W_2 = [v_1, \dots, v_{d_{\text{ff}}}]$ as columns (the "values"). Then
$$\text{FFN}(x) = \sum_{i=1}^{d_{\text{ff}}} \phi(\langle k_i, x \rangle + b_{1,i}) \cdot v_i + b_2.$$
Each hidden unit $i$ is a content-addressable "memory cell": its activation $\phi(\langle k_i, x \rangle + b_{1,i})$ is non-zero when $x$ matches key $k_i$, and the corresponding value $v_i$ is added to the output. The FFN is a sum of $d_{\text{ff}}$ such cells. *Direct expansion of the matrix product.*

**Empirical observation 8.6 (Hidden-unit specialization).** In trained transformers, individual FFN hidden units exhibit interpretable activation patterns: some fire on syntactic features, some on entity types, some on more abstract concepts. This is the basis of "neuron-level" mechanistic interpretability. Specialization is emergent, not enforced.

## 8.4 Numerical example — FFN as memory on $d_{\text{model}} = 2$

Build a tiny FFN with $d_{\text{ff}} = 3$ memory slots, $\phi = \text{ReLU}$, no biases, designed to recognize three patterns:
$$W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad W_2 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}.$$

Hidden unit 1 fires for tokens with positive coord 0; hidden unit 2 for positive coord 1; hidden unit 3 for tokens with positive sum.

For $x = (2, -1)$: hidden = $\text{ReLU}((2, -1, 1)) = (2, 0, 1)$. Output = $W_2 \cdot (2, 0, 1) = (2, 0)$ — passes coord 0 through, gates coord 1.

For $x = (-1, 2)$: hidden = $\text{ReLU}((-1, 2, 1)) = (0, 2, 1)$. Output = $(0, 2)$.

For $x = (-1, -1)$: hidden = $\text{ReLU}((-1, -1, -2)) = (0, 0, 0)$. Output = $(0, 0)$ — all memory slots silent.

This is a 3-cell key-value memory operating on 2D inputs. Real FFNs scale this to $d_{\text{ff}} = 4 \cdot d_{\text{model}}$ slots and learn the keys/values from data.

## 8.5 Mechanics check

- **Per-token, no positional mixing.** The same FFN parameters are applied to each token independently. All inter-token communication is in attention.
- **Wide hidden layer ($d_{\text{ff}} = 4 d_{\text{model}}$).** The "memory bank size." More slots = more learnable patterns; cost is the dominant FLOP term in the block.
- **Smooth activation (GELU / SiLU).** Smoother than ReLU near zero; trains slightly faster in transformers. Architecturally close to interchangeable.
- **No biases in modern LLMs.** Most production transformers have removed biases entirely (LLaMA, etc.). Rationale: small parameter saving + slight stability win + LayerNorm $\beta$ already provides the per-coordinate offset capability.
- **Gated variants are the modern default.** SwiGLU's extra gate projection improves quality at constant FLOPs (after compensating $d_{\text{ff}}$ down).
- **The FFN is where most of the parameters live.** In a standard 12-block transformer at $d_{\text{model}} = 768$, FFN holds $\approx 56\%$ of total params. Removing it is not an option.

## 8.6 Exercises

**8.1 ★★** *[Mechanism dissection — without an FFN, what fails?]* Construct a function $f: \mathbb{R}^d \to \mathbb{R}^d$ that an FFN can compute exactly but a single transformer block *without* an FFN cannot, regardless of attention configuration. (The FFN-free block is the composition of LN + attention + LN.) Use Corollary 1.3.1 as the obstruction. Make $d$ as small as possible.

**8.2 ★★** *[Failure-mode — narrow FFN.]* Empirically, $d_{\text{ff}} = 4 d_{\text{model}}$ is standard. Argue from Prop 8.5 (key-value memory view) what happens when $d_{\text{ff}}$ is much smaller — say $d_{\text{ff}} = d_{\text{model}} / 2$. What concrete capacity is lost? What kind of task would suffer most? Sketch a synthetic task on which FFN width should matter sharply.

**8.3 ★★★** *[Theorem about the numerics — gated FFN parameter accounting.]* SwiGLU has three projections $W_1, W_2, W_3$ instead of vanilla's two.

(a) Compute the parameter count of SwiGLU as a function of $d_{\text{model}}, d_{\text{ff}}$, and the FLOP count.

(b) Find the $d_{\text{ff}}^{\text{SwiGLU}}$ such that SwiGLU has the same FLOP budget as vanilla FFN at $d_{\text{ff}} = 4 d_{\text{model}}$. Express it as a fraction of $d_{\text{model}}$.

(c) Argue: at matched FLOP budget, why might SwiGLU outperform vanilla? *Hint: the gate is a learned data-dependent attention mask over the hidden units.*

(d) Bonus: argue why GeGLU (gate via GELU instead of SiLU) and SwiGLU usually perform within noise of each other. Which property of the activation is doing the work, vs. which is incidental?

**8.4 ★★★** *[Predict-then-verify — FFN width vs. capacity on a memory-like task.]* The notebook will train a tiny transformer on a "match the input" recall task with a vocabulary of $V = 64$ symbols. Sweep $d_{\text{ff}} / d_{\text{model}} \in \{0.5, 1, 2, 4, 8\}$. Predict, before running:

(a) Is there a width below which the model fails to memorize? Above? Justify from Prop 8.5.

(b) For widths above the threshold, is there diminishing return? Predict the marginal accuracy gain from $4 \times$ to $8 \times$.

(c) The task uses softmax-attention to retrieve, then FFN to map. Which sub-layer is doing the heavy lifting — attention or FFN? Predict by ablation: zero out the FFN entirely and report expected accuracy.

State confidence per part.

## 8.7 Before the notebook (`08_ffn_capacity.ipynb`)

Predictions to write down before opening.

1. **FFN as memory: hand example.** Build the $d_{\text{ff}} = 3, d_{\text{model}} = 2$ memory from §8.4 in code. Predict the output for inputs $(2, -1), (-1, 2), (-1, -1), (1, 1)$.
2. **Parameter and FLOP count.** Vanilla FFN at $d_{\text{model}} = 768, d_{\text{ff}} = 3072$: predict params and FLOPs/token. Same for SwiGLU at matched FLOPs.
3. **Capacity sweep on toy task.** Five widths, predict the accuracy curve. At what width does the model "click"?
4. **FFN ablation.** Zero out the FFN (replace with identity); predict the accuracy drop on the toy task. Order of magnitude.
5. **Activation ablation.** GELU vs. ReLU vs. SiLU at fixed width: predict whether the difference is statistically meaningful or noise-level.

For each: direction, magnitude, confidence.
