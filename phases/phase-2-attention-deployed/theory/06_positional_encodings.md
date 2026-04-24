# Chapter 6 — Positional encodings

```
Type: theory + notebook
Ordering: theory-first
Depends on: 1 (scaled dot-product attention)
```

## 6.1 Motivation

Self-attention is permutation-equivariant over tokens (Corollary 1.2.1): if we shuffle the input rows of $X$, the output rows shuffle the same way. The operator has *no intrinsic notion of position*. For a sequence model this is a defect — "the cat sat on the mat" must produce different outputs from "on the mat the cat sat" — so positional information must be injected into the representation. How you inject it is a consequential architectural decision that determines, among other things, whether the model generalizes to sequence lengths longer than those seen in training.

## 6.2 Storyline

Three main families, each answering "how is position encoded?" differently:

1. **Absolute additive (APE, sinusoidal or learned).** Add a position-dependent vector $p_i$ to the token embedding $x_i$ *before* the first attention layer. The rest of the network sees a position-tagged representation. Sinusoidal encodings use fixed $p_i = (\sin(i/10000^{2k/d}), \cos(i/10000^{2k/d}))_{k}$; learned encodings treat $p_i$ as trainable parameters. Simple; breaks for lengths outside the training range.

2. **Relative positional encoding (RPE).** Modify the attention computation so that the score between query $i$ and key $j$ depends on $i - j$ (the *relative* offset) rather than on $i$ and $j$ independently. Typically implemented by adding a learned bias $b_{i-j}$ to the pre-softmax scores. More inductive-bias-appropriate for translation-invariant structure.

3. **Rotary positional embedding (RoPE).** Rotate $q_i$ and $k_j$ by a position-dependent rotation $R_i, R_j$ before computing their inner product. The key identity: $\langle R_i q, R_j k \rangle = \langle q, R_{j-i} k \rangle$ — relative position falls out of the *geometry* of the inner product, not from an additive bias. This is currently the dominant choice for large LMs.

**What "extrapolation" means.** A positional encoding *extrapolates* if the model maintains performance at test-time sequence lengths beyond the maximum seen at training. Learned absolute embeddings do not extrapolate (the positions beyond training are uninitialized). Sinusoidal absolute encodings technically extend but still degrade — the model has not seen the specific high-frequency combinations. RoPE and relative biases extrapolate structurally because they depend only on *offsets*, not absolute positions; length generalization becomes a matter of *how sharply* the encoding decays at large offsets.

**The geometric intuition for RoPE.** In each 2D subspace, $q$ and $k$ are rotated by an angle proportional to position. The inner product of a rotated $q_i$ and rotated $k_j$ depends only on the difference $j - i$ via $\cos(\theta_k (j - i))$. Different subspaces rotate at different frequencies $\theta_k$ — low frequencies encode long-range position (monotone over the whole sequence), high frequencies encode short-range position. This is Fourier analysis applied to sequence position; the different subspaces form a "position-frequency decomposition" analogous to spatial-frequency tiling in Fourier images.

**Regime notes.**

- At training, the positional encoding sees positions $\{1, \dots, n_{\text{train}}\}$.
- At inference, if the sequence exceeds $n_{\text{train}}$, behavior depends on which family is used. RoPE can "extrapolate" but with known failure modes (attention-to-long-range degrades). Length-extension techniques — YaRN, Positional Interpolation, NTK scaling — are post-training fixes that rescale the RoPE frequencies to preserve attention patterns at longer sequences.
- For masked-LM training (encoder-only, e.g. BERT), absolute learned encodings are common; for autoregressive decoders (GPT family), RoPE has largely won.

## 6.3 Rigorous core

**Definition 6.1 (Sinusoidal absolute encoding).** For position $i \in \mathbb{Z}_{\ge 0}$ and dimension index $k \in \{0, \dots, d/2 - 1\}$,
$$p_{i, 2k} = \sin(i \cdot \omega_k), \qquad p_{i, 2k + 1} = \cos(i \cdot \omega_k), \qquad \omega_k = 10000^{-2k/d}.$$
The input to the first layer is $x_i + p_i$.

**Proposition 6.2 (Absolute encoding linearization).** For any two positions $i, j$ and any fixed frequency $\omega$, the 2D rotation matrix
$$R_\omega(i - j) = \begin{pmatrix} \cos(\omega(i-j)) & -\sin(\omega(i-j)) \\ \sin(\omega(i-j)) & \cos(\omega(i-j)) \end{pmatrix}$$
satisfies $(p_j^{(\omega)})^\top \cdot p_i^{(\omega)} = \cos(\omega(i - j))$, where $p_i^{(\omega)} = (\sin(\omega i), \cos(\omega i))$. So the inner product between sinusoidal encodings depends only on $i - j$.

*Proof.* $\sin(\omega i) \sin(\omega j) + \cos(\omega i) \cos(\omega j) = \cos(\omega(i - j))$ by the angle-difference formula.

**Corollary 6.2.1.** In the absence of token content, attention score $s_{ij} = \langle p_i, p_j \rangle$ under sinusoidal encoding depends only on the offset $i - j$. This is the theoretical basis for claiming sinusoidal APE is "almost relative" — but only in the purely-positional limit; mixing in token content breaks the pure offset dependence.

**Definition 6.3 (Rotary positional embedding, 2D block).** For position $i$ and frequency $\theta$, let
$$R_i^\theta = \begin{pmatrix} \cos(i \theta) & -\sin(i \theta) \\ \sin(i \theta) & \cos(i \theta) \end{pmatrix} \in SO(2).$$
Given a $d_k$-dimensional query or key, group coordinates into $d_k / 2$ pairs and rotate each pair by $R_i^{\theta_k}$ where $\theta_k = 10000^{-2k/d_k}$ for $k = 0, \dots, d_k/2 - 1$.

**Proposition 6.4 (RoPE gives exact relative scores).** Let $\tilde q_i, \tilde k_j$ denote the RoPE-rotated query and key. Then
$$\langle \tilde q_i, \tilde k_j \rangle = \langle q_i, R_{j - i}^{\cdot}\, k_j \rangle,$$
where $R_{j-i}^{\cdot}$ denotes the block-diagonal rotation with the same frequency structure. In particular, the pre-softmax score depends on $q_i, k_j$, and $j - i$ — never on $i$ or $j$ alone.

**Proof.** In each 2D block, $(R_i^\theta q)^\top (R_j^\theta k) = q^\top (R_i^\theta)^\top R_j^\theta k = q^\top R_{j - i}^\theta k$, where the last step uses $(R_i^\theta)^\top = R_{-i}^\theta$ and $R_a^\theta R_b^\theta = R_{a+b}^\theta$. Summing over blocks gives the full inner product, which depends only on the relative offset $j - i$ per block. $\blacksquare$

**Corollary 6.4.1 (Translation invariance of the score).** Translating the entire sequence in position by a constant ($i \to i + c$ and $j \to j + c$) does not change any pre-softmax score under RoPE. Absolute position is not represented; only relative offsets are.

**Empirical observation 6.5 (Extrapolation and frequency decay).** At training length $n_{\text{train}}$, the highest-frequency RoPE component has completed many rotations; the lowest-frequency component has rotated through only a small angle. At test length $n_{\text{test}} \gg n_{\text{train}}$, low-frequency components venture into unseen angles and high-frequency components alias. Empirically, attention-to-long-range degrades sharply past $n_{\text{train}}$ without post-training interventions (YaRN, PI, NTK scaling) that rescale the $\theta_k$'s.

## 6.4 Numerical example — RoPE rotations at $d_k = 4, n = 3$ by hand

$d_k = 4$ means 2 frequency bands: $\theta_0 = 1$, $\theta_1 = 10000^{-2/4} = 0.01$.

Position $i = 0$: $R_0 = I$ in both blocks. Position $i = 1$: rotate coords $(0, 1)$ by $1 \cdot \theta_0 = 1$ rad $\approx 57°$; rotate coords $(2, 3)$ by $1 \cdot \theta_1 = 0.01$ rad $\approx 0.57°$. Position $i = 2$: doubled angles.

For query $q = (1, 0, 1, 0)^\top$ at position $i = 1$, the rotated query is:
$$\tilde q = (\cos 1 \cdot 1 - \sin 1 \cdot 0, \sin 1 \cdot 1 + \cos 1 \cdot 0, \cos 0.01 \cdot 1 - \sin 0.01 \cdot 0, \sin 0.01 \cdot 1 + \cos 0.01 \cdot 0) \approx (0.540, 0.841, 0.99995, 0.00999).$$

The first block sees a large rotation (high-frequency) — position is strongly encoded in those two coords. The second block sees almost no rotation (low-frequency) — those two coords barely change with position for short sequences. This is the frequency-decomposition mental model: high-frequency bands carry local position, low-frequency bands carry global position.

## 6.5 Mechanics check

- **Sinusoidal APE is added *once*, before the first layer.** It composes with every later operation as a fixed input perturbation. Once deep in the network, position information is carried by whatever features the network has learned to represent position with.
- **RoPE is applied per layer, per head, to $Q$ and $K$ only — never to $V$.** $V$ carries content, not position; rotating $V$ would corrupt the content pipeline.
- **RoPE frequencies.** $\theta_k = \text{base}^{-2k/d_k}$ with base $= 10000$ is the original choice. Base is tunable; larger base means slower rotation at low frequencies — longer wavelength, better for long-context generalization. "NTK-scaled" and "YaRN" methods retune the base post-training.
- **Relative bias (e.g. T5, ALiBi) is additive to scores.** It does not modify $Q, K$. This is cheaper than RoPE (no multiplications) and often extrapolates better due to a simpler form.
- **Why $V$ is not rotated in RoPE.** Content mixing after attention is a linear combination of $V$ rows with weights from softmax. Rotating $V$ would encode position into the output in a way that the subsequent layers would have to un-rotate — wasteful and disruptive.

## 6.6 Exercises

**6.1 ★★** *[Mechanism dissection — sinusoidal APE as fixed linear subspace.]* Sinusoidal APE adds $p_i$ to $x_i$. Under a learned $W_Q = W_K$, the attention score decomposes as
$$s_{ij} = (x_i + p_i)^\top W_Q^\top W_K (x_j + p_j) = \underbrace{x_i^\top M x_j}_{\text{content}} + \underbrace{x_i^\top M p_j}_{\text{content-position}} + \underbrace{p_i^\top M x_j}_{\text{position-content}} + \underbrace{p_i^\top M p_j}_{\text{pure position}}.$$

(a) Identify each of the four terms in words.

(b) Which term depends only on $i - j$? (The pure-position term — by Corollary 6.2.1.)

(c) Argue why the cross terms "content-position" are *both* desirable (they allow content to interact with position) and *problematic* (they couple tokens' content to their absolute positions, which is why absolute APE doesn't extrapolate).

**6.2 ★★** *[Failure-mode — learned APE at unseen length.]* A model is trained with learned absolute positional embeddings up to length $n_{\text{train}} = 512$. At inference with length 1024, what happens? Construct the specific failure: which positions have undefined behavior, and why doesn't the model "just learn" to extrapolate? Cite what part of the architecture has zero training signal at positions $> 512$.

**6.3 ★★★** *[Theorem about the frequencies — when does RoPE alias?]* RoPE with $d_k$ dimensions has frequencies $\theta_0, \dots, \theta_{d_k/2 - 1}$ with $\theta_k = 10000^{-2k/d_k}$.

(a) Derive the period of each frequency: $T_k = 2\pi / \theta_k$.

(b) Identify the *shortest-period* component (highest frequency). At what sequence length does it complete a full rotation? What does this say about short-range position resolution?

(c) Identify the *longest-period* component (lowest frequency). At training length $n_{\text{train}} = 2048$, what fraction of its full rotation has elapsed? What does this say about whether the model can distinguish position 1 from position 2000 using that component?

(d) Aliasing: for sequence lengths beyond training, which frequency bands get corrupted first? Relate this to the empirical fact that long-context degradation affects attention-to-long-range first.

**6.4 ★★★** *[Predict-then-verify — positional encoding bakeoff.]* The notebook will implement three positional schemes (sinusoidal APE, learned APE, RoPE) on a simple copy task with train length 64, test length 256. Predict:

(a) Which scheme performs best at train length? At test length? Why?

(b) For RoPE, does changing the base from 10000 to 100000 help at the longer test length? Justify from Exercise 6.3.

(c) For learned APE, what does the test-time accuracy look like as a function of test length? Graceful degradation, step function, or noise?

State confidence per part with a one-line reason.

## 6.7 Before the notebook (`06_positional_bakeoff.ipynb`)

Write these predictions before opening the notebook.

1. **Sinusoidal encoding inner products.** Plot of $\langle p_i, p_j \rangle$ as a function of $j - i$ for fixed $i$. Predict the shape — smooth decay? oscillatory? asymptotic value as $|i - j| \to \infty$?
2. **RoPE rotation angles.** For $d_k = 64$, what are the three smallest-period frequencies? How many sequence steps does each need to complete a rotation?
3. **Training-vs-test bakeoff.** For three schemes (sinusoidal APE, learned APE, RoPE) on a length-64 → length-256 extrapolation: predicted rank ordering at train length, at test length.
4. **RoPE base ablation.** $\text{base} \in \{1000, 10000, 100000\}$: predicted effect on test-length extrapolation.
5. **Attention-to-distance plots.** For a trained model, the expected attention mass at offset $d$ as a function of $d$. Should decay — at what rate?

For each: direction, order-of-magnitude, confidence.
