# Chapter 2 — Softmax and temperature

```
Type: theory + notebook
Ordering: theory-first
Depends on: 1 (scaled dot-product attention)
```

## 2.1 Motivation

Attention scores $s_{ij}$ are arbitrary reals. To turn them into usable attention weights we need a map $\mathbb{R}^n \to \Delta^{n-1}$ (the probability simplex) that is non-negative, sum-to-one, differentiable, monotone in each coordinate, and — ideally — the *exponential-family* canonical link. Softmax satisfies all of these and is the unique such map up to scaling. Its shape parameter, *temperature*, controls how selectively attention concentrates. Understanding softmax deeply is understanding attention's information-retrieval dynamics.

## 2.2 Storyline

Softmax does one job with four useful consequences.

**The job.** Given logits $s \in \mathbb{R}^n$, softmax emits the Gibbs distribution $p_i = e^{s_i/T} / Z$ on $\{1, \dots, n\}$, with temperature $T > 0$ controlling concentration.

**Useful consequence 1: shift invariance.** Adding a constant to every logit leaves the distribution unchanged. This is why the standard numerical implementation subtracts $\max_i s_i$ before exponentiating — it prevents overflow without affecting the answer.

**Useful consequence 2: smoothness.** Softmax is $C^\infty$, with a clean Jacobian $\partial p_i / \partial s_j = p_i(\delta_{ij} - p_j) / T$. This enables gradient flow to the upstream $Q, K$ projections. Its failure at saturation (Lemma 3.3 preview) is the central reason scaling matters.

**Useful consequence 3: temperature controls entropy.** As $T \to 0^+$, softmax approaches a one-hot at $\arg\max s$; as $T \to \infty$, it approaches uniform. The entropy of $p$ is a smooth monotone function of $T$ for fixed $s$. This is the mechanism's information-control knob.

**Useful consequence 4: log-sum-exp as a soft max.** $\text{LSE}(s) = \log \sum_i e^{s_i}$ is a smooth upper bound on $\max_i s_i$, and $\nabla \text{LSE} = \text{softmax}$. This is why "softmax" is called that — it is the gradient of a smooth-max function.

**Temperature in attention.** In scaled dot-product attention the $1/\sqrt{d_k}$ factor is exactly an effective temperature: $\text{softmax}(QK^\top / \sqrt{d_k}) = \text{softmax}_{\sqrt{d_k}}(QK^\top)$. Chapter 3 derives why this particular temperature is the right one; Chapter 2 is the tool-building that makes Chapter 3 possible.

**Regime notes.** Softmax itself is stateless — no train / val / inference asymmetry in the operator. Temperature, however, can differ across regimes:
- Training uses whatever temperature is built into the architecture (typically $\sqrt{d_k}$ for attention, $1$ for the output layer).
- **Inference-time sampling** from a language model often uses a *different* temperature (sampling temperature) on the output softmax — concentrating or broadening the next-token distribution independently of how the model was trained. This is a *post-training* knob.
- Masking (causal or attention masks) is applied additively to the *pre-softmax* scores, usually by adding $-\infty$ to disallowed positions, producing zero weight there after softmax.

## 2.3 Rigorous core

**Definition 2.1 (Softmax with temperature).** For $s \in \mathbb{R}^n$ and $T > 0$,
$$\text{softmax}_T(s)_i = \frac{e^{s_i / T}}{\sum_{j=1}^n e^{s_j / T}}.$$
We write $\text{softmax}(s) := \text{softmax}_1(s)$ for the unit-temperature default.

**Proposition 2.2 (Shift invariance).** For any $c \in \mathbb{R}$, $\text{softmax}_T(s + c \mathbf{1}) = \text{softmax}_T(s)$. (The constant $e^{c/T}$ factors out of numerator and denominator.)

**Proposition 2.3 (Jacobian).** Let $p = \text{softmax}(s)$. Then
$$\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j).$$
For $\text{softmax}_T$, the Jacobian is $\tfrac{1}{T} p_i(\delta_{ij} - p_j)$. *Quotient rule, case-split on $i = j$.*

**Corollary 2.3.1 (Mass conservation).** The rows and columns of the Jacobian each sum to zero: $\sum_i \partial p_i / \partial s_j = 0$. *Why:* softmax maps to the simplex, so $\sum_i p_i = 1$ is constant; any perturbation in $s_j$ preserves this.

**Proposition 2.4 (Entropy bounds).** For $p = \text{softmax}(s)$, $H(p) = -\sum_i p_i \log p_i \in [0, \log n]$. The upper bound is attained iff all $s_i$ are equal; the lower bound is approached as the pre-softmax margin grows without bound. *Upper: Jensen / Lagrange. Lower: entropy non-negative on any distribution.*

**Proposition 2.5 (Saturation margin).** For $s = (0, \Delta, 0, \dots, 0) \in \mathbb{R}^n$ (winner at index 2 with margin $\Delta$), the winning probability is
$$p_2 = \frac{e^\Delta}{e^\Delta + (n-1)} = \frac{1}{1 + (n-1) e^{-\Delta}}.$$
To achieve $p_2 > 1 - \epsilon$ requires $\Delta > \log((n-1)/\epsilon)$ to leading order.

**Corollary 2.5.1.** Margin requirements are *logarithmic* in $n$ — doubling the number of keys costs one additional nat of margin. This is why attention over long contexts is not hopeless despite the $n$-way competition; the cost to distinguish a clear winner from $n$ distractors grows only as $\log n$.

**Proposition 2.6 (Log-sum-exp).** Let $\text{LSE}(s) = \log \sum_i e^{s_i}$. Then:

(a) $\nabla \text{LSE}(s) = \text{softmax}(s)$;
(b) $\max_i s_i \le \text{LSE}(s) \le \max_i s_i + \log n$;
(c) $\text{LSE}$ is convex.

*Sketches: (a) direct $\partial_{s_i} \text{LSE} = p_i$. (b) sandwich $e^m \le \sum_i e^{s_i} \le n e^m$. (c) Hessian is the softmax Jacobian, which is PSD as the covariance of the categorical $p$ — Exercise 2.1.*

**Proposition 2.7 (Temperature as inverse logit scaling).** $\text{softmax}_T(s) = \text{softmax}(s/T)$. Therefore the $1/\sqrt{d_k}$ factor in attention corresponds to effective temperature $T_{\text{eff}} = \sqrt{d_k}$.

## 2.4 Numerical example — shift invariance and saturation by hand

**Shift invariance.** $\text{softmax}((1, 2, 3))$ vs. $\text{softmax}((101, 102, 103))$: identical. The second form overflows a naive implementation; subtracting $103$ first recovers the first form. Hand-computed:
$$\text{softmax}((1, 2, 3)) = \frac{1}{e + e^2 + e^3} (e, e^2, e^3) \approx (0.090, 0.245, 0.665).$$

**Saturation by margin.** For $n = 10$ and $\Delta \in \{1, 3, 5, 10\}$ the winning probability $p_{\text{win}} = e^\Delta / (e^\Delta + 9)$ is:

| $\Delta$ | $p_{\text{win}}$ |
|---|---|
| 1 | $\approx 0.252$ |
| 3 | $\approx 0.691$ |
| 5 | $\approx 0.943$ |
| 10 | $\approx 0.9996$ |

Margin $\Delta \approx \log(n / \epsilon)$ is where $p_{\text{win}}$ crosses $1 - \epsilon$: for $n = 10$, $\epsilon = 0.01$, that's $\Delta \approx 6.9$ — the table above brackets it.

**LSE vs. max.** For the same vector $(0, 5, 0, 0, \dots, 0)$ with $n = 10$, $\max = 5$, $\text{LSE} = 5 + \log(1 + 9e^{-5}) \approx 5.061$. Tight at large margin, off by up to $\log n \approx 2.3$ at zero margin.

## 2.5 Mechanics check

- **Exponential** — converts any real vector to positive entries, enforcing non-negativity.
- **Normalizer $Z = \sum_j e^{s_j}$** — enforces sum-to-one, turning a positive-valued map into a probability distribution.
- **Shift-invariance** — makes the implementation numerically stable (subtract the max), and means softmax only cares about score *differences*, not absolute scales.
- **Temperature $T$ (or equivalently, pre-softmax scaling)** — the *only* knob controlling concentration. Every selectivity argument in attention is ultimately a temperature argument.
- **Not hard-max** — softmax's smoothness is what makes attention trainable (Prop 1.4 + Lemma 3.3). Argmax would give identical forward-pass semantics in the $T \to 0$ limit but zero gradient.
- **Asymptotics at $T \to 0$ and $T \to \infty$** — one-hot and uniform respectively. The entire useful range of attention lives between these two extremes.

## 2.6 Exercises

**2.1 ★★** *[Mechanism dissection — variance of the categorical.]* Let $p = \text{softmax}(s)$ and let $X$ be the categorical random variable with $P(X = i) = p_i$. For any function $f: \{1, \dots, n\} \to \mathbb{R}$, compute $\text{Var}_p(f(X))$ in terms of $p$ and $f$. Show that the softmax Jacobian (Prop 2.3) is precisely the covariance matrix of the indicator vector $\mathbf{1}_X$ of this categorical. What does this say about the Jacobian's eigenvalues?

**2.2 ★★** *[Failure-mode construction — temperature pathologies.]* Construct pre-softmax inputs $s^{(1)}, s^{(2)} \in \mathbb{R}^n$ such that:

(a) at $T = 1$, $\text{softmax}(s^{(1)})$ and $\text{softmax}(s^{(2)})$ have the same top-1 probability, but at $T = 10$ they differ in top-1 by more than $0.1$;

(b) the entropy of $\text{softmax}_T(s)$ is non-monotone in $T$ over some range.

For (b), state whether non-monotonicity is even possible. Prove or disprove.

**2.3 ★★★** *[Theorem about the optimization — LSE duality.]* Consider the variational characterization
$$\text{LSE}(s) = \max_{p \in \Delta^{n-1}} \left\{ \langle p, s \rangle + H(p) \right\},$$
where $\Delta^{n-1}$ is the probability simplex and $H(p) = -\sum_i p_i \log p_i$.

(a) Prove the identity: show the maximizer is $p = \text{softmax}(s)$ and the maximum value is $\text{LSE}(s)$.

(b) Generalize to temperature: express $T \cdot \text{LSE}(s/T)$ as a similar variational problem with the entropy term scaled appropriately. What does temperature correspond to in this dual picture?

(c) Interpret the duality in attention: what is the "entropy-regularized" optimization that softmax attention is implicitly solving at each query position?

**2.4 ★★★** *[Failure mode — hard attention and gradient estimators.]* A hard-attention layer selects one key per query via argmax instead of softmax. Gradients vanish almost everywhere (Prop 1.4 failure).

(a) Sketch the Gumbel-softmax estimator: add Gumbel noise to the scores, softmax at temperature $T$, anneal $T$ toward zero during training. Why does this preserve gradients while approaching hard selection?

(b) Sketch the straight-through estimator: forward pass is argmax; backward pass pretends the argmax was the identity map (or, more commonly, pretends it was softmax). Why is this *biased* in the gradient sense, and what is the bias? Name at least one failure mode in terms of training dynamics.

(c) Compare: under what circumstances would you choose Gumbel-softmax over straight-through, and vice versa? What role does the scale of $n$ (number of keys) play?

**2.5 ★★★** *[Cross-chapter combination — predict-then-verify.]* The notebook will compute softmax entropy of $\text{softmax}(\mathbf{s})$ for $\mathbf{s} \sim \mathcal{N}(0, \sigma^2 I_n)$ as a function of $(\sigma, n)$. Before running, predict:

(a) for fixed $n$, entropy as a function of $\sigma$ — direction, asymptotes at $\sigma \to 0$ and $\sigma \to \infty$, inflection behavior;

(b) for fixed $\sigma$, entropy as a function of $n$ — direction, scaling with $n$;

(c) is there a combined $(\sigma, n)$-quantity that controls entropy? Conjecture one, defend it, and state what you expect to see in a heatmap of entropy over $(\sigma, n)$. *Hint: the score variance of a single element is $\sigma^2$; relate this to Prop 2.5.*

State confidence per part with a one-line reason.

## 2.7 Before the notebook (`02_softmax_temperature.ipynb`)

Write down the following predictions *before* opening the notebook.

1. **Shift invariance check.** Direction and expected numerical error of the difference $\text{softmax}((1, 2, 3)) - \text{softmax}((101, 102, 103))$ under the safe (max-subtracting) and naive (no subtraction) implementations.
2. **Entropy vs. temperature.** Fix $s \sim \mathcal{N}(0, I_{50})$. Plot of $H(\text{softmax}_T(s))$ as $T$ ranges over $[0.01, 100]$ on a log-x axis. State the two asymptotes and the direction of the curve between them.
3. **Saturation margin.** For $n = 10$ and target $p_{\text{win}} = 0.99$, predict $\Delta^\star$ and the closed-form dependence on $n$ (from Corollary 2.5.1).
4. **Entropy heatmap over $(\sigma, n)$.** From Exercise 2.5(c): what combined quantity do you conjecture controls entropy? Predict the level-set shape of the entropy heatmap over $\log \sigma \times \log n$ — are level sets vertical, horizontal, or diagonal lines?
5. **Jacobian PSD check.** Softmax Jacobian is a covariance matrix (Exercise 2.1). Predict: is it PSD or strictly PD? What is its rank?
6. **LSE sandwich.** For $s \sim \mathcal{N}(0, I_n)$, predict the gap $\text{LSE}(s) - \max_i s_i$ as a function of $n$. Does it grow like $\log n$ (the worst-case bound) or something smaller?

For each: direction, magnitude, confidence. The notebook will confirm or update each prediction.
