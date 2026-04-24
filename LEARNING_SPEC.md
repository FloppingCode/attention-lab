# Learning Spec — attention-lab

This spec defines what this repo is for and what a good chapter / notebook / exercise looks like here. It is the contract against which content is written and audited. Rewrite to match when in doubt.

**Relationship to the global skill.** Generic learner profile and pedagogy (lesson structure, exercise taxonomy, experiment protocol, voice) live in the `AI-applied-math-tutor` skill. This spec **inherits** those defaults and only specifies what is particular to this repo. Read the skill first; read this spec second. If the two conflict on something repo-specific, this spec wins.

---

## 1. Purpose

Build **predictive mental models of attention and transformers** — mental models rigorous enough to support proofs, empirically grounded enough to predict what an experiment will show before running it, and complete enough to **narrate the method end-to-end** (training, validation, inference).

Concretely, by the end of a chapter the reader should be able to:

1. Give the **storyline** of the mechanism: what problem it solves, what each component does, how they connect, how behavior differs at train vs. val vs. inference.
2. State the **rigorous core** of the mechanism where one exists (definition, theorem, bound, invariance), and mark where the mechanism is heuristic rather than proved.
3. Run, perturb, and interpret a **numerical example** that makes the formula concrete.
4. Form a **falsifiable prediction** about a notebook experiment and update their mental model when results land.

Two failure modes, equally weighted, to avoid:
- **Formalism without feel** — proofs with no numerical anchor, no storyline, no "what does this look like."
- **Feel without rigor** — plots and vibes with no theorem cited, no mechanism dissected, heuristics presented as facts.

## 2. Track structure

### Chapter types

Every chapter declares one of three types in its header. The type determines which artifacts exist and how the reader engages.

- **Theory-only.** Markdown doc, no notebook. Used when content is a derivation or theorem without a meaningful experimental counterpart. Section 4 of the chapter uses a short inline code snippet (≤ 15 lines) for a numerical check rather than a standalone `.ipynb`. No "Before the notebook" section.
- **Theory + notebook.** Markdown doc + paired `.ipynb`. Default type. Full 7-section chapter structure applies.
- **Experiment-led.** Notebook is the primary artifact. The markdown doc is a short companion: motivation, the hypothesis being tested, and the mental-model target. The rigorous section, if any, is a retrospective — what the experiment revealed, formalized.

### Ordering modes

Every chapter also declares an ordering mode — how the reader is meant to engage with the two artifacts.

- **Theory-first.** Read the markdown end-to-end, do exercises, write a prediction, then open the notebook. The notebook confirms. Default for theorem-driven mechanisms (softmax scaling, $\sqrt{d_k}$, reverse-time SDE).
- **Notebook-first.** Open the notebook, state a naive hypothesis, run it, be surprised, then read the markdown. The markdown explains what you saw. Default for phenomenological chapters (attention sinks, induction-head phase transitions, loss-curve pathologies, sampler comparisons).
- **Mixed.** Read motivation + storyline in markdown, run the notebook, return to markdown for the rigorous core. Use sparingly — only when one order clearly doesn't fit.

Chapters state the declared type and mode in a small header block:

```
Type: theory+notebook
Ordering: theory-first
Depends on: 1, 2
```

### Math duplication rule

**No proofs, theorems, or derivations live in two places.** One source of truth per result — the theory chapter. Formulas, definitions, intuition refreshers, and restatements of the prediction are fine wherever they help the reader. If a notebook needs math to justify a step, that math belongs in the theory chapter and the notebook cites it ("see Theorem 3.2").

### `theory/` — chapter documents

Each chapter is a standalone markdown document with the following sections. Sections 2 and 3 may be shortened or skipped when the topic genuinely doesn't support them (say so explicitly — don't fake rigor or fake numerics).

1. **Motivation.** Two to four sentences. What object/mechanism this is, what it replaces, why it exists.
2. **Storyline / high-level overview (always present).** The informal backbone of the chapter. Narrate the mechanism end-to-end. Cover: what problem it solves, the parts and what each does, how the parts connect, **and how the mechanism behaves at training vs. validation vs. inference** when those differ. By the end of this section the reader could explain the method at a whiteboard without notes.
3. **Rigorous core (when applicable).** Precise definitions, theorems, derivations, bounds, invariances. Named theorems cited, not reproved. Proofs included only when the technique is itself a lesson. When a topic is fundamentally heuristic or empirical, *say so* and skip this section rather than manufacturing a theorem.
4. **Numerical example.** A hand-computable case (typically $d_k = d_v = 2$, $n = m = 2$ or $3$), every step shown, **or** a minimal code snippet the reader can run. Tie it explicitly to the storyline, the rigorous core, or both. For theory-only chapters this section carries the numerical work instead of a separate notebook.
5. **Mechanics check.** A short bulleted list: for each sub-component of the mechanism, one sentence stating *what that step does and why that step exists*. Catches the "glossed-over step" failure mode (softmax scaling, layer norm placement, masking, residual placement, etc.).
6. **Exercises.** See §3.
7. **Before the notebook.** A falsifiable prediction the reader writes down before running any code. Not "it will look reasonable" — a sign, a ranking, a scaling, an asymptote. Skipped for theory-only chapters; for notebook-first chapters this section instead states a *naive* hypothesis to be disturbed by the experiment.

For experiment-led chapters, the theory doc is compressed: sections 1, 2, and 7 only, and 7 names the empirical target the notebook will hit.

### `notebooks/` — experimental notebooks

**Format: real `.ipynb` files only.** Never a markdown outline, cell-listing, or `.py` stub posing as a notebook. The reader must be able to open the file in Jupyter / VS Code and execute code to understand the mechanism — that is the entire point of the artifact. If producing a notebook is too large for one operation, write the `.ipynb` in chunks; do not substitute another format.

Each notebook pairs with one theory chapter and follows the **QHMPC / F** convention (from the skill):

- Opening markdown: **Question, Hypothesis, Method, Prediction, Confidence** (low / medium / high, with a one-line reason). Prediction restated from the theory chapter; no re-derivation.
- Code cells: minimal, readable, perturbable. Every plot answers a sub-question. Every run has a reason.
- Closing markdown: **Finding** — did the prediction hold? What updates in the mental model? What's the next question this surfaces? If the prediction failed, say so and update. No rationalization.

**Length and scope norms:**

| Notebook type | Cells | Runtime | Session | Used in |
|---|---|---|---|---|
| Theory-companion | 15–30 | seconds to ~1 min | 30–60 min engaged | Phases I, II, V |
| Experiment-led | 30–80 | minutes to tens of minutes | 2–4 hours | Phases III, IV |
| Training-run (toy 2D) | 50–120 | up to ~1 hour per run | weekend-per-notebook | Phases VI, VII |

**Hard rules across all notebooks:**

- One "point" per notebook. If it's doing two things, split it.
- Longest code cell under ~40 lines. If longer, factor helpers into `src/attn_lab/`.
- Every run has a Question + Prediction *before* the cell. No speculative plotting.
- Compute fits on a laptop or a single consumer GPU. No image-diffusion-scale experiments in this repo; that belongs in a separate spike.

### `learnings.md` — rolling log

Append-only. One entry per notebook session. Format: `YYYY-MM-DD · [chapter/notebook] · 2–3 sentences on what surprised, what clicked, what was wrong, what the updated mental model is.`

## 3. Exercise contract

Inherit the skill's exercise taxonomy and difficulty calibration (★ routine / ★★ default-with-idea / ★★★ hard). Repo-specific additions and emphases:

- **Every exercise must pass the defense test:** the sentence *"this is worth doing because it teaches X about attention / about the math around attention"* must be writable in one line.
- **Prioritized exercise types for this repo**, in rough order:
  1. **Mechanism dissection.** Prove that a specific component (softmax scaling by $\sqrt{d_k}$, causal masking, residual-around-attention) does what it's claimed to do — e.g. variance control, rank preservation, gradient flow.
  2. **Failure-mode construction.** Construct inputs / configurations for which a naive version of the mechanism breaks (saturates, rank-collapses, becomes position-independent, etc.). These sharpen the mental model more than proving an upper bound does.
  3. **Predict-then-verify.** Precise prediction about the matching notebook experiment with reasoning. Confirmed or refuted by the notebook.
  4. **Theorems about the numerics.** Discretization error, precision effects, conditioning, variance of an estimator, stability of an integrator, sample complexity. These tie rigor and experiment and are *highly welcome* in this repo.
  5. **Generalization / limits.** Behavior as $d_k \to \infty$, temperature $\to 0$, sequence length $\to \infty$, key space non-Euclidean. Builds the instincts needed for the manifold / flow-matching direction in §5.
  6. **Cross-chapter combination.** Requires results from two earlier chapters. Frequency should grow as the curriculum progresses.

Plug-and-chug drills, restatement-of-definition exercises, and filler problems do not belong. A chapter with weak exercises is not ready.

**No solutions in this repo.** Exercises ship without solution files or inline answers. The reader works problems themselves. If the reader asks for a solution to a specific exercise in conversation, answer there — do not commit solutions to the repo.

## 4. Content style (repo-specific)

Inherit the skill's voice and density rules. Additional repo conventions:

- **Tag registers explicitly.** Use bold inline tags — **Claim**, **Theorem**, **Proof**, **Derivation**, **Heuristic**, **Empirical observation**, **Intuition**. The reader should never have to guess whether they're being given a theorem or a vibe.
- **LaTeX is first-class.** `$...$` inline, `$$...$$` display. Matrices, sums, expectations must render cleanly.
- **Train / val / inference differences are noted explicitly** whenever a mechanism's behavior differs across the three (e.g., dropout, layer-norm statistics, cached KV at inference, teacher forcing). This is a repeat failure mode in ML pedagogy and the repo makes it a first-class structural element.
- **No pep talk, no throat-clearing, no "now we're going to..."**. State the object. Do the work.

## 5. Curriculum direction

The current chapters (0–8) cover the mechanics of attention, softmax/scaling, positional encoding, and the entry points to mechanistic interpretability. This is the **foundations layer**.

Longer-term, the repo should extend toward the reader's research interests:
- Attention and transformers as a computational substrate for **flow matching and diffusion models**.
- **Geometric and manifold-valued attention** mechanisms; attention on non-Euclidean key / value spaces.
- Interactions between softmax geometry, token geometry, and the loss landscape.
- Training-dynamics instincts: what a loss curve tells you, what ablations isolate what mechanism, what distinguishes a real effect from a seed artifact.

New chapters are added when they either (a) close a gap in the foundations layer, (b) build a bridge toward one of the directions above, or (c) cover a specific mechanism the reader is trying to form a mental model of. Chapters added "because it's a common ML topic" do not belong here.

## 6. How AI collaborators should work in this repo

An AI collaborator (Claude, etc.) writing or revising content here must:

1. Load the `AI-applied-math-tutor` skill first for learner profile and pedagogy defaults.
2. Read this spec for repo-specific structure and scope.
3. **For a new chapter:** draft the **storyline** and the **exercises** first. If either is weak, the chapter is not ready. Fill in rigorous core and numerical sections only after the storyline holds up.
4. **For a new notebook:** draft the **Question, Prediction, and Confidence** first. If the prediction is vague, the notebook is not ready.
5. **When asked to "explain" or "teach" a concept in conversation:** follow the skill's three-layer structure (intuition → rigorous-when-applicable → numerical), then an exercise, then an experiment hook. Do not skip the storyline. Do not force a theorem onto a heuristic.
6. **Never reveal solutions unprompted.** Hints in layers when asked.
7. **Match the density of a research monograph written for a mathematical peer.** If a draft reads like a Medium post or an undergrad lecture, rewrite.

## 7. Revision policy

This spec evolves. When goals shift (especially §1 and §5), update explicitly and date the change in `learnings.md`. When a chapter is rewritten to match a revised spec, log it in `learnings.md` with a pointer.
