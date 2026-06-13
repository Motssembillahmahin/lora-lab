# 02 — Rank $r$ and the $\alpha$ scaling: what they control, and how to sweep them

> Sequel to [`01-lora-derivation.md`](./01-lora-derivation.md). 01 already
> derived $\Delta W = \frac{\alpha}{r}BA$ with $B\in\mathbb{R}^{d\times r}$,
> $A\in\mathbb{R}^{r\times k}$, the $B(Ax)$ forward, the $A$-random/$B$-zero init,
> gradient flow, the optimizer-memory math, and **three framings of $\alpha$**
> (§5). I will not re-derive those — I reference them and go deeper on the two
> knobs themselves. This doc is the **pre-registration** for a rank sweep: what we
> expect, and exactly how we'll read the result. The sweep has not run yet; there
> are no results here on purpose.
> Cross-refs: [`configs/qwen_0.5b_lora.yaml`](../../configs/qwen_0.5b_lora.yaml)
> (`r: 8`, `alpha: 16`, $\alpha/r=2$), the eval harness
> [ADR 0004](../decisions/0004-eval-harness.md) (`make eval` →
> held-out response perplexity, our **y-axis**), and
> [`experiments/log.md`](../../experiments/log.md) (Run 001/002, the seed study).

The experiment this doc justifies: **sweep $r \in \{2,4,8,16,32\}$ holding
$\alpha = 2r$ (so $\alpha/r = 2$ fixed), and plot held-out response perplexity
vs. $r$.** Everything below is "why that and not something else."

---

## 1. What rank actually controls — capacity, precisely

### Rank is a hard ceiling on the number of directions the update can use

From 01 §2: $\Delta W = \frac{\alpha}{r}BA$ has

$$
\operatorname{rank}(\Delta W) \le r,
$$

because every column of $BA$ is a linear combination of the $r$ columns of $B$,
so $\operatorname{col}(BA) \subseteq \operatorname{col}(B)$, a subspace of
dimension $\le r$. The scalar $\frac{\alpha}{r}$ doesn't change the rank — scaling
a matrix can't add independent directions.

So $r$ is **literally** the number of independent directions in weight space the
update is allowed to use. Not "roughly," not "on average" — a deterministic
upper bound from the factorization. Read through the forward keyhole (01 §2,
framing 1): no matter what input $x$ arrives, the adapter's contribution
$B(Ax)$ is confined to the $\le r$-dimensional column space of $B$.

### Tie to the SVD: rank-$r$ is the *best possible* rank-$r$ thing

Why is "use only $r$ directions" a principled constraint and not just a cheap
hack? Because of what the SVD says about low-rank approximation.

Any matrix $M \in \mathbb{R}^{d\times k}$ has a singular value decomposition

$$
M = U\Sigma V^\top, \qquad
\Sigma = \operatorname{diag}(\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{\min(d,k)} \ge 0),
$$

with $U\in\mathbb{R}^{d\times d}$, $V\in\mathbb{R}^{k\times k}$ orthogonal and the
$\sigma_i$ the singular values, sorted descending. Truncating to the top $r$
terms,

$$
M_r \;=\; \sum_{i=1}^{r} \sigma_i\, u_i v_i^\top
\;=\; U_{:,1:r}\,\Sigma_{1:r}\,V_{:,1:r}^\top,
$$

gives a rank-$r$ matrix. The **Eckart–Young–Mirsky theorem** states that this
truncation is the *optimal* rank-$r$ approximation of $M$ — simultaneously in
both the Frobenius and spectral norms:

$$
\min_{\operatorname{rank}(X)\le r} \|M - X\|_F = \|M - M_r\|_F
= \sqrt{\sum_{i>r}\sigma_i^2},
\qquad
\min_{\operatorname{rank}(X)\le r} \|M - X\|_2 = \|M - M_r\|_2 = \sigma_{r+1}.
$$

Read this carefully, because it is the whole justification for the rank knob:

- The best rank-$r$ approximation error is governed entirely by the
  **discarded singular values** $\sigma_{r+1}, \sigma_{r+2}, \dots$. If those are
  tiny, a rank-$r$ matrix loses almost nothing.
- So constraining the update to rank $r$ costs you exactly the "energy" in
  directions $r+1$ and beyond. If the *ideal* full-rank update $\Delta W^\star$
  has fast singular-value decay, then a rank-$r$ $BA$ can sit very close to it and
  the constraint is nearly free. If $\Delta W^\star$ has a long flat spectrum,
  rank-$r$ truncation throws away real signal.

Caveat worth stating: LoRA does **not** literally compute the SVD of some target
$\Delta W^\star$ and truncate it. Gradient descent finds *some* $B, A$ minimizing
the task loss, not the Eckart–Young optimizer of a fixed matrix. The SVD is the
**existence argument** — it tells us the *best achievable* rank-$r$ update is good
*iff* the needed update has decaying spectrum. It bounds the best case; training
may not reach it. (01 §Open-question 1 suggests the sharper probe: train, then
SVD the learned $\Delta W = \frac{\alpha}{r}BA$ and look at *its* singular-value
decay — how many directions actually carry weight after the fact.)

### Capacity grows linearly in $r$; a full update is quadratic

From 01 §2, the parameter count of one adapted matrix is

$$
\#\text{params}(BA) = \underbrace{dr}_{B} + \underbrace{rk}_{A} = r(d+k),
$$

**linear in $r$**, versus the full update's $dk$, which is independent of $r$ (it
*is* the full matrix). Every unit of $r$ buys exactly $d+k$ parameters and one
more usable direction. This is the central asymmetry: capacity (params, and the
rank ceiling) scales linearly, so doubling $r$ doubles cost — keep that in hand
for §5.

### Table: $r$ vs params vs fraction, for the square $q/o$ matrices

For Qwen2.5-0.5B's `q_proj` / `o_proj`, $d = k = 896$ (01 §1, §6 — the GQA-narrow
`k_proj`/`v_proj` are $128\times896$ and behave differently; see §2 below and
01 open-question 4). Here $r(d+k) = 1792\,r$, and the square-case closed form for
the fraction of a full update is $\frac{r(d+k)}{dk} = \frac{2r}{d} = \frac{2r}{896}$
(01 §2):

| $r$ | params $=1792r$ | fraction of full $=2r/896$ | rank ceiling $r/896$ |
|----:|----------------:|---------------------------:|---------------------:|
| 2   | 3,584           | 0.446%                     | 0.22%                |
| 4   | 7,168           | 0.893%                     | 0.45%                |
| 8   | 14,336          | 1.79%                      | 0.89%                |
| 16  | 28,672          | 3.57%                      | 1.79%                |
| 32  | 57,344          | 7.14%                      | 3.57%                |

(The $r=8$ row is the baseline from 01 §2/§6: 14,336 params/square matrix, 1.79%.)

Notice how little of full rank we ever use: even $r=32$ caps the $q/o$ update at
**3.57%** of the 896 available directions. The whole bet is that the task's
update lives in that thin slice. Note also the GQA wrinkle the table hides: the
*same* $r$ is a much looser constraint on the narrow $k/v$ ($r/128$) than on
$q/o$ ($r/896$) — uniform $r$ is not uniform capacity (01 open-question 4). The
sweep uses a single global $r$ for simplicity; per-module `rank_pattern` is a
separate future experiment.

---

## 2. The intrinsic-rank hypothesis, and what a sweep reveals

### The hypothesis

01 §2 stated the empirical claim (Hu et al., 2021): the *update* $\Delta W$ — the
change, not the pretrained $W_0$ — has low **intrinsic rank**. Restated in SVD
language from §1: the ideal update $\Delta W^\star$ has fast singular-value
decay, so the discarded tail $\sum_{i>r}\sigma_i^2$ becomes negligible once $r$
exceeds some small "intrinsic rank" $r^\star$ of the task.

The testable prediction that falls out: **eval loss should drop sharply as $r$
rises from tiny, then flatten once $r \gtrsim r^\star$.** Past the intrinsic
rank, extra directions have nothing left to capture — you pay linearly more
params/compute (§5) for vanishing return.

### Three readings of the measured perplexity-vs-$r$ curve

We plot held-out response perplexity (lower = better; ADR 0004, `make eval`) on
the y-axis against $r$ on the x-axis. There are exactly three qualitative shapes,
and each licenses a different conclusion:

**(a) Plateau by small $r$** — perplexity falls from $r=2$ then is flat from,
say, $r=4$ or $r=8$ onward.
$\Rightarrow$ the task's intrinsic rank is genuinely low; the singular tail past
$r^\star$ is negligible. Our baseline $r=8$ has **comfortable headroom** — we
could even drop to $r=4$ and lose little, saving half the adapter cost. This is
the outcome the LoRA paper's hypothesis predicts and the one we'd bet on a priori.

**(b) Still dropping at $r=32$** — the curve has not flattened by the right edge.
$\Rightarrow$ the task needs more rank than we're giving it; the spectrum of
$\Delta W^\star$ has not decayed by direction 8. Our baseline $r=8$ is
**underfitting** — the cheap rank-$r$ assumption is leaving accuracy on the table,
and we should extend the sweep ($r=64, 128$) or rethink which modules we adapt.

**(c) U-shape / rise at high $r$** — perplexity bottoms out at some middle $r$
then climbs again toward $r=32$.
$\Rightarrow$ **overfitting**. Extra capacity lets the adapter memorize the tiny
training slice; held-out perplexity (a generalization metric — the eval slice is
disjoint, ADR 0004) then *worsens* even as train loss keeps falling. The best $r$
is the bottom of the U, and bigger is strictly worse.

### Why reading (c) is genuinely plausible here — be honest

This is **not** a big run. Per the config and the seed study, we train on
$\sim$150–300 Dolly examples, 1 epoch, on CPU. With $r=32$ on the four attention
matrices we have on the order of $\sim$4M trainable params (vs. 1.08M at $r=8$,
01 §6) chasing a few hundred examples. That capacity-to-data ratio is exactly the
regime where a model memorizes. So unlike a paper trained on a large corpus —
where (a) or (b) dominate — our small-data CPU setting makes the U-shape (c) a
live possibility. Don't be surprised by it; it would be the data regime talking,
not a flaw in LoRA.

### This is open-question 1 from 01, now measurable

01 §Open-questions named this exact sweep, but at the time we had no clean metric
— Run 001/002 only had train loss (not comparable across masking, math/03 §2) and
three eyeballed generations. We now have the held-out response-NLL harness
(ADR 0004) and a seed study (Run 003) showing seed variance is tiny ($\pm0.0002$
NLL) on fixed CPU data. So a single training run per $r$ gives a trustworthy
point, and the curve is finally readable. This doc is the "why"; the run is next.

---

## 3. Why hold $\alpha = 2r$ during the sweep — and the trap if you don't

This is the crux of the experimental design. Get it wrong and the plot is
uninterpretable.

### The two confounded variables

Recall 01 §5: the effective update applied to $x$ is

$$
\Delta W\, x \;=\; \frac{\alpha}{r}\sum_{i=1}^{r} b_i\,(a_i^\top x),
$$

a sum of $r$ rank-one terms scaled by $\frac{\alpha}{r}$, where $b_i$ is column
$i$ of $B$ and $a_i^\top$ row $i$ of $A$. Changing $r$ moves **two** things at
once:

1. **Capacity** — the number of rank-one terms in the sum (the thing we *want* to
   study, §1–§2).
2. **Magnitude / effective learning rate** — the prefactor $\frac{\alpha}{r}$,
   which 01 §5 framing 1 showed behaves like a constant multiplier on the adapter
   output, i.e. an effective-LR scale on the adapter parameters.

If you hold $\alpha$ **fixed** and increase $r$, then $\frac{\alpha}{r}$
**shrinks**. So you are simultaneously (i) adding capacity *and* (ii) turning the
adapter's effective learning rate down. These are confounded. A flat
perplexity-vs-$r$ curve under fixed $\alpha$ could mean *either*:

- "rank doesn't help past here" (the conclusion you wanted to draw), **or**
- "I accidentally annealed the effective LR toward zero as $r$ grew, so the
  larger-$r$ adapters just trained less" (an artifact you didn't intend).

You cannot distinguish these from the curve. The experiment is ruined.

### The fix: hold $\alpha/r$ constant by setting $\alpha = 2r$

Setting $\alpha = 2r$ pins $\frac{\alpha}{r} = 2$ for every $r$. The
magnitude/effective-LR knob is **held fixed**, so the only thing moving across
the sweep is capacity. The curve then isolates the variable we actually care
about. This is precisely the $r$-**decoupling** property 01 §5 (framing 2) built
the $1/r$ for: "tune $\alpha$ and LR once, then sweep $r$ for capacity without
re-tuning the magnitude." The sweep *is* that property cashed in.

### The algebra, side by side

| $r$ | $\alpha/r$ with **fixed $\alpha=16$** | $\alpha/r$ with **$\alpha=2r$** |
|----:|--------------------------------------:|--------------------------------:|
| 2   | $16/2 = 8.0$                          | $4/2 = 2$                       |
| 4   | $16/4 = 4.0$                          | $8/4 = 2$                       |
| 8   | $16/8 = 2.0$                          | $16/8 = 2$                      |
| 16  | $16/16 = 1.0$                         | $32/16 = 2$                     |
| 32  | $16/32 = 0.5$                         | $64/32 = 2$                     |

Left column: a **16× swing** in effective-LR scale ($8.0 \to 0.5$) across the
sweep — that swing would dominate any capacity effect and you'd be plotting
mostly an LR sweep mislabeled as a rank sweep. Right column: dead flat at 2,
exactly the baseline value (config `alpha: 16`, $r=8$ → $\alpha/r = 2$). So the
$r=8$ point of our sweep *is* the existing baseline, which is a nice consistency
anchor.

### The rsLoRA wrinkle — a caveat on interpreting a plateau

Honesty check, from 01 §5 framing 3 (rsLoRA, Kalajdzievski 2023). Even with
$\alpha/r$ held at 2, the per-rank-one-term contribution is not perfectly
$r$-invariant. The variance argument: if the $r$ rank-one terms
$b_i(a_i^\top x)$ are roughly independent zero-mean contributions, their sum has
standard deviation $\propto \sqrt{r}$, not $\propto r$. So the **variance-correct**
normalization is $\frac{\alpha}{\sqrt r}$, and the vanilla $\frac{\alpha}{r}$
over-shrinks high-rank adapters by a factor $\sim \frac{1/r}{1/\sqrt r} =
\frac{1}{\sqrt r}$.

Consequence for *our* plot: even the $\alpha=2r$ sweep can **under-credit** large
$r$, because at $r=32$ vanilla $\alpha/r$ effectively scales the (variance-sense)
useful signal down by $\sim 1/\sqrt{32} \approx 0.18$ relative to where rsLoRA
would put it. So a plateau (reading (a)) under $\alpha/r$ is **ambiguous**: it
could be genuine low intrinsic rank, *or* it could be partly the $1/\sqrt r$
under-scaling masking a benefit that more-appropriate scaling would reveal.

I am flagging this as a **known confound to verify later, not overclaiming**. The
control experiment that disentangles it is to re-run the sweep with rsLoRA scaling
$\frac{\alpha}{\sqrt r}$ (PEFT supports `use_rslora=True`) and compare: if the
plateau lifts under $\alpha/\sqrt r$, the original plateau was partly a scaling
artifact; if it stays flat, it was genuine intrinsic rank. For our small $r$
range (and especially at the $r=8$ baseline) the $\alpha/r$ vs $\alpha/\sqrt r$
gap is mild (01 §5 framing 3), so $\alpha/r$ is a fine first pass — just don't
read a high-$r$ plateau as the final word.

---

## 4. Alpha as its own knob (hold $r$, vary $\alpha$)

The sweep above collapses the 2D $(r, \alpha)$ space onto a single diagonal
($\alpha/r = 2$). It's worth seeing the full plane so you know what we're *not*
varying and why.

At fixed $r$, $\frac{\alpha}{r}$ is an effective-LR scale on the adapter (01 §5
framing 1). The qualitative behavior of mis-setting it (01 open-question 2):

- **$\alpha$ too small** ($\alpha/r$ near 0): the adapter output is scaled toward
  zero, so $\Delta W$ barely moves the model — **underfit**, perplexity stays
  near the base line. The adapter exists but is muffled.
- **$\alpha$ too large**: the effective LR on the adapter is large, so updates
  overshoot — **unstable / oscillating / divergent** loss, or noisy generations.
- **Just right**: the sweet spot we approximate with the $\alpha = 2r$ heuristic.

So $\alpha$ (equivalently $\alpha/r$ at fixed $r$) is the **magnitude/LR axis**,
and $r$ is the **capacity axis**. They are genuinely different knobs that the
$1/r$ in the parametrization was designed to *separate* (§3, 01 §5).

### The $(r, \alpha)$ plane

```
  alpha
   ^                                    . alpha/r = 4  (hot: risk of overshoot)
64 |                          .        /
   |                    .            /        . alpha/r = 2  (our sweep diagonal)
32 |              .            .    /        /
   |        .            .        /        /          ...... alpha/r = 1
16 |  .            .   [r8,a16] /        /      ......
   |        .            .    /  .    /  ......
 8 |  .            .        /  .  ....../............... alpha/r = 0.5 (cold: underfit)
   |        .          ./.....x.........
 4 |  .          ..x..../  ......
   |  .   ......./ .....
 2 |..x....../............
   +----+----+----+----+----+----> r
        2    4    8    16   32

   x  = the alpha = 2r sweep points (this experiment): (2,4)(4,8)(8,16)(16,32)(32,64)
  []  = current baseline (r=8, alpha=16), which lies ON the alpha/r=2 diagonal
  rays from origin = lines of constant alpha/r (constant effective-LR scale)
```

Each ray through the origin is a constant-$\alpha/r$ contour — fixed
effective-LR scale, varying capacity. Our sweep walks **up the $\alpha/r=2$ ray**:
pure capacity change, magnitude held (§3). A pure $\alpha$-sweep (01
open-question 2) would instead move **vertically** at fixed $r=8$, crossing
contours — pure magnitude change, capacity held. The full picture needs both
axes; we deliberately do one clean 1D cut at a time. This doc pre-registers the
horizontal-along-the-diagonal cut.

---

## 5. The compute/memory cost of rank (the CPU-budget angle)

Everything that scales with $r$ scales **linearly**, because $r$ is an inner
dimension that appears once in every adapter quantity. Per adapted matrix
($d\times k$), reusing 01 §3 (FLOPs) and §6 (memory):

| quantity | formula | scaling |
|---|---|---|
| trainable params | $r(d+k)$ | $\propto r$ |
| Adam optimizer state (fp32 $m,v$) | $2 \times r(d+k) \times 4\text{ B}$ | $\propto r$ |
| forward FLOPs of $B(Ax)$ | $\approx r(d+k)$ mult-adds | $\propto r$ |
| backward FLOPs | $\approx$ const $\times r(d+k)$ | $\propto r$ |
| adapter file size on disk | $\propto r(d+k)$ | $\propto r$ |

(The forward-FLOP count being *equal* to the param count — one multiply-add per
param per input — is the §3 observation; it's why params and FLOPs share the same
$r(d+k)$.)

So **doubling $r$ roughly doubles all of it** — params, RAM for optimizer state,
per-step compute, and the saved adapter. Concretely, scaling 01 §6's measured
$r=8$ figures (1,081,344 trainable params total, $\approx 8.6$ MB Adam state,
4.2 MB adapter on disk):

| $r$ | total trainable params | Adam state (fp32) | adapter on disk |
|----:|-----------------------:|------------------:|----------------:|
| 2   | $\approx 270\text{k}$  | $\approx 2.2$ MB  | $\approx 1.1$ MB |
| 4   | $\approx 541\text{k}$  | $\approx 4.3$ MB  | $\approx 2.1$ MB |
| 8   | $1{,}081{,}344$ (measured) | $\approx 8.6$ MB | 4.2 MB (measured) |
| 16  | $\approx 2.16\text{M}$ | $\approx 17$ MB   | $\approx 8.4$ MB |
| 32  | $\approx 4.32\text{M}$ | $\approx 35$ MB   | $\approx 17$ MB  |

(These scale the exact $r=8$ counts linearly; the actual numbers come from
`print_trainable_parameters()` per run.) On a 14 GB CPU box even $r=32$ is small
in absolute RAM terms — the binding cost here is **wall-clock per step** (CPU
matmuls, 01 Run 001 ≈ 34 s/step at $r=8$), which also grows with $r$.

### The sweep as a cost/benefit search for the knee

Frame the whole experiment as finding the **knee of the curve**: the smallest $r$
past which perplexity stops meaningfully improving. Everything to the *right* of
the knee costs linearly more (every row above scales $\propto r$) for diminishing
— or in the overfit case (c), *negative* — return on held-out perplexity. The
goal is not "biggest $r$ wins"; it's "cheapest $r$ that's on the plateau." That's
the engineering decision the plot is meant to inform.

---

## What the plot should show / how to read it

x-axis: $r \in \{2,4,8,16,32\}$. y-axis: **held-out response perplexity** from
`make eval` (ADR 0004 — token-weighted NLL on the disjoint slice, then
$\exp(\cdot)$; lower is better). Each point: one training run at that $r$ with
$\alpha = 2r$, everything else fixed at the config baseline.

The decision rule (§2), restated:

- **Plateaus by small $r$ (a)** → intrinsic rank is low; $r=8$ has headroom, maybe
  drop to $r=4$. Pick the smallest $r$ on the flat part (the knee, §5).
- **Still dropping at $r=32$ (b)** → $r=8$ underfits; extend the sweep, or adapt
  more modules. The cheap assumption is costing accuracy.
- **U-shape, rises at high $r$ (c)** → overfitting the tiny dataset; best $r$ is
  the bottom of the U, bigger is strictly worse. Plausible here given small-data
  CPU regime (§2).

Honest caveats to attach to whatever curve we get:

1. **Single seed per $r$.** Acceptable because the seed study (Run 003) measured
   seed variance at $\pm 0.0002$ NLL on fixed CPU data — tiny relative to the
   effects we expect. But it's still one draw per point; a wobble smaller than
   $\sim 0.001$ NLL is within noise and shouldn't be over-read.
2. **Small dataset.** $\sim$150–300 examples, 1 epoch. This is *why* reading (c)
   is plausible and why absolute perplexities here aren't comparable to a larger
   run — only *within-sweep* comparisons across $r$ are valid (same data, same
   eval slice).
3. **$\alpha/r$ vs $\alpha/\sqrt r$ confound (§3).** A high-$r$ plateau under
   vanilla $\alpha/r$ may be partly the $1/\sqrt r$ under-scaling, not purely low
   intrinsic rank. The control is a second sweep with rsLoRA ($\alpha/\sqrt r$,
   `use_rslora=True`); if the plateau lifts there, the first plateau was partly a
   scaling artifact. Flag, don't conclude.
4. **Global $r$ under GQA (§1, 01 open-question 4).** The same $r$ is a looser
   constraint on narrow $k/v$ than on wide $q/o$, so the sweep conflates "more
   rank everywhere." A per-module `rank_pattern` study is the follow-up.

Result goes in `experiments/log.md` (one entry per $r$, tied to a commit SHA) and
the read-off updates the journal — **not this file**. This doc stays the
pre-registration: the prediction and the reading rule, written before the data.

> → **Result (cross-ref, not folded into the prediction above):** the sweep ran —
> see `experiments/log.md` Run 004, journal Session 5, and the figure
> [`assets/02-rank-sweep.png`](./assets/02-rank-sweep.png). Short version:
> **reading (a)** — perplexity fell monotonically ($7.70\to7.51$) and **plateaued
> by $r=16$** ($r{=}16$ and $r{=}32$ identical to 4 dp), confirming low intrinsic
> rank; $r=8$ has headroom. The §3 $\alpha/\sqrt r$ control remains to be run.
