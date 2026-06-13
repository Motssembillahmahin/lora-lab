# 01 — LoRA from scratch: the derivation

> Canonical reference for shapes and parameter counts in this repo. If a number
> here disagrees with code, the code wins — but tell me, because one of us has a
> bug. Cross-refs: [`configs/qwen_0.5b_lora.yaml`](../../configs/qwen_0.5b_lora.yaml),
> [`src/train.py`](../../src/train.py), [`src/merge.py`](../../src/merge.py),
> and the sequel [`02-rank-and-alpha.md`](./02-rank-and-alpha.md) (rank/alpha
> tuning, written next).

This document derives LoRA (Low-Rank Adaptation) from the ground up. The goal is
that by the end you could re-derive it on a whiteboard without notes. I'll show
every shape, and where there's a real design fork I'll show both branches and
the tradeoff rather than handing you one blessed answer.

A note on conventions before we start. Deep-learning libraries store a linear
layer's weight as $W \in \mathbb{R}^{d \times k}$ and compute $h = Wx$ for a
column vector $x \in \mathbb{R}^k$, producing $h \in \mathbb{R}^d$. PyTorch's
`nn.Linear(in_features=k, out_features=d)` literally holds a tensor of shape
`(d, k)` and applies `x @ W.T` on row-batched inputs. I'll use the math
convention ($Wx$, column vectors) throughout and call out the transpose only
where it bites you. $d$ = output dim, $k$ = input dim. Keep that fixed in your
head; every shape below hangs off it.

---

## 1. The setup, and why full fine-tuning hurts

A single linear layer holds a pretrained weight matrix

$$
W_0 \in \mathbb{R}^{d \times k}, \qquad h = W_0 x, \quad x \in \mathbb{R}^{k}, \; h \in \mathbb{R}^{d}.
$$

Full fine-tuning says: keep the architecture, but learn an additive correction
$\Delta W$ of the **same shape** and use $W_0 + \Delta W$ instead. Equivalently,
you let the optimizer move every entry of $W_0$ directly — the "correction" view
and the "move the weights" view are identical, since $W = W_0 + \Delta W$.

The cost is the shape itself:

$$
\Delta W \in \mathbb{R}^{d \times k} \;\Rightarrow\; \#\text{params} = d \cdot k.
$$

For one Qwen2.5-0.5B attention projection with $d = k = 896$ (hidden size; see
the assumption note in §2):

$$
d \cdot k = 896^2 = 802{,}816 \approx 8.03 \times 10^5 \text{ params per matrix.}
$$

That's *one* matrix. The attention block of a decoder layer has four projections
we adapt — `q_proj, k_proj, v_proj, o_proj` (see `target_modules` in the config).
Qwen2.5-0.5B has 24 transformer layers. So the attention-projection weight budget
alone is roughly

$$
24 \text{ layers} \times 4 \text{ proj} \times 8.03\times10^5 \approx 7.7 \times 10^7 \text{ params,}
$$

and that's before the MLP blocks, which are *larger* per layer than attention.
Full fine-tuning means: (a) a gradient for every one of those params, (b) Adam
optimizer state for every one (§6), and (c) a full-size checkpoint per saved
model. On a 14 GB CPU box that is a non-starter. The question LoRA answers is:
**can we get most of the adaptation benefit while only training a tiny slice?**

> Honesty check (now verified empirically): $q/k/v/o$ are not all $896\times896$
> in Qwen2.5-0.5B. It uses grouped-query attention (GQA): 14 query heads but only
> 2 KV heads at `head_dim=64`. So `q_proj` and `o_proj` are full hidden-size
> ($896\times896$), but `k_proj`/`v_proj` project to a *smaller* KV dimension of
> $2\times64 = 128$. The $7.7\times10^7$ figure above is an upper-bound-flavored
> approximation that pretends all four are square; the LoRA math below is
> identical regardless of the exact $d, k$ per matrix — only the per-matrix count
> changes. §6 carries the **measured** trainable-parameter count, which matches a
> hand GQA decomposition to the exact parameter.

---

## 2. The hypothesis: $\Delta W$ is low-rank

### Rank, precisely

The **rank** of a matrix $M \in \mathbb{R}^{d\times k}$ is the dimension of its
column space — the number of linearly independent columns (equivalently rows;
they're always equal). Three equivalent lenses:

- **Span:** $\operatorname{rank}(M) = \dim(\operatorname{col} M)$. Every output
  $Mx$ lives in an $r$-dimensional subspace of $\mathbb{R}^d$, no matter what $x$
  you feed in.
- **Factorization:** $\operatorname{rank}(M) = r$ iff $M = BA$ with
  $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$, and $r$ is the
  smallest such inner dimension. This is the lens LoRA exploits directly.
- **Singular values:** $\operatorname{rank}(M)$ = number of nonzero singular
  values in the SVD $M = U\Sigma V^\top$. The "soft" version of this — most
  singular values being *small* though nonzero — is the real claim (see below).

Maximum possible rank is $\min(d, k)$. For our square case, $\min(896,896)=896$.

### The claim

LoRA's empirical hypothesis (Hu et al., 2021): the *update* $\Delta W$ learned
during adaptation has low **intrinsic rank**. Note carefully — this is a claim
about $\Delta W$, the *change*, not about $W_0$. The pretrained matrix is
full-rank and information-dense; we're not touching it. The bet is that *steering*
an already-capable model toward a new task lives in a thin subspace: a few
directions in weight space do most of the work.

So we *constrain* the update to be exactly low-rank by construction:

$$
\Delta W = B A, \qquad
B \in \mathbb{R}^{d \times r}, \quad
A \in \mathbb{R}^{r \times k}, \quad
r \ll \min(d, k).
$$

Shape sanity: $(d\times r)(r\times k) = d\times k$. The inner $r$ contracts, the
outer dims survive — $BA$ has the right shape to stand in for $\Delta W$. And by
construction $\operatorname{rank}(BA) \le r$, because every column of $BA$ is a
linear combination of the $r$ columns of $B$, so the column space has dimension
at most $r$.

### Two framings of the same object (pick the one that sticks)

1. **Subspace projection.** Read $\Delta W\, x = B(Ax)$ right-to-left. $A$
   projects the input down into an $r$-dim coordinate system; $B$ re-expands
   those $r$ numbers back out into $\mathbb{R}^d$. The update can only "see" the
   input through an $r$-dimensional keyhole.
2. **Bottleneck autoencoder on the weight delta.** $A$ is an encoder
   $\mathbb{R}^k \to \mathbb{R}^r$, $B$ a decoder $\mathbb{R}^r \to \mathbb{R}^d$,
   with no nonlinearity between them. $\Delta W$ is the linear autoencoder's
   effective transform, and $r$ is the bottleneck width.

   Tradeoff between the lenses: the projection view makes the *rank constraint*
   obvious (why outputs are confined to a subspace). The autoencoder view makes
   the *capacity* intuition obvious (a wider bottleneck stores more) and connects
   to things you already know, but it's a slightly leaky analogy — there's no
   reconstruction loss and no nonlinearity, so don't over-trust it. Use framing 1
   for "why is this restrictive," framing 2 for "how much can it learn."

### Parameter count and the payoff

$$
\#\text{params}(BA) = \underbrace{d\,r}_{B} + \underbrace{r\,k}_{A} = r\,(d + k).
$$

Compression vs. full fine-tuning of that matrix:

$$
\frac{r(d+k)}{d\,k}.
$$

Concrete, this project ($d = k = 896$, $r = 8$):

$$
r(d+k) = 8 \times (896 + 896) = 8 \times 1792 = 14{,}336 \text{ params,}
$$

$$
\frac{14{,}336}{802{,}816} \approx 0.01786 \approx 1.79\%.
$$

So per square attention matrix we train about **1.8%** as many parameters — a
$\sim 56\times$ reduction ($802816 / 14336 \approx 56.0$). For the square case
this simplifies to $\frac{2r}{d} = \frac{16}{896} \approx 1.79\%$, which is a
nice closed form to remember: **when $d=k$, the fraction is just $2r/d$.**

Caveats, stated plainly:

- This is **per matrix** and **approximate** — the global trainable fraction
  depends on which modules you adapt (config adapts attention only, not MLP) and
  on GQA making $k/v$ smaller. The real number comes from
  `print_trainable_parameters()`, expect well under 1% of total model params.
- Low param *count* is not the same as low *rank capacity*. We bought cheapness
  by **assuming** the rank-$r$ constraint is good enough. Whether $r=8$ actually
  captures the needed update is an empirical question — see §Open questions and
  the whole of `02-rank-and-alpha.md`.

---

## 3. The forward pass and why multiplication order matters

The adapted layer computes

$$
h = W_0 x + \Delta W x = W_0 x + B(Ax).
$$

(Ignore the $\alpha/r$ scale for one section; §5 adds it.) Trace the shapes,
right to left, for a single input vector $x$:

| step | expression | shape | what happened |
|------|------------|-------|---------------|
| input | $x$ | $k$ | the activation entering the layer |
| down-project | $Ax$ | $r$ | $(r\times k)(k) \to r$; squeeze into the bottleneck |
| up-project | $B(Ax)$ | $d$ | $(d\times r)(r) \to d$; expand back to output space |
| base path | $W_0 x$ | $d$ | $(d\times k)(k) \to d$; the frozen pretrained output |
| sum | $W_0x + B(Ax)$ | $d$ | both live in $\mathbb{R}^d$, add elementwise |

Both terms must land in $\mathbb{R}^d$ to be addable — they do. Good.

### Associativity is free correctness but not free performance

Matrix multiplication is associative: $(BA)x = B(Ax)$ as *values*. They are
mathematically equal. But the FLOP costs differ wildly, and that's the whole
runtime story.

- **$(BA)x$** — materialize $\Delta W = BA$ first, then apply it.
  - Forming $BA$: $(d\times r)(r\times k)$ costs $\approx d\,r\,k$ multiply-adds.
  - Applying to $x$: $\approx d\,k$.
  - For $d=k=896, r=8$: forming costs $896\cdot8\cdot896 \approx 6.4\times10^6$.
    You pay the *full dense* cost every forward pass, having gained nothing at
    train time.

- **$B(Ax)$** — keep it factored, never materialize $\Delta W$.
  - $Ax$: $(r\times k)(k)$ costs $\approx r\,k = 8\cdot896 = 7168$.
  - $B(\cdot)$: $(d\times r)(r)$ costs $\approx d\,r = 896\cdot8 = 7168$.
  - Total $\approx r(d+k) = 14{,}336$ — same number as the *parameter* count,
    which is not a coincidence (one multiply-add per param per input).

Ratio: $6.4\times10^6$ vs $1.4\times10^4$, roughly $\frac{dk}{r(d+k)} = \frac{1}{2r/d} \approx 56\times$ cheaper to keep it factored. So during **training**, the
library always evaluates $B(Ax)$, never $(BA)x$. This mirrors a thing you already
know from query planning: the join is associative, but the planner picks the
order that keeps the intermediate small. Here the small intermediate is the
$r$-vector $Ax$.

The flip side: at **inference** you often *do* want to pay the one-time cost of
forming $BA$ and folding it into $W_0$, because then every forward pass is a
single dense matmul with zero extra ops. That's merging — §7.

---

## 4. Initialization: $A$ random, $B = 0$

LoRA initializes

$$
A \sim \mathcal{N}(0, \sigma^2) \;\;(\text{e.g. Kaiming-scaled Gaussian}), \qquad B = 0.
$$

The consequence is the key property:

$$
\Delta W = B A = \mathbf{0} \cdot A = \mathbf{0} \quad\text{at step 0.}
$$

So at initialization $h = W_0 x + 0 = W_0 x$ exactly. **Training starts at the
pretrained model**, bit-for-bit. The adapter is a no-op until gradients move it.
This is a genuinely nice property: you never inject random noise into a
carefully-pretrained network at step 0, so early training can't blow up the model
before it learns anything.

### Why not initialize both at zero?

If $A = 0$ *and* $B = 0$, then $\Delta W = 0$ still — but look at the gradients
(full derivation in §6):

$$
\frac{\partial L}{\partial B} = g\,(Ax)^\top, \qquad
\frac{\partial L}{\partial A} = B^\top g\, x^\top,
$$

where $g = \partial L / \partial h$. If $A=0$ then $Ax=0$ so
$\partial L/\partial B = 0$; if $B=0$ then $\partial L/\partial A = 0$. With
**both** zero, *both* gradients vanish — the adapter is stuck at the origin
forever, a saddle point it can never leave. Dead on arrival.

The asymmetric init ($A$ random, $B=0$) breaks this: $\Delta W$ still starts at
zero (so the model output is unchanged), but $A \ne 0$ means
$\partial L/\partial B \ne 0$, so $B$ moves on the first step, which then makes
$\partial L/\partial A \ne 0$, so $A$ moves on the second. The system unsticks.

### The mirror image, and the real tradeoff

You could equally choose $A = 0$, $B$ random — same $\Delta W = 0$, and by the
symmetric argument $A$ moves first. Both are used in practice; the choice
interacts with the scaling factor and with which of $A,B$ you'd rather have
"warm." What you must **not** do is make both nonzero (injects noise into a good
model) or both zero (saddle). So the honest framing: the constraint is
"$\Delta W = 0$ at init **and** not both factors zero," and within that there are
two valid choices with minor tradeoffs, not one mandated answer. PEFT's default
is $A$ random / $B$ zero; that's what `src/train.py` gets.

---

## 5. The $\alpha/r$ scaling factor

The update actually applied is not $BA$ but

$$
\Delta W = \frac{\alpha}{r}\, B A, \qquad h = W_0 x + \frac{\alpha}{r} B(Ax).
$$

Config: `r: 8`, `alpha: 16`, so $\alpha/r = 2$. Why is this here at all?

### Why scale, and why tie it to $r$

The factors $B, A$ have a free overall magnitude — you can absorb any constant
into them ($B \to cB$, $A \to A/c$ leaves $BA$ unchanged), so the *scale* of the
update is not pinned by the parametrization. The optimizer will find *some*
magnitude, but how big that magnitude is, and how sensitive it is to your
learning rate, drifts as you change $r$.

Intuition for the $r$-dependence: write the update applied to $x$ as a sum over
the $r$ rank-one components,

$$
\Delta W\,x = \sum_{i=1}^{r} b_i (a_i^\top x),
$$

where $b_i$ is column $i$ of $B$ and $a_i^\top$ is row $i$ of $A$. If each
component contributes a roughly fixed-magnitude term, then a sum of $r$ of them
grows with $r$. Increasing rank from 8 to 16 would roughly double the raw output
magnitude of the adapter, which means you'd have to re-tune the learning rate
every time you change $r$. Annoying and wasteful.

Dividing by $r$ normalizes this: $\frac{\alpha}{r}\sum_{i=1}^r(\cdot)$ keeps the
*effective* magnitude roughly constant as $r$ varies, so $\alpha$ becomes a knob
that means roughly the same thing across ranks. The point of the $1/r$ is
**decoupling**: you tune $\alpha$ (and the LR) once, then sweep $r$ for capacity
without re-tuning the magnitude. That's the property the $\alpha = 2r$ heuristic
buys — pick $\alpha = 2r$ and $\alpha/r = 2$ stays fixed at 2 whether $r$ is 8 or
64, so a rank sweep doesn't silently also become a learning-rate sweep.

### Three framings of what $\alpha$ "is"

1. **Just a constant LR multiplier (the deflationary view).** Since
   $\frac{\alpha}{r}$ multiplies the whole adapter output, and the adapter is
   linear in $B$, you can fold the constant into the effective learning rate for
   the adapter parameters. Under plain SGD this is *exactly* true:
   $\frac{\alpha}{r}$ and the LR are redundant up to how they interact with the
   $B=0$ init. Lens tradeoff: simplest, and correct under SGD — but it hides the
   $r$-decoupling motivation, and under Adam it's not the whole story (Adam
   normalizes by the gradient's running magnitude, so a constant prefactor partly
   cancels — see #2).
2. **A scale that decouples LR sensitivity from $r$ (the standard view).** As
   derived above: the $1/r$ exists specifically so $\alpha$ and LR keep their
   meaning across rank choices. This is why the config comments
   `alpha = 2r` and why `02-rank-and-alpha.md` treats $\alpha/r$, not $\alpha$
   alone, as the quantity that matters.
3. **Rank-stabilized scaling $\alpha/\sqrt{r}$ (rsLoRA).** Kalajdzievski (2023)
   argues the $1/r$ over-shrinks at high rank and *under*-trains large-$r$
   adapters; the variance-correct normalization for a sum of $r$ roughly
   independent rank-one terms is $1/\sqrt{r}$ (standard "sum of $r$ independent
   things has std $\propto \sqrt r$" argument). Tradeoff: $\alpha/\sqrt{r}$ tends
   to make large $r$ actually help, whereas vanilla $\alpha/r$ can plateau; but
   for small $r$ (like our 8) the difference is mild and the $\alpha/r$ default
   is fine. We'll measure this in `02-rank-and-alpha.md` rather than argue it.

Don't memorize a winner here. Memorize: **the prefactor exists to stabilize the
update's magnitude against the choice of $r$; whether the right exponent is $1$
or $1/2$ is an empirical, rank-dependent question.**

---

## 6. Gradient flow, and the optimizer-memory payoff

Let $L$ be the scalar loss and let the layer output be $h = W_0 x + s\,B(Ax)$
with $s = \alpha/r$ a constant. Let upstream gradient $g \equiv \dfrac{\partial L}{\partial h} \in \mathbb{R}^d$ (delivered by backprop from above). Treat $x$ as
fixed input to this layer. Using $h = W_0 x + s\,BAx$ and the standard matrix
calculus rule $\partial(\text{tr}\,) $ / outer-product form:

**Gradient w.r.t. $B$** (shape must match $B$: $d\times r$):

$$
\frac{\partial L}{\partial B} = s\, g\,(Ax)^\top
\;\in\; \mathbb{R}^{d\times r},
\qquad (d\times 1)(1\times r) = d\times r. \checkmark
$$

**Gradient w.r.t. $A$** (shape must match $A$: $r\times k$):

$$
\frac{\partial L}{\partial A} = s\, (B^\top g)\, x^\top
\;\in\; \mathbb{R}^{r\times k},
\qquad (r\times d)(d\times 1)(1\times k) \to (r\times1)(1\times k) = r\times k. \checkmark
$$

Both exist and are well-defined — the factorization is differentiable
everywhere (it's just two matmuls). The chain rule walks: $g$ comes down from
above, hits $B$ to form $B^\top g \in \mathbb{R}^r$ (the gradient at the
bottleneck), and that flows into $A$. Symmetric to the forward keyhole.

**Gradient w.r.t. $W_0$:** we *deliberately do not compute or apply it.* $W_0$ is
**frozen** — marked `requires_grad = False`. Mathematically $\partial L/\partial
W_0 = g\,x^\top$ exists, but we never form it and never update $W_0$. The base
model is a constant.

### Why this saves optimizer memory (the real win on a 14 GB box)

The headline "1.8% of params" undersells it, because the optimizer state is
where fine-tuning actually OOMs. Adam keeps, **per trainable parameter**, two
fp32 running statistics — first moment $m$ and second moment $v$ — on top of the
param itself and its gradient. Rough per-trainable-param memory:

$$
\underbrace{1}_{\text{param}} + \underbrace{1}_{\text{grad}} + \underbrace{2}_{\text{Adam } m, v}
= 4 \text{ fp32 slots} = 16 \text{ bytes (fp32).}
$$

The Adam states are the *extra* $2\times$ on top of params+grads, and crucially
they scale with the number of **trainable** params, not total params. Frozen
$W_0$ needs none of this — no grad, no $m$, no $v$ — it just sits there as a
constant we read during the forward pass.

Trainable count, attention-only — **measured**, not estimated. Running
`make train` printed:

```
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
```

That $1{,}081{,}344$ is reproducible by hand once you account for GQA (§1). With
$r=8$, LoRA params per matrix are $r(d+k)$:

| proj | shape $d\times k$ | $r(d+k)$ |
|------|-------------------|----------|
| `q_proj` | $896\times896$ | $14{,}336$ |
| `k_proj` | $128\times896$ | $8{,}192$ |
| `v_proj` | $128\times896$ | $8{,}192$ |
| `o_proj` | $896\times896$ | $14{,}336$ |
| **per layer** | | $45{,}056$ |

$45{,}056 \times 24\text{ layers} = 1{,}081{,}344$ — exact. The naive square
approximation ($24\times4\times14{,}336 \approx 1.38\times10^6$) overcounts by
~300k precisely because $k/v$ are narrow. Trainable fraction: $0.2184\%$ of the
$495\text{M}$-param model — well under 1%, as promised.

- LoRA optimizer state (Adam $m,v$, fp32):
  $1.08\times10^6 \times 2 \times 4\,\text{B} \approx 8.6$ MB.
- Full fine-tuning of those same matrices: $\approx 7.7\times10^7$ params, so
  Adam state $\approx 7.7\times10^7 \times 2 \times 4\,\text{B} \approx 616$ MB —
  and that's *attention only*; include MLP and the full-model optimizer state
  runs into multiple GB, plus full-precision grads for everything, plus a
  full-size checkpoint every save. The $\sim 56\times$ param ratio shows up
  again, now denominated in RAM you don't have. This is the difference between a
  run that fits and one that doesn't.

(Saved-adapter size is a third, separate win: you serialize only $A,B$ — a few
MB — not a half-gigabyte checkpoint. See `model.save_pretrained` in
`src/train.py`.)

---

## 7. Why it merges for free at inference

Training keeps $W_0, A, B$ separate so gradients flow only into $A, B$. But
nothing forces inference to keep them apart. Because the adapter is a *linear*
additive correction in the *same space* as $W_0$, you can fold it in once:

$$
W' = W_0 + \frac{\alpha}{r} B A \;\in\; \mathbb{R}^{d\times k}.
$$

Shape check: $W_0$ is $d\times k$, and $\frac{\alpha}{r}BA$ is $d\times k$ (§3),
so $W'$ is $d\times k$ — **the original shape.** After merging, the layer is just
$h = W'x$: one dense matmul, identical in cost to the un-adapted model. Zero
added parameters at load, zero added FLOPs per token, no PEFT dependency at
inference time. The factored form was only ever a *training-time* convenience
(§3); at serving time you cash it in.

This is exactly what [`src/merge.py`](../../src/merge.py) does —
`merged = model.merge_and_unload()` computes $W_0 + \frac{\alpha}{r}BA$ for every
adapted matrix and returns a plain model. Contrast with adapter approaches that
add *new layers* (e.g. serial bottleneck adapters): those can't be folded into an
existing weight and do add inference latency. LoRA's "additive, same-shape,
linear" design is precisely what makes the merge possible — that's not an
accident, it's the reason the parametrization was chosen this way.

One real tradeoff worth naming: once merged you lose the ability to *swap*
adapters cheaply or keep several around to mix at runtime. If you want
multi-adapter serving (different LoRAs per request), you keep them *unmerged* and
eat the small $B(Ax)$ cost per layer. Merge is for "one final model, lowest
latency"; unmerged is for "many adapters, flexibility." For this repo we merge,
because we're producing one model and we care about understanding the fold.

---

## Open questions — verify these empirically, don't take my word

The math above is exact given the rank-$r$ *assumption*. Whether the assumption
is *good* is something to measure. Things I'd actually run:

1. **Does $r = 8$ capture the update?** Train at $r \in \{2, 4, 8, 16, 32\}$ on
   the same data slice and plot final loss vs. $r$. Where does it plateau? If
   loss barely improves past $r=4$, the intrinsic rank really is tiny and $r=8$
   is comfortable headroom. If it keeps dropping to $r=32$, our cheap assumption
   is leaving accuracy on the table. (A sharper probe: take the trained
   $\Delta W = \frac{\alpha}{r}BA$, run its SVD, and look at the singular-value
   decay — how many directions actually carry weight?)

2. **What does mis-setting $\alpha$ do to the loss curve?** Hold $r=8$ fixed and
   sweep $\alpha \in \{4, 8, 16, 32, 64\}$. The §5 claim is that $\alpha/r$ acts
   like an effective-LR scale on the update. Do you see the classic too-small =
   underfit / barely moves, too-large = unstable or diverges signature? Does
   $\alpha/\sqrt r$ scaling (framing #3) change where the sweet spot sits? This
   is the empirical core of `02-rank-and-alpha.md`.

3. **Why attention-only `target_modules`, and not the MLP?** The config adapts
   `q,k,v,o` but leaves the (larger, more numerous) MLP weights frozen. Is that a
   capacity choice, a cost choice, or folklore? Test it: add MLP modules
   (`gate_proj, up_proj, down_proj`) to `target_modules` and compare loss vs. the
   extra trainable params and RAM. Does adapting MLP help per-param more or less
   than adapting attention? The original LoRA paper adapted attention only; find
   out for *this* model and task whether that still holds.

4. **Is a single global $r$ even the right knob under GQA?** The measured count
   (§6) shows `k_proj`/`v_proj` map into a 128-dim output but still get $r=8$.
   Rank 8 on a 128-dim space caps the update at $8/128 = 6.25\%$ of full rank;
   rank 8 on the 896-dim `q`/`o` outputs caps it at $8/896 = 0.89\%$. So the
   *same* $r$ is a far looser constraint on $k/v$ than on $q/o$ — uniform $r$ is
   not uniform capacity. Does that matter? PEFT supports per-module `rank_pattern`
   / `alpha_pattern`; one could give $q/o$ a higher rank than $k/v$ (or vice
   versa). Worth a controlled run in `02-rank-and-alpha.md`: does spending rank
   budget on the wide matrices beat spreading it evenly?
