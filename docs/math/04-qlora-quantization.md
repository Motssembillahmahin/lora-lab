# 04 — QLoRA quantization: NF4, double-quant, and the memory equation

> Fourth in the math series, after [`01-lora-derivation.md`](./01-lora-derivation.md)
> (the $\Delta W = \frac{\alpha}{r}BA$ derivation, GQA shapes, optimizer-memory
> math), [`02-rank-and-alpha.md`](./02-rank-and-alpha.md) (rank/$\alpha$ sweep,
> rsLoRA), and [`03-loss-masking.md`](./03-loss-masking.md) (causal-LM loss). This
> one is about the *base model*, not the adapter: how QLoRA stores the frozen
> $W_0$ in 4 bits, and why that costs almost nothing in accuracy. The LoRA math
> from 01 is unchanged — QLoRA only swaps how $W_0$ is *represented*. Reference:
> Dettmers, Pagnoni, Holtzman, Zettlemoyer, **"QLoRA: Efficient Finetuning of
> Quantized LLMs," NeurIPS 2023** (arXiv:2305.14314).
> Cross-refs: the planned `src/quant.py` (NF4 codec demo — see §5), and the CPU
> constraints in [`CLAUDE.md`](../../CLAUDE.md).

---

## 0. CPU caveat — read this first (it changes what "we do" means)

**Real QLoRA training requires `bitsandbytes` + CUDA. This box is CPU-only. We do
not, and cannot, run 4-bit fine-tuning here.** That is a hard constraint, stated
up front so nothing below is misread as "run this."

The honest split between *math we can fully study* and *kernels we cannot run*:

- **What `bitsandbytes` actually provides:** fused CUDA kernels for (a) packing
  weights into NF4, (b) **dequant-on-the-fly** inside the matmul — it
  dequantizes a 4-bit tile to bf16 in registers/shared memory, multiplies, and
  never materializes the full bf16 weight in HBM — and (c) paged optimizers via
  CUDA unified memory. None of these exist for CPU. There is no CPU NF4 matmul
  kernel; you would dequantize the *entire* weight to bf16 first, at which point
  you have spent the memory you were trying to save. So 4-bit training buys
  nothing on CPU and isn't supported anyway.
- **What we *can* do, in pure numpy/torch, and learn everything from:** implement
  the NF4 **codec** — quantize a real tensor to 4-bit indices + scales,
  dequantize back, and *measure the reconstruction error* $\|W_0 - \hat W_0\|$.
  Compare NF4 vs uniform INT4 on (i) synthetic Gaussian data and (ii) a real
  Qwen2.5-0.5B weight matrix. Verify the double-quant bit accounting numerically.
  This is what the planned `src/quant.py` is for (§5). Every equation in §§1–3 is
  exercisable this way; only the *fused training kernel* is out of reach.

Keep this distinction sharp: the **information theory** of NF4 (why 4 bits placed
at Gaussian quantiles beats 4 bits placed uniformly) is platform-independent and
is the actual intellectual content. The CUDA kernel is just the thing that makes
it *fast and memory-cheap at train time*, which is the part we can only read
about.

---

## 1. Uniform quantization — the baseline NF4 has to beat

Quantization maps a high-precision tensor (think fp16/bf16 weights) onto a small
finite set of levels addressed by $b$-bit integers, so storage drops from 16 bits
to $b$ bits per element. Everything in QLoRA is $b = 4$. We build up from the
simplest scheme so the NF4 win in §2 is measured against a fair baseline.

### 1.1 Absmax (symmetric) quantization

Take a block of values $x \in \mathbb{R}^n$ (for now, the whole tensor). Symmetric
absmax quantization picks a single scale from the largest magnitude and maps the
range $[-\max|x|, +\max|x|]$ onto the signed integer grid. For $b$ bits the signed
grid has $2^b$ levels, spanning $-(2^{b-1})\,..\,(2^{b-1}-1)$; using the symmetric
positive count $2^{b-1}-1$ as the full-scale integer:

$$
q_i = \operatorname{round}\!\left(\frac{x_i}{\max_j |x_j|}\cdot (2^{b-1}-1)\right)
\;\in\; \mathbb{Z},
\qquad
\hat x_i = q_i \cdot \frac{\max_j |x_j|}{2^{b-1}-1}.
$$

Define the **scale** (a.k.a. quantization constant)

$$
c \;=\; \frac{\max_j|x_j|}{2^{b-1}-1},
\qquad
q_i = \operatorname{round}(x_i / c), \quad \hat x_i = c\,q_i.
$$

So we store $q_i$ in $b$ bits and one fp16 scale $c$ for the block. The step
between adjacent representable values is exactly $c$.

Shape sanity: $x \in \mathbb{R}^n \to q \in \mathbb{Z}^n$ (each entry now $b$ bits)
$+$ one scalar $c$. For a weight matrix $W_0 \in \mathbb{R}^{d\times k}$ flattened
to $n = dk$, that is $n$ four-bit codes plus the scales (count depends on
blocking, §1.3).

### 1.2 Quantization error — derive the worst case

Let $\varepsilon_i = x_i - \hat x_i$ be the per-element error. Because
$\hat x_i = c\cdot\operatorname{round}(x_i/c)$ and rounding to the nearest integer
moves its argument by at most $\tfrac12$:

$$
\left|\frac{x_i}{c} - \operatorname{round}\!\Big(\frac{x_i}{c}\Big)\right| \le \tfrac12
\;\Longrightarrow\;
|\varepsilon_i| = c\left|\frac{x_i}{c} - q_i\right| \le \frac{c}{2}
= \frac{\max_j|x_j|}{2\,(2^{b-1}-1)}.
$$

So the **worst-case** absolute error is half the quantization step:

$$
\boxed{\;\max_i |\varepsilon_i| = \frac{\max_j|x_j|}{2\,(2^{b-1}-1)}.\;}
$$

This is the central tension. The error scales with $\max|x|$ (set by the single
largest entry, often an **outlier**) and shrinks with $2^{b-1}-1$ (the number of
levels). Two consequences fall out immediately.

### 1.3 INT4 vs INT8 — the bit count amplifies error by 256×

Hold $\max|x|$ fixed and compare $b=4$ to $b=8$:

$$
2^{b-1}-1 \Big|_{b=4} = 2^3 - 1 = 7,
\qquad
2^{b-1}-1 \Big|_{b=8} = 2^7 - 1 = 127.
$$

$$
\frac{\max|\varepsilon|_{\text{INT4}}}{\max|\varepsilon|_{\text{INT8}}}
= \frac{127}{7} \approx 18.1.
$$

The per-step error is ~18× larger at INT4. People often quote a "256×" figure —
that is the **expected squared error** ratio, which is what actually shows up in a
reconstruction-MSE or in downstream loss. Squaring the step ratio:

$$
\left(\frac{127}{7}\right)^2 \approx 329,
\qquad
\text{or with the clean powers-of-two } \left(\frac{2^7}{2^3}\right)^2 = (2^4)^2 = 2^8 = 256.
$$

(The $256$ is the idealized $b\to b-4$ bit-drop: each bit halves the step, so 4
fewer bits is $2^4 = 16\times$ the step and $16^2 = 256\times$ the MSE. The $329$
uses the exact $2^{b-1}-1$ level counts. Same story, two roundings.) The takeaway:
**dropping from 8 to 4 bits is not a mild compression — it inflates squared error
by over two orders of magnitude.** Naively, 4-bit weights should wreck the model.
NF4 (§2) and blockwise scaling (§1.4) are the two ideas that make 4-bit survivable
anyway.

### 1.4 Blockwise quantization — kill the outlier's blast radius

The problem with a single per-tensor scale: one giant outlier sets $\max|x|$ for
the *entire* tensor, so $c$ is large, so the step is coarse, so **every** other
(small, typical) weight is quantized coarsely. The outlier holds the whole tensor
hostage.

Fix: chop the flattened tensor into contiguous **blocks** of size $B$ and give
each block its own scale.

$$
x = [\,\underbrace{x^{(1)}}_{B}\,|\,\underbrace{x^{(2)}}_{B}\,|\cdots|\,\underbrace{x^{(n/B)}}_{B}\,],
\qquad
c_i = \frac{\max_{j\in\text{block }i}|x_j|}{2^{b-1}-1},
\qquad
q_j = \operatorname{round}(x_j / c_{\text{block}(j)}).
$$

Now an outlier inflates the step only **within its own block** of $B$ values; the
other $n/B - 1$ blocks keep tight scales. This is a variance-reduction trick: the
local $\max|x^{(i)}|$ tracks each block's actual dynamic range instead of being
dominated by a single global extreme. QLoRA uses $B = 64$.

**Cost of blocking — the scale overhead.** Per-tensor stored one fp16 scale total
(negligible). Blockwise stores one fp16 scale per block, i.e. $n/B$ scales of 16
bits each. Amortized over the $n$ weights:

$$
\text{scale overhead} = \frac{(n/B)\cdot 16 \text{ bits}}{n}
= \frac{16}{B} \;\text{bits/param}.
$$

For $B = 64$: $\frac{16}{64} = 0.25$ bits/param. So blockwise NF4 is really
$4 + 0.25 = 4.25$ bits/param **before** we get clever about the scales — and §3
(double-quant) is precisely the trick to shrink that 0.25.

> Engineering analogy (honest): blockwise scaling is per-shard normalization. One
> hot key (the outlier) shouldn't degrade the encoding of every other key in the
> table — so you scope the scale to a shard. Smaller shards (smaller $B$) =
> tighter local scales = better fidelity, but more per-shard metadata (more
> scales). The $B$ knob is the fidelity/metadata tradeoff, exactly like shard
> size.

### 1.5 Signed vs unsigned, and why the level count is 7 or 8

A small but real bookkeeping fork. With $b=4$ you have $2^4 = 16$ codes. How you
spend them changes the full-scale divisor:

- **Unsigned magnitude grid** $\{0,\dots,15\}$ used as a signed range about zero:
  the positive half-range is $7$ steps ($\max|x|/7$), matching the
  $2^{b-1}-1 = 7$ in §1.1. This is the "divide by 7" you'll see for INT4 absmax.
- **Signed grid** $\{-8,\dots,+7\}$: the magnitude extends to $8$ on the negative
  side, so a "divide by 8" appears if you normalize to the most-negative code.

Both are correct; they differ in which code is the full-scale anchor and whether
zero is exactly representable. NF4 (§2) sidesteps this entirely — its levels are
*not* an integer grid at all, they're chosen by a quantile function, and it pins
$\max(|\text{levels}|) = 1$ so the block scale $c = \max|x^{(i)}|$ directly
(no $\div 7$). Keep that in mind: the "$/7$ vs $/8$" question is a uniform-grid
artifact that NF4 makes moot.

---

## 2. NF4 — Normal Float 4, the key QLoRA contribution

Uniform quantization spends its 16 levels **evenly across the value range**. That
is optimal only if the values are uniformly distributed. Pretrained weights are
not — they are approximately **zero-mean Gaussian**. NF4 is the observation that
if you *know* the source distribution, you should place your scarce 16 levels
where the probability mass actually is.

### 2.1 The motivation, made precise

Let the (block-normalized) weights be modeled as $x \sim \mathcal{N}(0,\sigma^2)$.
A uniform grid puts equally-spaced levels from $-\max|x|$ to $+\max|x|$. But for a
Gaussian, $\max|x|$ sits ~3–4$\sigma$ out in the tail, where there is almost no
mass — yet a uniform grid dedicates several of its 16 precious levels to that
near-empty tail region, and correspondingly **starves the dense region near
zero** where most weights actually live. You are spending bits on values that
rarely occur and under-resolving the values that occur constantly.

NF4 reallocates: dense levels near 0, sparse levels in the tails — matching the
PDF.

### 2.2 Information-optimal levels = equal-probability-mass bins (quantiles)

Here is the principle. Suppose you must pick 16 representative levels for a
continuous source with PDF $p(x)$ and CDF $\Phi$. A *quantile* quantizer chooses
the bin boundaries so that **each bin carries equal probability mass** $1/N$
(here $N=16$). Equivalently, the levels are the quantiles

$$
\ell_m = \Phi^{-1}\!\Big(\frac{m + \tfrac12}{N}\Big),
\qquad m = 0, 1, \dots, N-1,
$$

where $\Phi^{-1}$ is the inverse CDF (for a Gaussian, the **probit**/normal
quantile function). The intuition: where the PDF is tall (near 0), equal mass
$1/N$ spans a *narrow* $x$-interval, so levels pack tightly; where the PDF is low
(tails), equal mass spans a *wide* $x$-interval, so levels spread out. The level
density automatically tracks $p(x)$.

> Lens 1 — **information theory.** Equal-mass bins mean each of the 16 codes is
> used with equal frequency $1/16$, so the code has maximum entropy: every 4-bit
> symbol carries a full 4 bits of information. Uniform-grid codes are used with
> wildly unequal frequency (the near-zero codes fire constantly, the tail codes
> almost never), wasting code-space. This is the same instinct as Huffman/entropy
> coding — match the code to the source statistics.
>
> Lens 2 — **estimation / MSE (Lloyd–Max).** Equal-mass bins are (to first order)
> the minimizer of expected squared reconstruction error for the source. See §2.5
> for the argument. Tradeoff between lenses: the information lens explains *why
> the codes are efficient symbols*; the MSE lens explains *why the
> reconstruction is accurate*. They coincide for smooth log-concave densities like
> the Gaussian, which is why one construction serves both.

### 2.3 The asymmetric 8-1-7 split and why zero must be exact

A subtlety the pure quantile formula misses: we want **0 to be exactly
representable**. Zero matters because (a) many weights are near zero, (b) padding
/ masked / pruned entries are exactly zero, and (c) a symmetric codec that can't
represent 0 introduces a systematic bias. But $N=16$ is even, and a symmetric set
of nonzero levels about 0 would use all 16 on nonzero values with none landing on
0. So NF4 builds an **asymmetric** set:

$$
\underbrace{8 \text{ negative levels}}_{\text{quantiles of the left half}}
\;\;+\;\;
\underbrace{1 \text{ exact zero}}_{} \;\;+\;\;
\underbrace{7 \text{ positive levels}}_{\text{quantiles of the right half}}
\;=\; 16 \text{ levels}.
$$

Construction (following Dettmers et al. §3): compute quantiles separately for the
negative and positive halves of $\mathcal{N}(0,1)$ — using $2^{b-1}+1 = 9$
quantile points on one side and $2^{b-1} = 8$ on the other so that the two halves
**share** the single zero — then concatenate, drop the duplicate zero, and
**normalize so $\max(|\ell_m|) = 1$** (the most-negative level becomes $-1$, the
most-positive becomes $+1$). The two halves use a slightly different number of
quantile offsets, which is exactly why the resulting table is *not* symmetric
($|\ell_{\min}| \ne |\ell_{\max}|$ in the interior).

### 2.4 The 16 NF4 levels

Normalized to $[-1, 1]$ with exact zero, the NF4 codebook (Dettmers et al., 2023)
is approximately:

$$
\begin{aligned}
\text{neg: } &-1.0,\; -0.6962,\; -0.5250,\; -0.3949,\; -0.2844,\; -0.1848,\; -0.0911,\\
\text{zero: } &\phantom{-}0.0,\\
\text{pos: } &\phantom{-}0.0796,\; 0.1609,\; 0.2461,\; 0.3379,\; 0.4407,\; 0.5626,\; 0.7229,\; 1.0.
\end{aligned}
$$

Read the **spacing**, that's the whole point:

- Near zero the gap between adjacent levels is small ($0.0796 - 0 = 0.0796$;
  $0.1609 - 0.0796 \approx 0.081$) — **fine resolution where mass concentrates.**
- Out toward $\pm1$ the gap blows up ($1.0 - 0.7229 = 0.2771$) — **coarse
  resolution in the sparse tails.**

A uniform 16-level grid on $[-1,1]$ would have a *constant* gap of
$\frac{2}{15} \approx 0.133$ everywhere. NF4's near-zero gap (~0.08) is roughly
**half** that, doubling resolution exactly where weights live, and pays for it
with coarse tails (~0.28 gap) where almost nothing lives.

### 2.5 Why equal-mass bins minimize expected squared error (Lloyd–Max)

The claim: for a source $x\sim p(x)$, the quantizer with reconstruction levels
$\{\ell_m\}$ and bin boundaries $\{t_m\}$ that minimizes
$\mathbb{E}[(x-\hat x)^2]$ satisfies the **Lloyd–Max** conditions. Write the MSE:

$$
D = \sum_{m} \int_{t_m}^{t_{m+1}} (x - \ell_m)^2\, p(x)\, dx.
$$

Stationarity in the two sets of variables gives the two famous conditions:

1. **Centroid condition** ($\partial D/\partial \ell_m = 0$): each level is the
   conditional mean of its bin,
   $$
   \ell_m = \mathbb{E}[\,x \mid t_m \le x < t_{m+1}\,]
          = \frac{\int_{t_m}^{t_{m+1}} x\,p(x)\,dx}{\int_{t_m}^{t_{m+1}} p(x)\,dx}.
   $$
2. **Nearest-neighbor condition** ($\partial D/\partial t_m = 0$): each boundary
   is midway between the two surrounding levels,
   $$
   t_m = \tfrac12(\ell_{m-1} + \ell_m).
   $$

Now the **high-resolution** approximation (Panter–Dite / Bennett), valid when $N$
is large enough that $p(x)$ is ~constant across each bin: minimizing $D$ subject
to $N$ levels yields an optimal level *density* proportional to $p(x)^{1/3}$, and
the resulting distortion is

$$
D^\star \;\approx\; \frac{1}{12\,N^2}\left(\int p(x)^{1/3}\,dx\right)^{3}.
$$

The point for us: the optimal allocation puts **more levels where $p(x)$ is
large** — qualitatively exactly what the equal-mass / quantile construction does.
NF4's equal-probability bins are the *quantile* quantizer, which is the
information-theoretically clean cousin of the strict $p^{1/3}$ MSE-optimal
quantizer; for the Gaussian both heavily concentrate levels near 0 and the
difference between "$p$-proportional mass" and "$p^{1/3}$ density" is a second-order
refinement. Dettmers et al. deliberately use the equal-mass quantile construction
(it's simpler, data-free given the Gaussian assumption, and the codebook is fixed
once and baked in). **The honest statement: NF4 is the equal-mass quantile
quantizer for $\mathcal N(0,1)$, which is near-optimal — not exactly Lloyd–Max
optimal — in MSE, and exactly optimal in code entropy.**

### 2.6 Quantize / dequantize as table lookup

With the fixed 16-level codebook $\{\ell_0,\dots,\ell_{15}\}$ (the §2.4 table) and
a per-block scale $c_i = \max_{j\in\text{block }i}|x_j|$ (recall NF4 normalizes the
table to $\max|\ell| = 1$, so the scale is just the block absmax — no $\div 7$):

**Quantize** $x_j$ in block $i$:
$$
\tilde x_j = x_j / c_i \in [-1, 1],
\qquad
\text{code}(j) = \arg\min_{m \in \{0,\dots,15\}} \big|\,\tilde x_j - \ell_m\,\big|.
$$
Store the 4-bit `code(j)`. Finding the nearest level is a 16-way comparison (or a
binary search over the sorted table) — cheap, and on GPU it's a fused kernel.

**Dequantize** (pure table lookup + scale):
$$
\hat x_j = c_i \cdot \ell_{\text{code}(j)}.
$$

No arithmetic decoding, just an index into a 16-entry LUT times the block scale.
This is what "dequant-on-the-fly" does per tile in the forward pass (§4).

Shapes: a weight $W_0 \in \mathbb{R}^{d\times k}$, $n = dk$, becomes $n$ four-bit
codes ($\frac{n}{2}$ bytes, two codes per byte) $+$ $n/B$ block scales. The
codebook itself is 16 shared constants — negligible, stored once for the whole
model.

### 2.7 Why NF4 beats uniform INT4 on Gaussian inputs — the analytic argument

Both schemes spend the same 4 bits and the same per-block absmax scale; the only
difference is **where the 16 levels sit**. Compare expected squared error
$\mathbb{E}[(x-\hat x)^2]$ for $x\sim\mathcal N(0,1)$:

- **Uniform INT4:** levels equally spaced; bin width constant $\approx \Delta$.
  The near-zero bins (huge mass) and tail bins (tiny mass) all have the *same*
  width $\Delta$. Error is dominated by the near-zero bins because that's where
  the mass is, and those bins are *wider than they should be* (they share width
  with the empty tail bins). The MSE integral $\int (x-\hat x)^2 p(x)\,dx$ is
  large precisely because the high-$p(x)$ region is under-resolved.
- **NF4:** near-zero bins are *narrow* (small $(x-\hat x)^2$ exactly where $p(x)$
  is large), tail bins are wide (large $(x-\hat x)^2$ but multiplied by tiny
  $p(x)$, so they barely contribute to the integral). Every term in
  $\int (x-\hat x)^2 p(x)\,dx$ is kept small *because the construction co-locates
  small error with large mass.*

Formally, NF4 approximates the minimizer of that integral (§2.5) while uniform
INT4 ignores $p(x)$ entirely. So for any non-uniform $p$ — and Gaussian is
strongly non-uniform — NF4's expected squared error is strictly lower. The gap
**grows** with how peaked the source is. This is the experiment to actually run
(§5): sample $x\sim\mathcal N(0,1)$, quantize both ways, and compare empirical
$\|x-\hat x\|_2^2$. NF4 should win clearly on Gaussian, and the margin should
**shrink** if you instead feed uniform-on-$[-1,1]$ data (where uniform INT4 is by
construction optimal and NF4's tail levels are wasted) — a nice falsifiable check
that we understand *why* it wins, not just *that* it wins.

> Caveat, stated honestly: "weights are Gaussian" is an approximation. Real
> attention/MLP weight blocks are roughly bell-shaped but can be heavier-tailed or
> slightly skewed, and outliers are real (that's why we block, §1.4). NF4's win is
> empirical and robust in the paper, but the clean "strictly lower MSE" statement
> is exact only under the literal Gaussian model. §5 measures it on *actual* Qwen
> weights to see how well the assumption holds for our model.

---

## 3. Double quantization — quantize the scales too

### 3.1 The leftover cost

After §1.4 we have, per parameter: $4$ bits for the NF4 code plus the blockwise
scale overhead. With fp16 scales at block size $B = 64$:

$$
\text{scale overhead} = \frac{16 \text{ bits}}{64} = 0.25 \text{ bits/param}.
$$

That 0.25 is not nothing — it's a 6.25% tax on top of the 4-bit payload
($0.25/4$). On a 7B model it's $\approx 0.22$ GB of pure metadata. Double
quantization (DQ) shrinks it.

### 3.2 The idea: the scales are themselves a tensor to be quantized

The first-level scales $\{c_1, c_2, \dots, c_{n/B}\}$ form their own array of
positive fp32/fp16 numbers. They have structure (they're all positive, similar
magnitude block-maxima), so they compress well. DQ quantizes *them* with an
8-bit blockwise scheme:

1. Group the first-level scales into **super-blocks** of $B_2 = 256$ scales each.
2. (Subtract the mean of the super-block — the scales are all positive, so
   centering them lets a symmetric 8-bit codec use its full range; this is a
   detail in the paper.)
3. Quantize each scale to **INT8** ($2^8 = 256$ levels — plenty for these smooth
   positive values) with one fp32 **super-scale** per super-block.

So the storage for scales becomes: 8-bit quantized first-level scales, plus a
tiny number of fp32 super-scales (one per 256 scales).

### 3.3 Full bit accounting — the table

Per parameter, $B = 64$ (first-level block), $B_2 = 256$ (second-level
super-block), NF4 payload 4 bits:

| component | precision | how many | bits per *param* |
|---|---|---|---:|
| NF4 weight code | 4-bit | 1 per param | $4$ |
| first-level scale (no DQ) | fp16 | $1$ per $64$ params | $16/64 = 0.25$ |
| first-level scale (**with DQ**) | INT8 | $1$ per $64$ params | $8/64 = 0.125$ |
| second-level super-scale | fp32 | $1$ per $64\times256$ params | $32/(64\cdot256) \approx 0.00195$ |

**Without DQ:** $4 + 0.25 = 4.25$ bits/param.

**With DQ:**
$$
4 \;+\; \underbrace{8/64}_{0.125} \;+\; \underbrace{32/(64\cdot256)}_{\approx 0.002}
\;\approx\; 4.127 \text{ bits/param}.
$$

DQ saves $0.25 - 0.127 \approx 0.123$ bits/param — about **0.5 bits per weight
relative to fp16 scales' worst case, ~3% of the total budget.** That matches the
paper's headline "~0.37 bits/param saved" framing once you account for their exact
block sizes; on a 65B model that's ~3 GB of VRAM, which is the difference between
fitting on a 48 GB card and not. The second-level super-scale term
($0.002$ bits/param) is genuinely negligible — listed for completeness so the
accounting closes, not because it matters.

> Why stop at two levels? You *could* quantize the super-scales too, but their
> overhead is already $0.002$ bits/param — three more levels of recursion would
> save microscopic amounts while adding dequant latency. The recursion has sharply
> diminishing returns; two levels is the engineering sweet spot. (Same shape as
> deciding how many index levels a B-tree needs: you stop when the next level's
> metadata is smaller than the lookup cost it saves.)

### 3.4 The reconstruction path with DQ

Dequantizing one weight now chains two lookups:

$$
\hat x_j
= \underbrace{\big(\text{INT8-dequant}(\,\hat c_i\,;\ \text{super-scale}_{s(i)})\big)}_{\text{recover first-level scale } c_i}
\;\cdot\;
\underbrace{\ell_{\text{code}(j)}}_{\text{NF4 lookup}}.
$$

Read it inside-out: recover the block's fp16-ish scale $c_i$ from its INT8 code
and the super-scale, then multiply by the NF4 level. Two cheap multiplies and two
table lookups per weight — fused into the matmul kernel on GPU so the
fully-reconstructed bf16 weight never lives in HBM.

---

## 4. QLoRA assembly — putting it on the LoRA skeleton

QLoRA = **NF4-quantized frozen base $W_0$** + **bf16 LoRA adapters $A, B$** +
paged optimizer. The LoRA math from `01` is untouched; only $W_0$'s storage
changes.

### 4.1 The forward pass: dequant-on-the-fly

From `01` §3 the adapted layer is $h = W_0 x + \frac{\alpha}{r}B(Ax)$. In QLoRA,
$W_0$ is stored as NF4 codes + (double-quantized) scales, and each forward pass
reconstructs it transiently:

$$
W_0^{\text{bf16}} = \operatorname{dequant}_{\text{NF4}}(W_0^{\text{nf4}}),
\qquad
h = W_0^{\text{bf16}}\, x \;+\; \frac{\alpha}{r}\,B(Ax).
$$

The crucial property: the dequant is **on-the-fly and tile-local**. The kernel
dequantizes a tile of $W_0$ to bf16 in fast memory, uses it in the matmul, and
discards it — the full bf16 $W_0$ is never materialized in HBM. Only the 4-bit
codes persist. This is the part that has no CPU equivalent (§0): on CPU you'd
dequant the whole matrix to bf16 first, defeating the purpose.

### 4.2 Where the gradient flows (and doesn't)

Identical to `01` §6, because $A, B$ are the only trainable tensors:

$$
\frac{\partial L}{\partial B} = \frac{\alpha}{r}\, g\,(Ax)^\top,
\qquad
\frac{\partial L}{\partial A} = \frac{\alpha}{r}\, (B^\top g)\, x^\top,
\qquad
\frac{\partial L}{\partial W_0^{\text{nf4}}} = \text{not computed.}
$$

$W_0$ is frozen — `requires_grad = False` — so **no gradient, no optimizer state,
no dequant of gradients** for the base. The dequantized $W_0^{\text{bf16}}$
appears in the forward graph only as a constant multiplier of $x$; gradient w.r.t.
$x$ flows through it (needed to reach earlier layers), but nothing flows *into*
the 4-bit codes because they require no grad. This is why a 4-bit base is
compatible with full-precision training of the adapter: the precision that matters
for the optimizer ($A, B$ and their Adam moments) is still bf16/fp32; only the
*frozen constant* is 4-bit.

> Subtlety worth flagging: the quantization error $W_0 - \hat W_0$ is a fixed,
> non-trainable perturbation of the frozen base. QLoRA's empirical finding is that
> the LoRA adapter, trained *on top of* the quantized base, can **absorb** much of
> the systematic part of that error — the adapter learns to compensate. That's a
> genuinely nice interaction: you're not just hoping the 4-bit error is small,
> you're giving the trainable part a chance to correct for it. (Don't overclaim —
> it compensates for the *learnable/systematic* component, not the random
> rounding noise.)

### 4.3 The memory equation — 7B model, three regimes

This is the number that justifies the whole exercise. Per trainable parameter,
Adam keeps the param + grad + two fp32 moments (`01` §6). Frozen params keep only
their stored representation. Take a 7B-parameter model ($P = 7\times10^9$).

**Full fine-tuning** (everything trainable; bf16 weights+grads, fp32 Adam
$m, v$):

$$
\underbrace{2P}_{\text{weights bf16}}
+ \underbrace{2P}_{\text{grads bf16}}
+ \underbrace{4P}_{\text{Adam } m\ \text{fp32}}
+ \underbrace{4P}_{\text{Adam } v\ \text{fp32}}
= 12P \text{ bytes}
= 12 \cdot 7\times10^9 \approx 84\ \text{GB}.
$$

**LoRA** (bf16 base frozen, only adapters trainable; adapter params
$P_{\text{adapt}} \approx 0.1\text{–}0.2P_{\%}$, here take $\approx 100$M
$\approx 0.014P$):

$$
\underbrace{2P}_{\text{base bf16, frozen}}
+ \underbrace{2P_{\text{adapt}}}_{\text{adapter bf16}}
+ \underbrace{8P_{\text{adapt}}}_{\text{grad+Adam fp32 on adapters}}
\approx 14\ \text{GB} + 0.2\ \text{GB} + 0.4\ \text{GB} \approx 14.6\ \text{GB}.
$$

**QLoRA** (NF4 base frozen at ~4.13 bits/param $\approx 0.52$ bytes/param, bf16
adapters trainable):

$$
\underbrace{0.52P}_{\text{base NF4, frozen}}
+ \underbrace{2P_{\text{adapt}} + 8P_{\text{adapt}}}_{\text{adapter + grad + Adam}}
\approx 3.5\ \text{GB} + 0.2\ \text{GB} + 0.4\ \text{GB} \approx 4.1\ \text{GB}.
$$

| regime | frozen base | trainable (param) | grad + optimizer | total (7B) |
|---|---|---|---|---:|
| Full FT | — (all trainable) | $2P = 14$ GB | $2P + 8P = 70$ GB | $\approx 84$ GB |
| LoRA | $2P = 14$ GB (bf16) | $\approx 0.2$ GB | $\approx 0.4$ GB | $\approx 14.6$ GB |
| QLoRA | $0.52P \approx 3.5$ GB (NF4) | $\approx 0.2$ GB | $\approx 0.4$ GB | $\approx 4.1$ GB |

Two distinct wins, stacked:

1. **LoRA vs Full FT** ($84 \to 14.6$ GB): freezing the base eliminates the
   $10P$ of grad+optimizer state (the dominant term — `01` §6's point). This is
   the *optimizer-memory* win and it's the bigger one.
2. **QLoRA vs LoRA** ($14.6 \to 4.1$ GB): quantizing the frozen base from bf16
   (2 bytes) to NF4 (~0.52 bytes) shrinks the one term LoRA *couldn't* touch — the
   frozen base itself — by ~4×. This is the *weight-storage* win.

The two are orthogonal: LoRA attacks the optimizer state, QLoRA additionally
attacks the frozen weights. That's why "4-bit base, full-precision adapter, full
optimizer math on a tiny adapter" all coexist coherently — each technique targets
a different line of the memory budget.

### 4.4 Paged optimizers (the third QLoRA ingredient)

Even the adapter's Adam state can spike during `optimizer.step()`, and gradient
checkpointing creates transient allocation bursts. **Paged optimizers** put the
Adam moments $(m_t, v_t)$ in CUDA *unified memory*: they live in CPU RAM and are
paged into GPU memory on demand during the step, then evicted. This addresses
**OOM spikes** (transient peaks that would otherwise crash a run that fits in
steady state), not steady-state footprint. It's the OS demand-paging trick applied
to optimizer tensors — VRAM is the "RAM," host memory is the "disk," and the
unified-memory driver handles the page faults.

**Irrelevant on CPU:** there's no GPU/host split to page across — everything is
already in the same RAM. We note it for completeness; it does nothing for us.

---

## 5. What we can actually demo on CPU (`src/quant.py`)

Per §0, the planned `src/quant.py` implements the NF4 **codec** in pure
numpy/torch — no `bitsandbytes`, no CUDA, no training. It exists to make §§1–3
*exercisable*, turning every equation above into a measured number. Scope:

1. **The NF4 codebook.** Either hard-code the §2.4 table or *derive* it from the
   normal quantile function $\Phi^{-1}$ (the 8-1-7 construction, §2.3) and verify
   it reproduces the published levels. Deriving it is the better exercise — it
   forces you to get the asymmetric split and the $\max|\ell|=1$ normalization
   right.
2. **Blockwise absmax.** Reshape a tensor into blocks of $B = 64$, compute
   per-block scales $c_i = \max|x^{(i)}|$, normalize, nearest-level lookup
   (§2.6), pack codes. Then dequantize and return the reconstruction.
3. **Double-quant bit accounting.** Numerically confirm the §3.3 table: count the
   actual bits stored (codes + INT8 scales + fp32 super-scales) and check it lands
   at $\approx 4.127$ bits/param for $B=64, B_2=256$.
4. **Reconstruction-error comparison.** The headline experiment (§2.7): quantize
   with NF4 vs uniform INT4 and report $\|x - \hat x\|_2^2$ (and max error, §1.2)
   on:
   - **Synthetic** $x \sim \mathcal N(0,1)$ — NF4 should win clearly.
   - **A real Qwen2.5-0.5B weight matrix** (load `q_proj.weight` from the base
     model, §0-legal — it's just reading fp16 weights, no training) — tests how
     well the Gaussian assumption (§2.7 caveat) holds for *our actual model*.
   - **Control:** $x \sim \text{Uniform}[-1,1]$ — uniform INT4 should win or tie
     here, confirming we understand *why* NF4 wins on Gaussian (it's the
     distribution, not magic).

What the demo deliberately does **not** do: any 4-bit *matmul* or *training*.
There is no CPU fused dequant-matmul kernel; doing it manually would mean
dequantizing the whole matrix to bf16 first (§0), which measures nothing about
QLoRA's memory claim. The codec round-trip and the error numbers are the entire,
honest, CPU-reachable lesson.

---

## What to verify empirically

The information theory in §2 is exact under the Gaussian model; how it cashes out
for *our* model and setup is measured.

1. **Does NF4 actually beat INT4 on real Qwen weights, and by how much?** Run the
   §5 codec on `q_proj`/`k_proj`/`down_proj` and tabulate
   $\|W_0-\hat W_0\|_2^2$ for NF4 vs uniform INT4 vs INT8. Expected: NF4 < INT4 on
   the Gaussian-ish attention weights; the gap should be largest where the weight
   histogram is most bell-shaped. If some MLP weight is heavy-tailed or bimodal,
   NF4's edge may shrink — that would tell us the Gaussian assumption is leaky for
   that layer, which is interesting in itself.

2. **How much does block size $B$ matter?** Sweep $B \in \{16, 32, 64, 128, 256\}$
   and plot reconstruction MSE vs scale-overhead bits ($16/B$). This makes the
   §1.4 fidelity/metadata tradeoff concrete: smaller $B$ → lower MSE → more scale
   bits. Is $B=64$ a defensible knee, or would $B=32$ be worth the extra
   $0.25\to0.5$ bits/param on this model?

3. **Does double-quant cost measurable accuracy?** Compare reconstruction error of
   single-quant (fp16 scales) vs double-quant (INT8 scales). The §3 claim is that
   8-bit scales are "plenty" for these smooth positive block-maxima. Measure the
   added error from quantizing the scales and confirm it's negligible relative to
   the 4-bit weight error — i.e. DQ buys 0.12 bits/param essentially for free.

4. **Can a LoRA adapter absorb NF4 error? (the §4.2 claim — out of full scope on
   CPU, but partially probeable.)** We can't train QLoRA, but we *can*: take the
   quantized base $\hat W_0$, compute the per-layer error $W_0 - \hat W_0$, and ask
   whether it's low-rank-ish (SVD it). If the systematic part of the quantization
   error lies in a low-rank subspace, that's *mechanistically why* a rank-$r$
   adapter could compensate for it (`02` §1, Eckart–Young). A clean, CPU-only
   probe of the §4.2 hand-wave — does the error live where the adapter can reach?

Results go in `experiments/log.md` (codec runs, tied to a SHA) and the journal;
this file stays the derivation.
