# 03 — Prompt loss masking: the math of training on the response only

> The "why" half of the prompt-masking change. The decision record lives in
> ADR 0003; this file is the derivation. Cross-refs:
> [`src/train.py`](../../src/train.py) (the bug, §0),
> [`configs/qwen_0.5b_lora.yaml`](../../configs/qwen_0.5b_lora.yaml) (`max_len`,
> the new `mask_prompt` flag), and [`01-lora-derivation.md`](./01-lora-derivation.md)
> for where the gradient this loss produces ultimately lands ($A, B$ only — §6
> there).

This document derives why instruction tuning should compute the loss **only on
the assistant's response tokens**, not on the prompt tokens, and the exact
mechanics of the fix ("Approach A": manual prefix masking). As in `01`, every
shape is shown and every design fork gets both branches.

---

## 0. The bug, stated precisely

`src/train.py` currently does this (lines ~73–85):

```python
ds = ds.map(lambda e: tok(e["text"], truncation=True, max_length=cfg["max_len"]), ...)
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
```

`apply_chat_template` produces one flat string containing **both** the user turn
and the assistant turn. We tokenize that whole string into `input_ids`, and
`DataCollatorForLanguageModeling(mlm=False)` constructs `labels` by *copying*
`input_ids` verbatim (then it only replaces pad positions with `-100`). The
result: the loss is averaged over **every** token in the sequence — the system
preamble, the user's instruction, the context paragraph, AND the response.

That is the standard next-token / continued-pretraining objective applied to a
chat transcript. It is not the instruction-tuning objective we actually want.
§3 quantifies exactly how much of the gradient that misdirects. First we need
the loss defined precisely.

---

## 1. The causal-LM loss, precisely

Fix one tokenized sequence of length $T$:

$$
x = (x_1, x_2, \dots, x_T), \qquad x_t \in \{1, \dots, V\},
$$

where $V$ is the vocabulary size (Qwen2.5: $V = 151{,}936$). The model with
parameters $\theta$ is an autoregressive factorization of the joint:

$$
p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}),
\qquad x_{<t} \equiv (x_1, \dots, x_{t-1}).
$$

The per-sequence training loss is the **mean** negative log-likelihood over
positions:

$$
L(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\!\left(x_t \mid x_{<t}\right).
$$

### Teacher forcing and the label shift — be exact about who shifts

The model is a single function from the input sequence to a logit tensor:

$$
\text{logits} = f_\theta(x) \in \mathbb{R}^{T \times V}.
$$

Row $t$, written $\text{logits}_t \in \mathbb{R}^{V}$, is the model's
prediction of the token that comes **after** position $t$ — i.e. a prediction of
$x_{t+1}$ — conditioned on $x_{\le t}$ (the causal mask guarantees row $t$ only
attends to positions $\le t$). Softmax turns each row into a distribution:

$$
p_\theta(\,\cdot \mid x_{\le t}) = \mathrm{softmax}(\text{logits}_t) \in \Delta^{V-1}.
$$

"Teacher forcing" means: when predicting position $t{+}1$ we feed the model the
*true* $x_{\le t}$, not its own previous samples. So all $T$ predictions can be
computed in **one** forward pass over the real sequence, and each is scored
against the real next token.

Now the part people get wrong about HuggingFace. The user supplies `labels`
**aligned to `input_ids`** — same length $T$, `labels[t]` sits at the same index
as `input_ids[t]`. The one-position shift happens **inside the model's
`forward`**, not in your data pipeline. Concretely the model computes:

$$
\text{loss} = \texttt{CrossEntropy}\big(\underbrace{\text{logits}[:T-1]}_{\text{predict } x_2..x_T}, \;\underbrace{\text{labels}[1:T]}_{\text{the actual } x_2..x_T}\big).
$$

So row $0$ of logits is scored against `labels[1]`, row $1$ against `labels[2]`,
and the last row is dropped (nothing follows it). Shapes through the pipeline:

| object | shape | dtype | who makes it |
|--------|-------|-------|--------------|
| `input_ids` | $\mathbb{Z}^{T}$ | long | tokenizer |
| `labels` (you supply, aligned to input_ids) | $\mathbb{Z}^{T}$ | long | **you** |
| `logits` $= f_\theta(\text{input\_ids})$ | $\mathbb{R}^{T \times V}$ | float | model |
| internal shift | logits$[:T{-}1]$ vs labels$[1:]$ | — | model.forward |
| per-position loss | $\mathbb{R}^{T-1}$ | float | CrossEntropy |

The practical upshot for us: **to mask a token's contribution to the loss, set
its `labels` entry to the sentinel, at the same index as its `input_ids` entry.**
We never touch `input_ids` (the model still needs the full context to predict
later tokens) — we only blank out `labels`. That is the whole trick.

---

## 2. What `-100` does — and the denominator it changes

PyTorch's `nn.CrossEntropyLoss` (and `F.cross_entropy`) takes an `ignore_index`,
defaulting to `-100`. HuggingFace models use exactly that default. With the
shifted targets $y_t \equiv \text{labels}[t{+}1]$ and reduction `"mean"`:

$$
L = \frac{\displaystyle\sum_{t \,:\, y_t \neq -100} \big(-\log p_\theta(y_t \mid x_{\le t})\big)}
         {\displaystyle\sum_{t} \mathbb{1}[\,y_t \neq -100\,]}.
$$

Two distinct effects, and the second is the one people miss:

1. **Numerator — zero loss and zero gradient at ignored positions.** A position
   with $y_t = -100$ contributes nothing to the sum. And because it never enters
   the computation graph, its gradient is exactly zero. This is *not* the same as
   setting the loss term to zero by hand after the fact — `ignore_index` skips
   the term entirely, so there is no spurious gradient path through that logit
   row. (Forward and backward agree: an ignored position is genuinely absent from
   the objective.)

2. **Denominator — the mean is over the COUNT of non-ignored positions.**
   Look at the denominator: it is
   $N_{\text{kept}} = \sum_t \mathbb{1}[y_t \neq -100]$, **not** $T$. Masking
   removes terms from the numerator *and shrinks the denominator to match*. The
   loss is the average over the tokens you actually scored.

### Why masked and unmasked loss are not on the same scale

This denominator point has a sharp consequence for reading the experiment log.
Let $S$ be a set of positions and $L_S = \frac{1}{|S|}\sum_{t\in S}(-\log p_\theta(\cdot))$
the mean NLL over just those positions. The full-sequence (unmasked) loss is
$L_{\text{all}}$ averaged over all $T{-}1$ scored positions; the masked loss is
$L_R$, averaged over only the response positions. These are means over
**different denominators of different per-token difficulties**. There is no
arithmetic that converts one into the other without knowing every per-token NLL.

Concretely: prompt tokens (especially boilerplate chat scaffolding and copied
context) are often *easy* to predict — low NLL — so including them can pull the
unmasked average **down**. A masked run can therefore show a numerically
**higher** loss than the unmasked baseline while being a strictly better
instruction follower, simply because it is now averaging only over the hard part
(the response). **Do not compare the two loss numbers directly.** §"What to
verify empirically" returns to this: the honest comparison is generation
quality, not loss.

---

## 3. The decomposition that proves masking matters

Partition the scored positions of one sequence into the **prompt** set $P$ and
the **response** set $R$ (disjoint, covering all scored positions). Let
$|P|, |R|$ be their sizes and define the group means

$$
L_P = \frac{1}{|P|}\sum_{t\in P}\!\big(-\log p_\theta(y_t\mid x_{\le t})\big),
\qquad
L_R = \frac{1}{|R|}\sum_{t\in R}\!\big(-\log p_\theta(y_t\mid x_{\le t})\big).
$$

Because the full loss is one big mean over $P \cup R$, split the sum:

$$
L_{\text{full}}
= \frac{1}{|P|+|R|}\Big(\sum_{t\in P}(\cdot) + \sum_{t\in R}(\cdot)\Big)
= \frac{|P|}{|P|+|R|}\,L_P \;+\; \frac{|R|}{|P|+|R|}\,L_R.
$$

So $L_{\text{full}}$ is a **convex combination** of $L_P$ and $L_R$ with weights
equal to the token-count fractions. Now the gradient. $\nabla_\theta$ is linear,
and the weights $\tfrac{|P|}{|P|+|R|}, \tfrac{|R|}{|P|+|R|}$ are constants (they
depend only on token counts, not on $\theta$), so the gradient inherits the
*same* convex combination:

$$
\boxed{\;\nabla_\theta L_{\text{full}}
= \frac{|P|}{|P|+|R|}\,\nabla_\theta L_P
\;+\; \frac{|R|}{|P|+|R|}\,\nabla_\theta L_R.\;}
$$

This is the crux. **The optimizer step is a blend of two gradients:** one that
makes the model better at *generating the instruction* ($\nabla_\theta L_P$) and
one that makes it better at *generating the response given the instruction*
($\nabla_\theta L_R$). Each step spends a $\tfrac{|P|}{|P|+|R|}$ fraction of its
"gradient budget" on the former.

In `01-lora-derivation.md` §6 we saw this gradient ultimately lands only on the
adapter factors $A, B$. So concretely: a fixed fraction of every update to $A$
and $B$ is being spent teaching the rank-8 adapter to *reproduce user
instructions* — capacity we already argued is scarce (1.8% per matrix).

### A concrete ratio

Dolly examples carry a `context` field that gets concatenated into the user
turn (`src/train.py` `to_chat`), so prompts are often long. Take an illustrative
example with a 400-token prompt and a 100-token response:

$$
\frac{|P|}{|P|+|R|} = \frac{400}{500} = 0.80,
\qquad
\frac{|R|}{|P|+|R|} = \frac{100}{500} = 0.20.
$$

**80% of the gradient on every step is pushing the model to predict the
prompt** — a distribution we will never sample at inference. Only 20% trains the
thing we actually want. Masking sets the first term to zero:

$$
\nabla_\theta L_{\text{masked}} = \nabla_\theta L_R
\quad\Longleftrightarrow\quad \text{labels}[P] = -100,
$$

so the entire gradient budget goes to the input$\rightarrow$output mapping. With
$r=8$ adapters on a CPU box running few steps, recovering that 80% is not a
rounding error — it is most of the useful signal in the run.

(Caveat for rigor: $\nabla_\theta L_P$ and $\nabla_\theta L_R$ may partially
align — improving response modeling can incidentally improve prompt modeling and
vice versa — so it is not literally "80% wasted compute." But the *objective*
being optimized is wrong by that weight, and there is no guarantee the two
gradients agree; in general they don't. The honest claim is: 80% of the gradient
optimizes a distribution we never query.)

---

## 4. Why masking is the right objective — and when it isn't

### What we actually want at inference

At serving time the interaction is always: the prompt is **given** (the user
types it; we condition on it), and the model **generates** the response. We
sample from $p_\theta(\text{response} \mid \text{prompt})$. We never ask the
model to produce the user's instruction — that text is an input, not an output.

The training objective should match the inference distribution. $L_R$ is exactly
$-\log p_\theta(\text{response}\mid\text{prompt})$ averaged over response tokens
— the conditional we sample from. $L_P$ optimizes
$p_\theta(\text{prompt})$, the marginal over inputs, which is **never queried**.
Spending gradient on $L_P$ is optimizing a distribution off the support of how
the model is used. Masking aligns train-time and test-time objectives — the
same discipline as not leaking the label into the features.

### The honest counterpoint — masking is an empirical choice, not a law

There are real regimes where you *do* train on the prompt:

- **Continued pretraining / domain adaptation.** If the goal is to absorb a
  corpus's style and facts (not to follow instructions), the whole-sequence LM
  loss is the *right* objective — there is no prompt/response split to respect.
- **Very short or templated prompts.** When $|P| \ll |R|$, the wasted fraction
  $\tfrac{|P|}{|P|+|R|}$ is tiny, and the prompt loss can act as a mild
  regularizer or help the model lock onto the template. The whole argument in §3
  is weighted by $|P|/(|P|+|R|)$; when that is small, so is the problem.
- **Prompts with learnable structure.** If prompts themselves contain a pattern
  you *want* the model to internalize (e.g. it must learn to emit a structured
  scaffold before answering), training on them can be deliberate.

Whether masking actually helps **for this model, dataset, and budget** is an
empirical question. That is precisely why the fix adds a `mask_prompt` boolean to
the config rather than hard-coding the behavior: so we can A/B it
(`mask_prompt: true` vs `false`) against the unmasked Run 001 baseline and let
the measurement decide. (I'm deliberately not invoking specific "less is more"
data-efficiency findings here — the mechanism above stands on its own, and I'd
rather measure on our setup than cite a result I can't reproduce on this box.)

---

## 5. Prefix-masking mechanics (Approach A) and its sharp edges

Approach A computes the prompt's token length and blanks those labels manually,
then uses a collator that preserves a `labels` field instead of regenerating it.

### The construction

```text
full_ids   = tokenize( apply_chat_template(messages, add_generation_prompt=False) )
prompt_ids = tokenize( apply_chat_template(user_only, add_generation_prompt=True) )
prompt_len = len(prompt_ids)

labels = list(full_ids)            # aligned to input_ids (§1)
labels[:prompt_len] = [-100] * prompt_len   # blank the prefix
# DataCollatorForSeq2Seq pads input_ids AND labels, padding labels with -100
```

`input_ids = full_ids` (unchanged — the model still reads the whole prompt to
condition on). Only the first `prompt_len` **labels** become `-100`.

### Why `add_generation_prompt=True` on the prompt tokenization

A chat template wraps each turn in role markers. For Qwen2.5 the assistant turn
opens with a header like `<|im_start|>assistant\n`. With
`add_generation_prompt=True`, the prompt tokenization includes that opening
assistant header but **none** of the response content — it is exactly the prefix
the model sees right before it should start generating at inference. Therefore:

- `prompt_len` covers the system preamble + user turn + the assistant header.
- The **first unmasked label** is the first *real* response token.
- We do **not** waste an unmasked position teaching the model to emit
  `<|im_start|>assistant` — that scaffolding is fixed and template-supplied, not
  something the model should "learn" to generate.

If you used `add_generation_prompt=False` for the prompt count, `prompt_len`
would stop before the assistant header, leaving those header tokens *unmasked* —
a small leak of exactly the kind §3 warns about.

### The exact-prefix assumption, and how BPE can break it

The construction silently assumes:

$$
\text{prompt\_ids} = \text{full\_ids}[\,:\text{prompt\_len}\,]
\quad\text{(token-for-token a prefix).}
$$

This is *usually* true but **not guaranteed**, because tokenization is not
compositional across a boundary. BPE/byte-level merges are greedy over the
character stream: a merge can straddle the prompt/response junction. Tokenizing
the prompt **alone** ends the stream there, so the boundary token may merge
differently than when the response characters follow it in the **full** string.
The two tokenizations can then disagree by a token right at the seam — meaning
`full_ids[prompt_len-1]` differs from `prompt_ids[-1]`, or an off-by-one in
`prompt_len`. The damage is bounded (one token at the boundary), but a one-token
shift means either the last prompt token leaks into the loss, or the first
response token gets masked out.

Mitigation: a **unit test** asserts `full_ids[:prompt_len] == prompt_ids` on a
sample of real examples (and ideally adversarial ones whose response starts with
no leading space / mid-word). If it ever fails, switch to the more robust
"tokenize each turn separately and concatenate ids" strategy, where there is no
re-tokenization across the seam to disagree about. (That alternative trades the
clean "tokenize once" code for guaranteed alignment — a fair trade if the test
ever trips.)

### The all-prompt edge case: $0/0 \to$ NaN

`max_len` truncates from the right. If a prompt is long enough that truncation
removes the entire response, then after masking **every** label in the sequence
is `-100`: $|R| = 0$. Look back at the §2 denominator:

$$
L = \frac{\sum_{t:\,y_t\neq-100}(\cdot)}{\sum_t \mathbb{1}[y_t\neq-100]}
  = \frac{0}{0} = \text{NaN}.
$$

The mean over zero kept positions is $0/0$. That NaN does not stay local: the
batch loss for an effective batch is itself a mean (or sum) over examples, so one
NaN example produces a NaN batch loss, a NaN gradient, and — once the optimizer
applies it — NaN-poisoned adapter weights $A, B$. **One bad example silently
destroys the whole run.**

Defense: **filter** any example whose *prompt alone* already fills (or
overflows) `max_len`, i.e. drop it when

$$
\text{prompt\_len} \ge \text{max\_len}
\quad(\text{equivalently, } |R| = 0 \text{ after truncation}).
$$

This guarantees $|R| \ge 1$ for every surviving example, so the denominator is
always positive and the loss is always finite. The cost is a (logged) drop in
dataset size — quantified next.

---

## What to verify empirically

The math says masking optimizes the right objective; whether it *helps here* is
measured, not assumed.

1. **Generation quality vs. the unmasked Run 001 baseline — judged by
   generation, not loss.** As §2 proved, the masked loss and the unmasked loss
   have different denominators over different token sets, so a lower or higher
   loss number tells you nothing comparable. The honest test is behavioral: run
   both adapters (`mask_prompt: true` and the Run 001 `mask_prompt:
   false`/whole-sequence baseline) on the **same held-out infer prompts** and
   compare the generated **responses** — do they follow the instruction, stay on
   task, avoid regurgitating the prompt? That measures
   $p_\theta(\text{response}\mid\text{prompt})$, which is the only distribution
   we ever sample. If masking works as derived, expect the masked adapter to
   produce tighter, more instruction-faithful responses even if its training
   loss reads higher.

2. **Dropped-example count vs. `max_len`.** The §5 filter removes every example
   with $\text{prompt\_len} \ge \text{max\_len}$. Sweep `max_len`
   $\in \{256, 512, 1024\}$ and log how many of the $n_{\text{train}}$ examples
   survive. Dolly's long-`context` rows make this non-trivial: at small
   `max_len` you may discard a large fraction of the data (changing *what* the
   adapter sees, a confound for experiment 1), while large `max_len` keeps more
   examples but costs CPU time per step. Record the survivor count alongside each
   run's SHA so the loss/generation results are interpreted against the dataset
   they actually trained on.
