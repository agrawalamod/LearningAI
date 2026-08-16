# Chapter 2 — How Training Works

Chapter 1 walked the **forward pass**: text in, next-token probability out. But that
only works because the weight matrices (`W_E`, `Wq/Wk/Wv/Wo`, FFN `W_1/W_2`, `W_U`, …)
already hold good values. This chapter is about **how those values get there** —
the training process that turns a randomly-initialized stack into GPT-3.

Notation and dimensions are the same as Chapter 1 (GPT-3 175B: `V = 50,257`,
`embed_dim = 12,288`, `n_layers = 96`, etc. — these are the descriptive names; the
textbook symbols are `d_model`/`d_k`/`d_ff`, see Chapter 1's notation note).

---

## 2.1 — The one and only objective: predict the next token

GPT-3 is trained on a single, breathtakingly simple task: **given a sequence of
tokens, predict the next one.** That's it. Everything the model "knows" — grammar,
facts, reasoning, translation — is a side effect of getting very good at this one game,
played over trillions of tokens.

This is called **self-supervised** learning: the data labels itself. You don't need
humans to annotate anything. Take any text, and for every position the "correct answer"
is simply the token that actually came next.

```
Text:    "The cat sat on the mat"
Tokens:   The   cat   sat   on   the   mat

Training turns this into 5 prediction problems at once:
  given "The"                      → target: "cat"
  given "The cat"                  → target: "sat"
  given "The cat sat"              → target: "on"
  given "The cat sat on"           → target: "the"
  given "The cat sat on the"       → target: "mat"
```

One sentence = many training examples. A whole book = millions. The internet =
trillions. The model never runs out of free supervision.

---

## 2.2 — Teacher forcing: all positions trained at once

Here's where Chapter 1's §1.4 (and the §1.6 parallelism note) pays off. You might
think you'd train those 5 problems above one at a time. You don't. During training the
**whole sequence is fed in at once**, and the causal mask (Ch.1 §1.4 step 3c) ensures
each position only sees earlier tokens. The LM head (`W_U`) is applied to all columns
of the final residual stream simultaneously, producing a **`50,257 × 6` logit matrix**
— one full 50,257-dim prediction vector per position, all in one forward pass:

```
input tokens:    The      cat      sat      on       the      mat
                  │        │        │        │        │        │
                  ▼        ▼        ▼        ▼        ▼        ▼
        [ 96 blocks — shape stays 12,288 × 6 throughout ]
                  │        │        │        │        │        │
                  ▼        ▼        ▼        ▼        ▼        ▼
      LM head W_U (50,257 × 12,288) applied to all 6 columns at once
                  │        │        │        │        │        │
                  ▼        ▼        ▼        ▼        ▼        ▼
logits:       50,257×1  50,257×1 50,257×1 50,257×1 50,257×1 50,257×1
              col 0     col 1    col 2    col 3    col 4    col 5
                                                              (unused)
predicts:     "cat"?   "sat"?    "on"?   "the"?   "mat"?    "???"
targets:       cat      sat       on      the      mat
losses:        L0       L1        L2      L3       L4       (averaged → batch_loss)
```

The shape throughout the 96 blocks is `12,288 × 6` (the residual stream). After the
final LayerNorm, this is where inference and training diverge:

**Inference:** `W_U` is applied to the **last column only** (`H_norm[:, seq_len−1]`,
`12,288 × 1`) → `50,257 × 1` logits → softmax → one next token. The other `seq_len−1`
columns are never multiplied with `W_U`. They were needed earlier only so attention had
context to look at.

**Training:** `W_U` is applied to **all `seq_len` columns** (`H_norm`, `12,288 × seq_len`)
→ `50,257 × seq_len` logit matrix → `seq_len` softmax + cross-entropy losses computed
simultaneously → averaged → one `batch_loss` → one backward pass → one weight update.

**This is the core training parallelism.** One forward pass produces `seq_len` losses
at once — the equivalent of `seq_len` separate inference passes, computed simultaneously.
A sequence of 2,048 tokens gives 2,047 training samples from a single forward pass.

This is **teacher forcing**: the model is fed the *real* previous tokens (the "teacher"
supplies the ground truth), not its own guesses. So even if it predicted "dog" after
"The", training still feeds it the real "cat" for the next position. This keeps all
positions independent and lets them produce predictions simultaneously.

> At inference there is no teacher — the model feeds its own outputs back in one token
> at a time, and only the last column's logits are used each step. Training is parallel
> across all positions; generation is serial, one column at a time.

---

## 2.3 — Turning predictions into a loss: cross-entropy

The forward pass ends (Ch.1 step 8) with a probability distribution over all 50,257
vocab tokens at each position. Training needs a single number measuring "how wrong was
that?" — the **loss**. The standard choice is **cross-entropy loss**.

For one position, cross-entropy only cares about the probability the model assigned to
the **correct** token:

```
loss_position = − log( P(correct token) )
```

Worked example — model predicting the token after "The cat sat on the":

```
target = "mat"

Case A — model is confident & right:
   P("mat") = 0.90   →  loss = −log(0.90) = 0.105   (small loss, good)

Case B — model is unsure:
   P("mat") = 0.10   →  loss = −log(0.10) = 2.303   (big loss, bad)

Case C — model is confidently wrong:
   P("mat") = 0.01   →  loss = −log(0.01) = 4.605   (huge loss, very bad)
```

The shape of `−log(p)`:
- Right and confident (`p → 1`) → loss → 0.
- Wrong and confident (`p → 0`) → loss → ∞.

So the loss **punishes confident mistakes brutally** and rewards confident correct
answers. The total loss for the batch is the **average** of `loss_position` over every
position in every sequence in the batch.

```
batch_loss = mean over all positions of  −log P(correct token at that position)
```

That single scalar is what training tries to minimize.

---

## 2.4 — Backpropagation: assigning blame

We have one number (`batch_loss`) and ~175 billion parameters. Training needs to know,
for each parameter: *"if I nudge you slightly, does the loss go up or down, and by how
much?"* That quantity is the **gradient** — the partial derivative of the loss with
respect to that parameter.

**Backpropagation** computes all 175B gradients efficiently in one backward sweep. The
idea is the chain rule from calculus: the forward pass is a long chain of differentiable
operations, so you can propagate "blame" backward from the loss through each operation
to its inputs and weights.

```
forward:   weights ─► ... ─► block 95 ─► LayerNorm ─► W_U ─► softmax ─► loss
backward:  gradients ◄─ ... ◄─ block 95 ◄─ LayerNorm ◄─ W_U ◄──────────┘
           (chain rule pushes ∂loss/∂x backward through every op)
```

Key facts:
- Backprop flows through attention like any other layer. Attention isn't special — the
  dot products, softmax, masking, and weighted sums are all differentiable, so gradients
  pass straight through them.
- **Only the weights are learned.** The gradient tells each weight matrix
  (`Wq/Wk/Wv/Wo`, FFN, embeddings, …) which direction reduces loss. The parameter-free
  ops (softmax, mask, `/√head_dim`) get gradients passing *through* them but have nothing
  to update.
- **The residual stream's `+` (Ch.1 §1.5) is a gradient highway.** Because the derivative
  of `e + attn_out` w.r.t. `e` is 1, gradients flow back through 96 layers without
  vanishing. This is *why* deep stacks are trainable at all.
- **No supervision on attention itself.** Nobody labels "token 5 should attend to token
  2." The only signal is the next-token loss; backprop *discovers* that certain `Wq/Wk`
  configs lower loss, and the attention patterns emerge implicitly. (This is also why
  attention is hard to interpret.)

**The full set of trained parameters** (everything backprop updates):
- Embedding / LM-head matrix (tied in some models — one matrix gets gradients from both
  the input path and the output/logit path)
- Per attention block: `Wq, Wk, Wv, Wo` + norm gains
- Per FFN block: `W_1, W_2` + biases
- All LayerNorm gains

Everything *else* in the forward pass (the `q·k` dot product, `/√head_dim`, causal mask,
softmax, weighted sum) is **parameter-free** — differentiable so gradients pass through,
but with nothing of its own to learn.

The output of backprop: a gradient for every parameter, same shape as the parameter.

---

## 2.5 — The optimizer: turning gradients into weight updates

Gradients say *which way* to move each weight. The **optimizer** decides *how far*. The
simplest rule is gradient descent:

```
new_weight = old_weight − learning_rate × gradient
```

The `learning_rate` (a small number like 0.0001) controls step size. Subtract because we
want to go *downhill* on the loss.

GPT-3 uses **AdamW**, a smarter optimizer that improves plain gradient descent in three
ways:

1. **Momentum (1st moment).** Keeps a running average of recent gradients, so updates
   build up speed in consistent directions and don't zig-zag. Like a ball rolling
   downhill rather than teleporting each step.
2. **Adaptive per-parameter scaling (2nd moment).** Tracks how large each parameter's
   gradients have been and scales the step accordingly — parameters with consistently
   small gradients get bigger relative steps, and vice versa. Each weight effectively
   gets its own tuned learning rate.
3. **Weight decay (the "W").** Gently pulls weights toward zero each step, discouraging
   any single weight from growing huge — a regularizer that improves generalization.

AdamW therefore stores **two extra numbers per parameter** (the momentum and variance
estimates). For 175B params that's ~350B extra values — a big reason training needs far
more memory than inference.

---

## 2.6 — The training loop

Put it together. One **step** of training:

```
1. Sample a batch of token sequences from the dataset.
2. FORWARD: run the 96-block stack → logits at every position (Ch.1 steps 1–7).
3. LOSS:    cross-entropy vs the real next tokens, averaged → batch_loss (§2.3).
4. BACKWARD: backprop → a gradient for every one of the 175B params (§2.4).
5. UPDATE:  AdamW nudges every param a little to reduce loss (§2.5).
6. Repeat — hundreds of thousands of times.
```

```
   ┌─────────────────────────────────────────────┐
   │  sample batch                                 │
   │      │                                        │
   │      ▼                                        │
   │  forward pass ──► logits ──► cross-entropy ──►│ loss
   │      ▲                                    │   │
   │      │                                    ▼   │
   │  AdamW update ◄── gradients ◄── backprop ◄────┘
   │      │                                        │
   └──────┴────────── repeat ~300K+ steps ─────────┘
```

Over many steps, `batch_loss` trends downward: the model's `P(correct token)` climbs
from random (`1/50,257 ≈ 0.00002`, loss ≈ 10.8) toward genuinely good predictions
(loss ≈ 1.9 for GPT-3 scale). That slow descent *is* learning.

### How one scalar loss becomes 175B individual gradients

This is the part step 4 glosses over. The key is that backprop never computes 175B
gradients independently — it computes them all **in one backward pass** by reusing
intermediate results via the chain rule. Here is the full chain walked with actual
numbers using GPT-3's real dimensions.

#### The chain of operations (last few steps of the forward pass)

```
h_last          W_U              logits           softmax    loss
(12,288 × 1) → (50,257×12,288) → (50,257 × 1) →  P        → −log(P[target])
```

**Concrete values** (showing 3 of 50,257 logits; target = token 42):

```
W_U[42, :] = [ 3.0,  1.0,  ...,  0.5 ]   ← row 42 of W_U, 12,288 values
              dim0   dim1  ...   dim12287

h_last     = [ 2.0,  4.0,  ...,  1.5 ]   ← final residual vector, 12,288 values
              dim0   dim1  ...   dim12287

logit[42]  = W_U[42,0]·h_last[0]  +  W_U[42,1]·h_last[1]  +  ...  +  W_U[42,12287]·h_last[12287]
           = 3.0×2.0               +  1.0×4.0               +  ...  +  0.5×1.5
           = 6.0                   +  4.0                   +  ...  +  0.75
           = 10.0                                                       (showing only 3 of 12,288 terms)

logit[43]  = 9.0    (from its own W_U row · h_last)
logit[44]  = 8.0
  ...                                                                   (50,257 logits total)
```

**Softmax → probabilities:**

```
P[42] = e^10.0 / (e^10.0 + e^9.0 + e^8.0 + ... )   ← 50,257 terms in denominator
      = 22026  / (22026 + 8103 + 2981 + ... )
      ≈ 0.665

P[43] ≈ 0.245
P[44] ≈ 0.090
  ...                                               (50,257 probabilities, summing to 1)
```

**Loss (target = token 42):**

```
loss = −log(P[42]) = −log(0.665) = 0.408
```

#### Hop 1 — ∂loss/∂P[42] (why it's −1/P)

The loss is `−log(P)`. The calculus rule for log is `d/dP[log(P)] = 1/P`, so:

```
∂loss/∂P[42] = d/dP[−log(P)] = −1/P = −1/0.665 = −1.504
```

Verify numerically — nudge P up by 0.001:

```
−log(0.665) = 0.4082
−log(0.666) = 0.4067
change / nudge = (0.4067 − 0.4082) / 0.001 = −1.50  ≈  −1.504  ✓
```

The gradient is **negative**: increasing P[42] decreases the loss — correct, the model
should be more confident about the right token.

The magnitude `1/P` is large when P is small (model was very wrong, needs strong signal)
and small when P is large (model was already right, only a nudge needed):

```
P = 0.90  →  −1/P = −1.11   (small signal — already confident & right)
P = 0.50  →  −1/P = −2.00   (medium)
P = 0.10  →  −1/P = −10.0   (large signal — model was very wrong)
```

#### Hop 2 — ∂P[42]/∂logit[42] (softmax local derivative)

The softmax derivative for the target class is (quotient rule on `e^x / S`, the `e^x`
cancels to leave `P×(1−P)`):

```
∂P[i]/∂logit[i] = P[i] × (1 − P[i])
                = 0.665 × 0.335
                = 0.223
```

#### Combining hops 1 and 2 — ∂loss/∂logit[42]

```
∂loss/∂logit[42] = (∂loss/∂P[42]) × (∂P[42]/∂logit[42])
                 = (−1.504)        × (0.223)
                 = −0.335

This simplifies to:  P[42] − 1  =  0.665 − 1.0  =  −0.335  ✓
                     (always true for the target class)
```

For non-target classes: `∂loss/∂logit[i] = P[i] − 0 = P[i]`

```
∂loss/∂logit[43] = +0.245
∂loss/∂logit[44] = +0.090
  ...                         (50,257 values total, target gets P−1, rest get P)
```

#### Hop 3 — ∂logit[42]/∂W_U[42,1] (local derivative of the matmul)

```
logit[42] = W_U[42,0]·h_last[0]  +  W_U[42,1]·h_last[1]  +  ...  +  W_U[42,12287]·h_last[12287]

∂logit[42]/∂W_U[42,1] = h_last[1] = 4.0
```

The local derivative is simply the activation the weight was multiplied by. The other
terms in the sum drop out because they don't contain `W_U[42,1]`.

#### Chain rule — final gradient for W_U[42,1]

```
∂loss/∂W_U[42,1] = (∂loss/∂logit[42]) × (∂logit[42]/∂W_U[42,1])
                 = (−0.335)            × (4.0)
                 = −1.34
```

**Meaning:** increasing `W_U[42,1]` by 0.001 decreases the loss by ~0.00134. Since the
gradient is negative, the optimizer increases this weight:

```
W_U[42,1]_new = W_U[42,1] − lr × gradient
              = 1.0         − 0.1 × (−1.34)
              = 1.0         + 0.134
              = 1.134
```

#### Why every weight gets a unique gradient

Now compute the gradient for the neighbouring weight `W_U[42,0]` — same row, one column
to the left:

```
∂logit[42]/∂W_U[42,0] = h_last[0] = 2.0    ← different activation

∂loss/∂W_U[42,0] = (−0.335) × 2.0 = −0.670
```

Same `∂loss/∂logit[42] = −0.335`, but `h_last[0] = 2.0` instead of `h_last[1] = 4.0`
→ gradient is **−0.670 vs −1.34** — different. Every weight in `W_U` has the same
first factor (`P[i] − target[i]` for its row) but a unique second factor (the specific
`h_last[j]` value for its column). That's why 175B weights get 175B unique gradients.

For all 12,288 weights in row 42 of `W_U`:

```
∂loss/∂W_U[42, 0]     = −0.335 × h_last[0]     = −0.335 × 2.0  = −0.670
∂loss/∂W_U[42, 1]     = −0.335 × h_last[1]     = −0.335 × 4.0  = −1.340
  ...                                                               ...
∂loss/∂W_U[42, 12287] = −0.335 × h_last[12287] = −0.335 × 1.5  = −0.503
```

Then backprop continues **leftward**: the gradient flows back through `W_U` into
`h_last`, then into block 95, block 94, …, all the way to the embedding matrix.

#### The general pattern for any weight W[i,j]

```
gradient = (how wrong was logit[i]) × (what activation was at h[j])
         = (P[i] − target[i])       × h[j]
```

One backward sweep reuses `(P[i] − target[i])` across the whole row and `h[j]` across
the whole column — no weight is computed from scratch independently.

**The memory cost.** The forward pass must **save every intermediate activation**
(`h_last`, every `q/k/v`, every FFN hidden state) because the backward pass needs them
as the second factor in the chain rule (e.g. `h_last[j]` above). For GPT-3 this
activation cache is large — the main reason training uses far more memory than inference,
which discards activations immediately.

**Summary: one loss → 175B gradients by reuse, not repetition.** Backprop makes one
backward sweep; at each layer it multiplies the incoming gradient by the local
derivative (which came from the cached activations). Efficient because intermediate
results are computed once and reused across all parameters in a layer.

---

## 2.7 — Vector vs. matrix, one more time (the crux)

This is the single most important distinction in training, and it's worth restating from
Ch.1 §1.5 in the training context:

| | What changes | When | Persists? |
|---|---|---|---|
| **Forward pass** | activations / the residual *vectors* (`e`, `attn_out`, …) | every forward pass, per token | **No** — recomputed and discarded each run |
| **Backprop + update** | the weight *matrices* (`Wq`, FFN, `W_E`, …) | once per optimizer step | **Yes** — this is the learning |

So when we said in Ch.1 "`delta e` updates the embedding," that's the **forward-pass
activation** moving through the residual stream — temporary, thrown away after each run.
**Training** is the separate backprop+AdamW step that nudges the **weight matrices**.
The vectors change every single run; the matrices change only when you train.

**Embedding matrix nuance — sparse gradients:** in any given batch, only the `W_E` rows
for tokens that actually appeared get gradient updates. Common tokens ("the") update
almost every step; rare tokens update slowly.

---

## 2.8 — Scale: what GPT-3 training actually took

The loop in §2.6 is conceptually simple; the engineering is not. GPT-3 scale, roughly:

| Aspect | Approximate figure |
|---|---|
| Training data | ~300 billion tokens (filtered Common Crawl, WebText2, books, Wikipedia) |
| Parameters | 175 billion |
| Context length per sequence | 2,048 tokens |
| Optimizer state (AdamW) | 2 extra values/param → ~350B extra numbers |
| Compute | ~3,000+ petaflop/s-days |
| Hardware | thousands of V100 GPUs, weeks of wall-clock time |
| Cost | commonly estimated in the millions of USD |

Why so heavy:
- **Memory.** Weights + gradients + AdamW's two moments + activations (saved for
  backprop) must all fit. This is why training needs many times the memory of inference,
  and why models are sharded across many GPUs.
- **Batching.** Effective batches are huge (hundreds of thousands of tokens) for stable
  gradients, often built up via **gradient accumulation** (sum gradients over several
  mini-batches before one update) when they don't fit at once.
- **Mixed precision.** Math runs in 16-bit (FP16/BF16) for speed/memory, with a 32-bit
  master copy of weights for stability.
- **Parallelism.** Data parallelism (same model, different data per GPU), plus
  tensor/pipeline parallelism (one model split across GPUs) because 175B params don't fit
  on a single device.

---

## 2.9 — Learning-rate schedule: warmup then decay

The learning rate isn't constant. GPT-3 uses a schedule:

```
lr
 │        ╭──────╮
 │       ╱        ╲────────╴ cosine decay ╴────────╮
 │      ╱                                           ╲
 │     ╱ warmup                                      ╲___
 └────┴───────────────────────────────────────────────── steps
      (linear ramp     (long, slow cosine decay toward ~0)
       up for the
       first steps)
```

- **Warmup**: start near 0 and ramp up over the first few thousand steps. Early on the
  weights are random and big steps would destabilize training.
- **Cosine decay**: after the peak, slowly decrease the learning rate toward near zero
  over the rest of training, so updates get gentler as the model fine-tunes its weights.

---

## 2.10 — After pretraining: where alignment fits

Everything above is **pretraining** — the giant next-token phase that produces the
"base model." For GPT-3 (2020), that's essentially the whole story; the base model is
steered only by **few-shot prompting** (showing examples in the prompt), as Ch.1's table
notes.

Later models add stages *on top* of pretraining (worth knowing as a roadmap, expanded in
later chapters):

1. **Pretraining** — next-token prediction on huge web-scale text. Learns language and
   world knowledge. (This chapter.)
2. **Supervised fine-tuning (SFT)** — continue training on smaller, curated
   instruction→response pairs so the model follows instructions instead of just
   continuing text.
3. **Preference optimization (RLHF / DPO)** — train on human preference comparisons
   ("response A is better than B") to make outputs helpful, harmless, and honest.

GPT-3 itself stops at stage 1. The Gemma 4 line (Chapter 3) is instruction-tuned and
preference-optimized — stages 2 and 3 — which is why it follows instructions out of the
box while raw GPT-3 needs few-shot prompting.

---

## 2.11 — The one-paragraph summary

GPT-3 learns by playing "guess the next token" over ~300B tokens of text. Each step:
feed a batch through the 96-block stack — blocks are **sequential** (each feeds the
next), but within each block all token positions are processed **in parallel** (teacher
forcing + causal mask eliminate the need to run one position at a time), and all
sequences in the batch run in parallel — measure how wrong the predictions are with
cross-entropy loss, run backpropagation to compute how every one of the 175B weights
affected that loss, and let the AdamW optimizer nudge each weight slightly downhill.
Repeat hundreds of thousands of times. The activations (vectors) are recomputed and
discarded every run; the weights (matrices) are what slowly improve. Nobody ever tells
the model how to attend or what to store — grammar, facts, and reasoning all emerge
implicitly as gradient descent discovers whatever configuration minimizes next-token
surprise.
