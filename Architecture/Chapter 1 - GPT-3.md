# Chapter 1 — GPT-3 (2020): The Baseline Transformer

GPT-3 is the "classic" large transformer — the reference point everything else
improves on. This chapter builds from the architecture table, through the
self-attention mechanism, and then walks a full end-to-end forward pass with a
worked example so every number can be referenced directly.

---

## 1.1 — The architecture at a glance

| Trait | GPT-3 |
|---|---|
| Structure | Dense decoder-only stack, 175B params (all active every token) |
| Layers / heads | 96 layers, 96 heads, `embed_dim = 12,288`, `head_dim = 128` |
| Positional info | Learned **absolute** positional embeddings |
| Attention | Mostly full attention (+ alternating banded-sparse layers) |
| Normalization | Standard LayerNorm |
| Context window | ~2,048 tokens |
| Modality | Text only |
| Alignment | Base model; steered only via few-shot prompting |

### Reference dimensions (GPT-3 175B)

| Symbol | Meaning (what it actually is) | Value |
|---|---|---|
| `V` | vocab size — how many distinct tokens exist | 50,257 |
| `embed_dim` | width of each token's vector / the residual stream | 12,288 |
| `n_layers` | number of stacked decoder blocks (depth) | 96 |
| `n_heads` | attention heads per block (parallel views) | 96 |
| `head_dim` | width of one head's q/k/v slice | 128 (96 × 128 = 12,288 ✓) |
| `n_ctx` | max context — most tokens the model can hold | 2,048 |
| `ffn_dim` | inner width of the feed-forward sublayer | 49,152 (4 × embed_dim) |
| `seq_len` | number of tokens in the current input (≤ `n_ctx`) | varies per input |

> **Notation note — these are renamed for clarity.** The textbook / "Attention Is All
> You Need" symbols are `d_model` (= `embed_dim`), `d_k` (= `head_dim`), and `d_ff`
> (= `ffn_dim`). This doc uses the descriptive names everywhere; if you read papers,
> mentally map `embed_dim → d_model`, `head_dim → d_k`, `ffn_dim → d_ff`.

> **Important — `96 layers` and `96 heads` is a coincidence.** These are two
> independent knobs (see §1.7). The only real constraint is
> `n_heads × head_dim = embed_dim`. Layer count is unrelated.

---

## 1.2 — Context window: is the whole 2,048 always fed in?

**No.** The 2,048 tokens is the *maximum capacity* of the context window, not a
fixed amount that always gets fed in.

- The 2,048 is a **ceiling, not a floor**. The window holds *up to* 2,048 tokens,
  shared between input (prompt) and output (completion):
  `prompt_tokens + completion_tokens ≤ 2048`.
- If your query is 50 tokens, only 50 tokens go in. You don't pad it out to 2,048.
  The unused space is just headroom.
- **Per-request, not cumulative.** Each call is stateless; GPT-3 has no memory
  between calls. "Conversation history" means *you* resend prior turns, and all of
  that counts against the 2,048 budget.
- **Reserve room for the answer.** A 2,000-token prompt leaves only 48 tokens for the
  response. `prompt + max_tokens` must fit under the limit.
- **Cost** is for tokens actually processed (prompt + completion), not the full 2,048.
- **Model variation.** The original GPT-3 models were 2,048. Later variants
  (`text-davinci-003`, GPT-3.5) extended to 4,096; newer models go much higher.

---

## 1.3 — Self-attention recap (one head, then combining heads)

Steps 1–7 describe what happens **inside a single head**. All 96 heads run these same
steps independently and in parallel; steps 8–10 then combine them. (Single-head shapes
shown; the full multi-head matrix dimensions are in the note at the end of this section
and in §1.4 Step 3.)

```
steps 1–3   build q/k/v for each token
steps 4–7 mix information *between* tokens (that's the attention)
Steps 8–10 act on each token independently — no cross-token interaction
```


So a sequence of `seq_len` tokens first produces `seq_len` queries, keys, and values
(each `128 × 1`):

```
token 0 → q_0, k_0, v_0
token 1 → q_1, k_1, v_1
  ...        ...
token (seq_len−1) → q_{n−1}, k_{n−1}, v_{n−1}
```

`e` is a column of the sequence matrix `E` (shape `embed_dim × seq_len`), so each `e` is
`12,288 × 1`. The steps:

1. `q = Wq · e`   `(128 × 12,288) · (12,288 × 1) → q is 128 × 1`  (query)
2. `k = Wk · e`   `(128 × 12,288) · (12,288 × 1) → k is 128 × 1`  (key)
3. `v = Wv · e`   `(128 × 12,288) · (12,288 × 1) → v is 128 × 1`  (value)

   *(After steps 1–3 you have `seq_len` of each; stack them as columns into `Q`, `K`, `V`,
   each `128 × seq_len`. Steps 4–6 use these.)*
4. Score `= (q_i · k_j) / √head_dim` for **each token pair** `(i, j)` →
   matrix `M`. A "pair" = the query from token `i` against the key from token `j`;
   `M[i,j]` = *how much token i should attend to token j*.

   **The shape of M differs between training and inference — this is important:**

   - **Training / prefill**: all `seq_len` tokens are present at once, so all queries
     run simultaneously. M is the full `seq_len × seq_len` matrix — every token queries
     every past token at the same time.
   - **Inference (decoding)**: only one new token is being generated. Only that token's
     query is needed. M collapses to **one row** (`1 × seq_len`) — the new token's query
     scored against all past keys. Past tokens' Q vectors are never used here.

   ```
   TRAINING M (seq_len × seq_len):    INFERENCE M at step N (1 × N, one row only):

        k_0  k_1  k_2  k_3             k_0  k_1  k_2  ... k_{N-1}  k_N
   q_0 [  ✓    ✗    ✗    ✗ ]    q_N [ qN·k0 qN·k1 qN·k2 ... qN·k_{N-1} qN·kN ]
   q_1 [  ✓    ✓    ✗    ✗ ]
   q_2 [  ✓    ✓    ✓    ✗ ]    only this one row computed — q_0..q_{N-1} not needed
   q_3 [  ✓    ✓    ✓    ✓ ]
   ```

   This is why the KV cache only stores K and V — past Q vectors are never used during
   decoding. See §1.8 for the full KV cache explanation.
5. **Causal mask**: a token may only attend to itself + earlier tokens (M is
   lower-triangular before softmax).

   > **Training/prefill**: the mask is necessary. All `seq_len` tokens are present
   > simultaneously, so without it "The" could attend to "mat" (which comes later in
   > the sequence). The upper triangle is forced to `-∞` so those scores become zero
   > after softmax.
   >
   > **Inference (decoding)**: the mask is redundant. Only k_0..k_N exist in the cache
   > — future tokens haven't been generated yet, so their K vectors simply don't exist.
   > The single row `q_N · k_j` naturally only covers past positions; there is nothing
   > future to accidentally attend to. In practice most implementations still apply the
   > mask during decoding (same code path, negligible cost) but it is a no-op.
6. Softmax each row of `M` → the **attention-weight matrix `M'`**
   (`seq_len × seq_len`). Each row sums to 1.
7. **Per-head output — weighted sum of the value vectors.** For each query
   token, take its row of `M'` (`seq_len` scalar weights `w_1, w_2, …`) and blend the
   `seq_len` value vectors (`v_1, v_2, …`, each `128 × 1`). For one query token:

   ```
   out = w_1·v_1 + w_2·v_2 + ... + w_seq_len·v_seq_len
       = (128×1) + (128×1) + ... + (128×1)   ← seq_len terms, each 128×1
       = 128 × 1                              ← summing 128×1 vectors stays 128×1
   ```
   Doing this for all `seq_len` query tokens and placing the results side by side gives
   this single head's full output: **`128 × seq_len`**.
8. **Concatenate the 96 heads.** Stack the 96 per-head outputs from step 7
   (each `128 × seq_len`) on top of each other → **`12,288 × seq_len`** (`96 × 128 =
   12,288`). Equivalently, for each token its 96 head vectors stack into one `12,288 × 1`.
9. **Output projection `Wo`** (`12,288 × 12,288`) applied to the whole
   matrix → **`12,288 × seq_len`**. Call the result `delta e`.
10. **Residual add:** `e' = e + delta e`, stays **`12,288 × seq_len`** —
    **same shape as the input `E`** (add, don't replace; §1.5).

Steps 1–3 are why it's called a **down-projection**: each `W` matrix is `128 × 12,288`,
so multiplying the `12,288`-dim token vector produces a smaller `128`-dim vector
(`embed_dim → head_dim`). The model works in the narrow 128-dim per-head space for the
attention math, then `Wo` brings the result back up to the full `12,288`-dim residual
stream (step 9).

Q, K, V are the three learned projections. Many heads run in parallel, then an
output projection `Wo` maps the concatenated heads back to `embed_dim`.

> **Per-head vs. full matrix.** The `128 × 12,288` above is **one head's** slice of
> `Wq` (this section shows a single head). Across all 96 heads, each of `Wq`, `Wk`, `Wv`
> is really `96 × 128 × 12,288` (3D) ≡ `12,288 × 12,288` (2D stacked), because
> `96 × 128 = 12,288`. So per block there are three full `12,288 × 12,288` projection
> matrices (`Wq`, `Wk`, `Wv`) plus `Wo` (`12,288 × 12,288`). Full shapes, counts, and
> the per-block parameter total are in §1.4 Step 3 ("What 'each head has its own
> projections' really means").

### The key limitation: position

Attention by itself is **order-blind** — shuffle the tokens and you get the same
result shuffled. Position must be injected. GPT-3 does it once at the very bottom:

```
e_i = token_embedding[word_i] + position_embedding[i]
```

`position_embedding` is a learned lookup table (`2048 × 12,288`), one vector per slot.
Because it's a fixed-size table, GPT-3 **can't go past slot 2,047** — there's no
learned vector for position 5,000. This is the wall that later models break.

---

## 1.4 — End-to-end forward pass: "Who was the richest king?"

This walks the full pipeline using GPT-3 175B's actual dimensions, so every number
lines up with the §1.1 tables. **Dots (`...`) mean "more values exist"** — the count
is always stated.

Convention (matches §1.3): the sequence is a matrix `E` of shape
`embed_dim × seq_len` — **each column is one token's vector**.

### Step 0 — Tokenize

GPT-3's BPE turns the string into integer IDs. Plausible tokenization:

```
"Who"      → 8241
" was"     → 373
" the"     → 262
" richest" → 16539      (note: real BPE may split this into " rich"+"est";
" king"    → 5822        using 6 clean tokens here for index sanity)
"?"        → 30
```

So `seq_len = 6`, positions `0..5`. Everything below is a 6-column pipeline.

### Step 1 — Token embedding lookup

The token embedding matrix `W_E` has shape `V × embed_dim = 50,257 × 12,288`. Each
token ID picks **one row** (a 12,288-dim vector).

Token "Who" (ID 8241) → row 8241:

```
tok_emb("Who") = [ 0.013, -0.041,  0.207, ... , -0.008 ]
                   └───────── 12,288 values total ─────────┘
                   (showing 3 of 12,288, the rest hidden as ...)
```

Do this for all 6 tokens → six 12,288-dim vectors.

### Step 2 — Add absolute positional embedding

GPT-3 uses a **learned** position table `W_P` of shape
`n_ctx × embed_dim = 2,048 × 12,288`. Position `i` picks row `i`. The input vector for
each token is the **sum**:

```
e_i = tok_emb(word_i) + pos_emb(i)
```

For "Who" at position 0:

```
tok_emb("Who") = [ 0.013, -0.041,  0.207, ... , -0.008 ]   (12,288)
pos_emb(0)     = [ 0.002,  0.015, -0.030, ... ,  0.011 ]   (12,288)
                ───────────────────────────────────────────
e_0            = [ 0.015, -0.026,  0.177, ... ,  0.003 ]   (12,288)
```

Stack all six `e_i` as columns → the input sequence matrix:

```
        Who    was    the   richest  king    ?
      ┌                                         ┐
      │ 0.015  0.022 -0.07   0.131  -0.044  0.06│  ← dim 0
E  =  │-0.026  0.310  0.21  -0.052   0.180  0.01│  ← dim 1
      │ 0.177 -0.090  0.04   0.260  -0.011 -0.12│  ← dim 2
      │  ...    ...    ...    ...     ...    ...│  ← dims 3 … 12,286
      │ 0.003  0.044 -0.02   0.019   0.077  0.05│  ← dim 12,287
      └                                         ┘
      shape = 12,288 rows × 6 columns
      (showing 4 of 12,288 rows, all 6 columns)
```

This `E` is what enters block 0.

### Step 3 — FIRST attention (decoder block 0, one head shown)

Each of the 96 heads has its own learned projections, each of shape
`head_dim × embed_dim = 128 × 12,288`. Take head `h=0`.

#### What "each head has its own projections" really means (shapes + counts)

This is worth pinning down precisely, because it's easy to miscount.

**There are only 4 *types* of learned matrix in an attention block** — `Wq`, `Wk`,
`Wv`, `Wo`. No fifth type. The first three come in **96 copies** (one per head); `Wo`
is a **single** shared matrix that merges the heads back together (see 3f).

```
        ┌──────── per-head (×96) ────────┐   ┌─ shared (×1) ─┐
input ─►│ Wq⁽ʰ⁾  Wk⁽ʰ⁾  Wv⁽ʰ⁾  → attention│─►│      Wo        │─► output
        └────────────────────────────────┘   └───────────────┘
              96 independent triples              one merger
```

**The shape of `Wq` (and `Wk`, `Wv`) can be written two equivalent ways:**

```
3D view (per-head):    96 × 128 × 12,288
                       └┬┘  └┬┘   └──┬──┘
                     heads  head_dim  embed_dim (input)

2D view (stacked):     12,288 × 12,288
                       └──┬──┘   └──┬──┘
                  96×128 = 12,288   embed_dim (input)
```

The `96 × 128` collapses into `12,288` because 96 heads × 128 dims/head = 12,288. Both
views hold the *same* numbers — the 2D matrix is just the 96 head-matrices stacked into
one tall matrix. In the 2D view you compute `q = Wq · e` once (→ 12,288-dim) and **slice
the result into 96 chunks of 128**, one chunk per head. That's identical math to running
96 separate `128 × 12,288` matrices, but packed into one big matmul — which is exactly
why it's GPU-friendly (one large multiply instead of 96 small ones).

**Parameter count, per block:**

```
Wq:  12,288 × 12,288   ≈ 151M   (≡ 96 × 128 × 12,288, same number ✓)
Wk:  12,288 × 12,288   ≈ 151M
Wv:  12,288 × 12,288   ≈ 151M
Wo:  12,288 × 12,288   ≈ 151M
                       ───────
                       ≈ 604M params per block (attention only)
× 96 blocks            ≈ 58B params in attention across the whole model
```

(The rest of GPT-3's 175B is mostly the FFN matrices `W_1`/`W_2`, even bigger at
`49,152 × 12,288` each — see Step 4.)

#### How the 96 heads end up *different* from each other

The line "each head has its own projections" says *that* they differ, not *why*. Three
facts together explain it:

1. **They start different — random initialization.** When the model is built, every
   weight matrix is filled with small random numbers, and each head gets a *different*
   random draw. So before any training the 96 heads are already slightly different, by
   chance, not design. This is **symmetry breaking** (point 3 needs it).

2. **Nothing forces them to specialize OR to stay identical.** The architecture treats
   all 96 heads the same. There is no rule saying "head 7 handles grammar." The only
   pressure is the shared next-token loss (Ch.2 §2.3).

3. **Backprop rewards division of labor.** All 96 head outputs are concatenated and
   fused by `Wo` (3f). If two heads computed the *same* thing, one is redundant — wasted
   capacity, higher loss than necessary. Gradient descent relentlessly lowers loss, so
   it pushes heads toward **complementary, non-overlapping** roles. Specialization
   *emerges* because diversity lowers loss and redundancy doesn't — nobody assigns it.
   (This is the "no supervision on attention itself" point from Ch.2 §2.4.)

> Why symmetry breaking matters: if all heads started with *identical* weights, they'd
> get *identical* gradients every step and update identically forever — staying clones.
> Random init breaks the tie so backprop can push them apart.

**What the specializations look like** (from interpretability research — illustrative,
not precise): previous-token heads (attend to the immediately prior token), positional
heads (attend to fixed slots), syntactic heads (verb → its subject), induction heads
(detect a repeated pattern and predict its continuation — thought to underlie in-context
learning), and entity/name-tracking heads. For "Who was the richest king?":

```
head 0  → "?" attends to "king"       (tracks the question's subject noun)
head 5  → "?" attends to "richest"    (tracks the key qualifier)
head 12 → "?" attends to "Who"        (who-question → expects a name)
head 47 → "richest" attends to "king" (binds adjective to its noun)
```

Each head views the sequence through a different lens; `Wo` blends all 96 lenses into
one vector. That's the "multi" in multi-head: **96 complementary views of the same
sequence.** Caveat: most heads are *not* cleanly interpretable — specializations are
partial, overlapping, and statistical (like the MoE experts in Chapter 3). The clean
labels are a useful mental model, not a precise truth.

#### 3a. Project Q, K, V (per token)

For token "Who" (`e_0`, 12,288-dim):

```
q_0 = Wq · e_0      Wq: 128 × 12,288   →  q_0 is 128-dim
k_0 = Wk · e_0      Wk: 128 × 12,288   →  k_0 is 128-dim
v_0 = Wv · e_0      Wv: 128 × 12,288   →  v_0 is 128-dim

q_0 = [ 0.21, -0.55,  0.08, ... ,  0.13 ]
       └──────── 128 values total ────────┘  (showing 4 of 128)
```

Do this for all 6 tokens → three Q, K, V matrices, each `128 × 6`:

```
Q = [q_0 q_1 q_2 q_3 q_4 q_5]   128 × 6
K = [k_0 k_1 k_2 k_3 k_4 k_5]   128 × 6
V = [v_0 v_1 v_2 v_3 v_4 v_5]   128 × 6
```

#### 3b. Scores = QᵀK / √head_dim

`√head_dim = √128 ≈ 11.314`. This is the **training / prefill case** — all 6 tokens
are present at once, so all 6 queries run simultaneously and M is the full
`seq_len × seq_len = 6 × 6` matrix; entry `M[i,j] = (q_i · k_j) / 11.314`.

> **At inference (decoding)**, only the *new* token's query is computed, so M is just
> **one row** (`1 × 6` at this point) — the new token scored against all past keys.
> The full 6×6 shown here only exists during training/prefill.

#### 3c. Causal mask (lower-triangular)

A token may only attend to itself and earlier tokens. Upper triangle → `-∞` before
softmax:

```
            key:  Who    was    the  richest king    ?
query Who   [   1.20   -inf   -inf   -inf   -inf   -inf ]
query was   [   0.40    0.95  -inf   -inf   -inf   -inf ]
query the   [  -0.10    0.22   1.05  -inf   -inf   -inf ]
query rich. [   0.33   -0.12   0.40   1.30  -inf   -inf ]
query king  [   0.51    0.18  -0.04   0.88   1.15  -inf ]
query  ?    [   0.22    0.07   0.31   0.66   0.74   0.90 ]
                              6 × 6, raw scores after /√128
```

#### 3d. Softmax each row → attention-weight matrix `M'`

Each row normalizes to sum 1 (only over unmasked entries). The result is `M'` (the
attention weights, `6 × 6`). Last row ("?"), i.e. `M'["?", ·]`:

```
M'("?", ·) = [ 0.08, 0.06, 0.09, 0.18, 0.21, 0.38 ]   sums to 1.00
               Who   was   the  rich  king   ?
```

At **block 0**, weights are still fairly diffuse — the model hasn't built much
context yet. "?" leans slightly toward "king"/"richest" and itself.

#### 3e. Head output = weighted sum of V (the `M'` row IS the mixer)

The 6 weights in `M'("?", ·)` from 3d are exactly the mixing coefficients. Each token
also produced a value vector `v_j` (128-dim) in 3a. 3e is a **weighted average of those
value vectors**:

```
head0_out("?") = 0.08·v_Who + 0.06·v_was + 0.09·v_the
               + 0.18·v_rich + 0.21·v_king + 0.38·v_?
               = [ -0.07, 0.12, 0.41, ... , 0.05 ]
                  └────────── 128 values ──────────┘
```

Data flow that ties 3b–3e together:

```
q·k  →  M (6×6)  →  /√128  →  mask  →  softmax  →  M' (weights)
                                                     │
                                                     ▼
                          head_out = Σ_j  M'[i,j] · v_j
```

Mental model: **Q·K decides *where to look*, softmax turns that into *how much*, V is
*what you actually collect*.**

#### 3f. Concat 96 heads → output projection Wo → residual add

`Wo` is a **learned weight matrix** (the "output projection"), a trained parameter
like Wq/Wk/Wv. The 96 heads each produced an independent 128-dim answer; they must be
fused back into one 12,288-dim vector.

```
concat("?") = [ head0(128) | head1(128) | ... | head95(128) ]  = 12,288
attn_out("?") = Wo · concat("?")     Wo: 12,288 × 12,288  →  12,288
```

What `Wo` does: every output dimension becomes a learned weighted blend of all 96
heads' findings. It (a) recombines the parallel heads into one coherent vector and
(b) remaps back into the residual stream's coordinate system (`embed_dim`-shaped). It's
a full linear projection — 12,288 dot products, one per output dimension — not a dot
with a single vector.

Then the **residual add**:

```
e_0' = e_0 + attn_out                  (still 12,288-dim, per token)
       residual stream after block-0 attention
```

### Step 4 — Flow through all 96 blocks

```
E ──► [block 0] ──► [block 1] ──► ... ──► [block 95] ──► H
       attn+FFN      attn+FFN              attn+FFN
```

The shape stays `12,288 × 6` the entire way. What changes is the **content**: by deep
layers each column has absorbed information from the tokens it's allowed to see.

(FFN sublayer per block: LayerNorm → `W_1` `49,152 × 12,288` → GELU →
`W_2` `12,288 × 49,152` → another residual add.)

### Step 5 — FINAL attention (decoder block 95)

Mechanically identical to block 0 (same shapes: Q,K,V are `128 × 6` per head, scores
`6 × 6`, masked, softmaxed). The difference is **what the vectors now mean**. By block
95 the "?" column has aggregated the whole question, so its query/key projections
produce a much **sharper, more semantic** attention pattern.

Last row ("?") softmax at block 95 — `M'_L95("?", ·)` — contrast with block 0's diffuse
row:

```
M'_L95("?", ·) = [ 0.02, 0.01, 0.03, 0.31, 0.55, 0.08 ]
                   Who   was   the  rich  king   ?
```

Now "?" attends hard to "king" (0.55) and "richest" (0.31) — the model has decided the
relevant content is "richest king." Head output, concat, `Wo`, residual add proceed
exactly as in Step 3e–3f, producing the final residual stream `H` (`12,288 × 6`).

### Step 6 — Final LayerNorm

GPT-3 applies a final LayerNorm to the entire `H` matrix (`12,288 × 6`):

```
H_norm = LayerNorm(H)    shape: 12,288 × 6    ← all 6 columns normalised
```

> **Inference vs training diverge here.** At inference (generating one token) you only
> need the **last column** (position 5, "?") — that's the one predicting the next word.
> During training you need **all 6 columns** — each column produces a prediction for its
> position and contributes a loss. Both paths go through the same LayerNorm; they just
> use different columns afterwards.

### Step 7 — De-embedding (LM head → logits)

The unembedding matrix `W_U` has shape `V × embed_dim = 50,257 × 12,288`. (In GPT-3
this is **tied** to `W_E` — same matrix reused.) Applied to the full `H_norm`:

```
logits = W_U · H_norm     (50,257 × 12,288) · (12,288 × 6)  →  50,257 × 6
```

The result is a **`50,257 × 6` matrix** — one full 50,257-dim logit vector per
position. Every column is an independent prediction over the entire vocabulary:

```
logits shape = 50,257 × 6

              col0       col1       col2       col3       col4       col5
            ┌                                                              ┐
"cat"       │  18.7   │   3.1   │   0.4   │  ...    │  ...    │  ...    │
"sat"       │   3.1   │  21.4   │  ...    │  ...    │  ...    │  ...    │
"on"        │  ...    │  ...    │  17.9   │  ...    │  ...    │  ...    │
  ...       │  ...    │  ...    │  ...    │  ...    │  ...    │  ...    │
            └                                                              ┘
              ↑           ↑          ↑          ↑          ↑          ↑
            "cat"?      "sat"?     "on"?      "the"?     "mat"?    "???"
           training    training  training   training   training   (unused
           sample 1    sample 2  sample 3   sample 4   sample 5   in training)
```

> **At inference** (Chapter 1's worked example), only column 5 ("?") is used:
> `logits[:, 5]` → `50,257 × 1` → softmax → pick next token.
>
> **During training**, all 6 columns are used: each column's `50,257 × 1` logit vector
> → softmax → cross-entropy loss vs its target → 5 loss values → averaged →
> `batch_loss`.

### Step 8 — Softmax → pick next token (inference only)

At inference, softmax over the last column's 50,257 logits → a probability distribution:

```
P(next token | "Who was the richest king?") =
   "Charles"  → 0.24
   "Henry"    → 0.11
   "King"     → 0.07
   "Mansa"    → 0.06     ← (Mansa Musa, the historically common answer)
   ...                      (50,257 entries, summing to 1.0)
```

Greedy decoding takes the argmax → emits **"Charles"** (or with sampling, draws from
the distribution). That token is appended, the sequence becomes 7 tokens long, and the
whole pipeline (Steps 1–8) runs again to produce the token after that — the
autoregressive loop.

### One-glance shape trace

```
"Who was the richest king?"
   │  tokenize
   ▼   6 IDs
W_E lookup           50,257 × 12,288  →  6 vecs of 12,288
+ W_P position       2,048  × 12,288  →  E: 12,288 × 6
   │
   ▼  block 0 attention:  Wq/Wk/Wv 128×12,288 per head ×96
      Q,K,V 128×6 │ M: 6×6 (masked) → M': 6×6 │ Wo 12,288×12,288 │ +residual
   ▼  ... ×96 blocks (shape stays 12,288 × 6 throughout) ...
   ▼  block 95 attention (same shapes, sharper weights)
final LayerNorm      → H_norm: 12,288 × 6          (all 6 columns)
W_U de-embed         50,257 × 12,288  →  logits: 50,257 × 6   (one 50,257-vec per position)

  INFERENCE: W_U applied to col (seq_len−1) only → 50,257 × 1 → softmax + argmax → "Charles"
             (only the last column is needed; other columns never multiplied with W_U)

  TRAINING:  W_U applied to ALL seq_len columns  → 50,257 × seq_len
             → seq_len losses computed simultaneously (one per position)
             → averaged → batch_loss → one backward pass → one weight update
             THIS is the training parallelism: one forward pass does the work
             of seq_len separate inference passes
```

---

## 1.5 — The residual stream: why `e_0 + attn_out` (add, not replace)

A common confusion: *why add `attn_out` back to `e_0`, who consumes it, and why isn't
`attn_out` the next token?*

### attn_out is NOT a prediction

A token prediction must be a vector of size `V = 50,257` (one score per vocab word).
That only exists at the very end, after the LM head:

```
logits = W_U · h_last      → 50,257 numbers (one per vocab word)  ← THIS can become a token
```

But `attn_out` is `12,288` numbers — the wrong size and the wrong *kind* of thing.
It's one attention sublayer's contribution at **layer 0 of 96**. The model has barely
started. The next token only materializes at the very end, for the **last position**,
after unembedding turns a 12,288-vector into 50,257 word-scores.

```
e_0 ──► block 0 ──► block 1 ──► ... ──► block 95 ──► LayerNorm ──► W_U ──► softmax ──► token
        (attn_out                                                  ▲
         lives in here,                                            │
         deep inside step 1)                          ONLY here does a token appear
```

### Who consumes `e_0 + attn_out`

The **next sublayer**. `e_0' = e_0 + attn_out` is the input to the FFN sublayer in the
same block; that block's output feeds block 1, and so on. There is a single shared
12,288-wide bus that every sublayer reads from and writes to — the **residual stream**.

```
e_0 ─►(＋)─► e_0' ─► [FFN] ─►(＋)─► e_0'' ─► block 1 ─► ... ─► block 95
      ▲                     ▲
   attention            FFN writes
   writes here          here
```

### Why add instead of replace

`attn_out` is **not "the new meaning of the token"** — it's the **context** gathered
from the rest of the sentence (a `delta`). The token also needs to remember **what it
itself is**; that identity lives in `e_0`.

```
e_0'   =   e_0        +   attn_out
           ▲              ▲
        "I am '?'      "context: this is about
         at pos 5"      the richest king"
```

If you replaced (`e_0' = attn_out`), you'd **erase the token's own identity** and keep
only the gathered context — the next layer wouldn't know what token it's working on.
Adding keeps both, and each layer adds another increment on top.

Two reasons it must be `+`:

1. **Attention is a refinement, not a rewrite.** `attn_out` = "here's context to add."
   The token keeps its identity (`e_0`) and *gains* context.
2. **Gradient flow.** The `+` creates a highway for gradients to skip backward through
   the sublayer untouched (derivative of `e_0 + attn_out` w.r.t. `e_0` is 1). Across 96
   layers this prevents vanishing gradients and makes deep stacks trainable. (Residual /
   skip connection, from ResNets.)

**Mental model — the running notebook.** The residual stream is a per-token notebook.
`e_0` is what's already written; attention reads it and **writes a margin note**
("relates to 'richest king'"); the `+` appends that note. You never tear the page out.
After 96 layers the notebook is dense enough that the LM head can read it and pick the
next word.

---

## 1.6 — Parallel vs. sequential: what runs at the same time

The 96 heads and 96 blocks share the same number in GPT-3, but they mean completely
different things — one is parallel, one is sequential:

```
HEADS  → parallel  (all 96 heads inside one block run at the same time)
BLOCKS → sequential (block 0 finishes, feeds block 1, feeds block 2, ...)
```

Inside one block, all 96 heads fire simultaneously on the same input, then `Wo` merges
them:

```
            ┌───────────────────────── BLOCK k ─────────────────────────┐
            │  head0  head1  head2  ...  head95                          │   ← 96 heads, ALL parallel
   input ──►│   │      │      │           │                              │
  12,288×   │   └──────┴──────┴─────┬─────┘  each produces 128×seq_len  │
  seq_len   │              concat (12,288×seq_len) → Wo → FFN            │
            └────────────────────────────┬──────────────────────────────┘
                                          ▼
                                       output  (goes to block k+1)
                                       12,288 × seq_len
```

Then 96 blocks run one after another:

```
input ─► [BLOCK 0] ─► [BLOCK 1] ─► [BLOCK 2] ─► ... ─► [BLOCK 95] ─► final
          96 heads     96 heads     96 heads            96 heads
          (parallel)   (parallel)   (parallel)          (parallel)
         └──────────────────── sequential chain, 96 deep ───────────────┘
```

**Why heads are parallel but blocks are not:**
- **Heads are parallel** — they all read the *same* block input (`12,288 × seq_len`) and
  don't depend on each other. Head 5 doesn't need head 4's answer. `Wo` combines them at
  the end. No ordering required → parallel.
- **Blocks are sequential** — block 1's input *is* block 0's output. Block 1 can't start
  until block 0 has produced its refined `12,288 × seq_len` residual stream. Strict order.

Kitchen analogy with GPT-3 numbers:
- **96 heads** = 96 cooks at one station, each working on a different 128-dim slice of
  the 12,288-wide input at the same time, then everything combined by `Wo`. Parallel.
- **96 blocks** = 96 stations in a line; the sequence moves station to station, each
  refining the previous result. Sequential.

So the 96 in "96 heads" and the 96 in "96 layers" are the same *number* but describe
completely different things: one is the width of parallel work inside a single step, the
other is the depth of sequential steps.

---

## 1.7 — Heads vs. layers: two independent knobs

**The number of heads does NOT need to match the number of layers.** GPT-3 175B having
`96 heads` *and* `96 layers` is a coincidence. They control different axes:

- **Number of layers (blocks) = depth.** How many times you stack the
  "attention → FFN" refinement. The *vertical* dimension.
- **Number of heads = width of one attention layer.** How many parallel perspectives a
  *single* block splits into. A *horizontal* dimension *inside one block*.

```
        ┌─ block 0:  [ head0 head1 ... head95 ] → Wo → FFN ┐
        │  block 1:  [ head0 head1 ... head95 ] → Wo → FFN │
depth   │   ...                                            │  ← n_layers = 96 (vertical)
(96)    │  block 95: [ head0 head1 ... head95 ] → Wo → FFN ┘
                      └──────── width: n_heads ────────┘
                              (horizontal, per block)
```

### The ONE real constraint on heads

```
n_heads × head_dim = embed_dim
```

Multi-head attention splits the 12,288-wide vector into `n_heads` slices, runs
attention on each, then concatenates back to 12,288. For GPT-3:
`96 × 128 = 12,288 ✓`. So the requirement is just: **`embed_dim` divisible by
`n_heads`** (integer `head_dim`). Layer count is unconstrained.

### Worked example: "131 heads, 96 blocks"

96 blocks is fine. The only question is whether 131 heads divides `embed_dim`:

```
12,288 / 131 = 93.8...   ✗  not an integer → fractional head dim, invalid
```

Two fixes:
- **Pick a divisor of 12,288**: `128 × 96 = 12,288 ✓` or `192 × 64 = 12,288 ✓` —
  both valid with 96 blocks.
- **Change embed_dim**: `131 × 96 = 12,576 = embed_dim ✓` — now 131 heads is legal.

Dependency graph:

```
n_layers        ── independent knob (depth)
n_heads ─┐
         ├─ must satisfy:  n_heads × head_dim = embed_dim
head_dim ─┘                (embed_dim divisible by n_heads)
embed_dim         ── independent knob (width of residual stream)
```

---

## 1.8 — KV Cache

### The problem it solves

During inference the model generates one token at a time. Each step takes the full
sequence so far, runs it through all 96 blocks, and uses only the last column's logits
to pick the next token. Then the new token is appended and the whole thing runs again.

The expensive part is inside every attention block, where K and V are computed for
every token in the sequence:

```
K = Wk · E     (128 × 12,288) · (12,288 × seq_len)  →  128 × seq_len
V = Wv · E     (128 × 12,288) · (12,288 × seq_len)  →  128 × seq_len
```

Without any optimisation, step N recomputes K and V for **all N tokens** — including
the N−1 tokens already processed in previous steps. Those tokens haven't changed, so
their K and V are identical to last step. Pure repeated work.

For a 2,048-token sequence:

```
Step 1:    compute K,V for  1 token
Step 2:    compute K,V for  2 tokens  (token 1 recomputed unnecessarily)
Step 3:    compute K,V for  3 tokens  (tokens 1–2 recomputed)
...
Step 2048: compute K,V for 2,048 tokens

Total ∝ 1 + 2 + 3 + ... + 2048 ≈ 2M computations  — quadratic
```

### What the KV cache does — step by step example

Input prompt: `"Who is the richest king?"` → output: `"The richest king is Charles"`

**Step 0 — Prefill** (the prompt, all tokens known upfront):

```
Compute K,V for all 6 prompt tokens across all 96 blocks simultaneously.
Cache now holds:  [Who] [is] [the] [richest] [king] [?]
Last column logits → predict → "The"
```

This first pass is free in the sense that the full sequence is already available, so
all 6 K/V pairs are computed in one parallel matmul — no sequential cost.

**Step 1 — Decode "The":**

```
New token "The" enters.
Compute K,V for "The" only → store in cache.
Retrieve K,V for [Who][is][the][richest][king][?] from cache (no recompute).
Cache now holds:  [Who] [is] [the] [richest] [king] [?] [The]
Last column logits → predict → "richest"
```

**Step 2 — Decode "richest":**

```
New token "richest" enters.
Compute K,V for "richest" only → store in cache.
Retrieve K,V for all 7 prior tokens from cache.
Cache now holds:  [Who] [is] [the] [richest] [king] [?] [The] [richest]
Last column logits → predict → "king"
```

**Step 3, 4, ... — same pattern:**

```
Each step: compute K,V for the ONE new token, store, retrieve all prior from cache.
Cache grows by exactly one column per step.
```

Total K,V computations = one per token, once each = **linear**, not quadratic.

### Why only K and V are cached — not Q

**During inference, only the latest token's Q is used (`q_N · k_j` for all j). Past
tokens' Q is never used by anyone. So there is nothing to cache.**

In detail: at step N, M collapses to one row — the new token's query scored against
all past keys (§1.3 step 4). That row only needs `q_N`. Past tokens' Q vectors were
each used once, at the step when *they* were the new token, to compute their own row
of M. After that they served their purpose and were discarded. No future step ever
reads a past Q.

K and V are the opposite — every future token needs all past K's (to compute its row
of M) and all past V's (to blend the result). They keep being reused → worth caching.

```
At step N (generating token N):

  q_N            ← computed fresh — used once to compute q_N · k_j for all j
  k_0 .. k_N     ← retrieved from cache + k_N computed fresh and stored
  v_0 .. v_N     ← retrieved from cache + v_N computed fresh and stored

  q_0 .. q_{N-1} ← never used at step N. Each was used once at its own step.
                   Not cached — no future step will ask for them.
```

**K and V face outward** — left for future tokens to use, reused every step → cache.
**Q faces inward** — used once to compute your own row of M, then gone → don't cache.

### What gets cached, and the memory cost

Per block, per head, the cache stores:

```
K cache:  128 × seq_len_so_far   (grows by one column each step)
V cache:  128 × seq_len_so_far   (grows by one column each step)
```

Across all layers and heads at `seq_len = 2,048`, FP16 (2 bytes):

```
cache size = 2 × 96 layers × 96 heads × 128 head_dim × 2,048 tokens × 2 bytes
           ≈ 9.7 GB
```

The cache grows with every token. This is the **memory cost** — trading compute
(not recomputing K/V) for memory (storing all past K/V). For long sequences the cache
can rival the model weights in size. This is exactly the problem GQA (Chapter 3) solves
by sharing K/V heads across query heads, shrinking the cache proportionally.

### Summary

| | Without KV cache | With KV cache |
|---|---|---|
| K,V computation | Recomputed for all past tokens every step | Computed once per token, stored, reused |
| Cost | ∝ seq_len² (quadratic) | ∝ seq_len (linear) |
| Memory | None extra | Grows each step (~9.7 GB at 2,048 tokens) |

The KV cache trades memory for speed — why generating a 2,048-token response doesn't
take 2,048× longer than generating a 1-token response.
