# Chapter 3 — Gemma 4 (2026): The Improvements

Gemma 4 is Google DeepMind's 4th-gen open model family (released April 2, 2026,
Apache 2.0). Sizes: E2B, E4B, 12B, 26B-A4B (MoE), 31B dense. Multimodal
(text + image + audio), up to **256K context**, 140+ languages.

Core theme: **more capability per parameter and per FLOP** through smarter
architecture — not just a bigger dense stack.

## 3.1 — Architecture diagram (decoder-only)

Unlike the classic Transformer (encoder + decoder, absolute positions added at input),
Gemma 4 is a **decoder-only** stack with **no input positional encoding** (RoPE is
applied inside attention) and **pre + post RMSNorm** around each sublayer.

```
                         Token IDs
                             │
                  ┌──────────▼───────────┐
                  │   Token Embedding     │  262K vocab (tied with LM head)
                  └──────────┬───────────┘
  image / audio ──►[encoder → projector]──┐
                             │            │  multimodal tokens
                             ▼◄───────────┘  merged into stream
                  ╔══════════════════════╗
                  ║     Decoder block      ║  × 30   (26B-A4B)
                  ║   (expanded below)     ║
                  ╚══════════┬═══════════╝
                             │
                  ┌──────────▼───────────┐
                  │     Final RMSNorm     │
                  └──────────┬───────────┘
                  ┌──────────▼───────────┐
                  │   LM head  → logits   │
                  └──────────┬───────────┘
                             ▼
                        next token
        (optional MTP drafter head ──► speculative decoding, ~3×)
```

One decoder block (pre-norm + post-norm "sandwich", two residual adds):

```
   residual in ─────────────────────────┐
        │                                │
   ┌────▼─────┐                          │
   │ RMSNorm   │ (pre)                    │
   └────┬─────┘                          │
   ┌────▼───────────────────────┐        │
   │  ATTENTION                  │        │
   │   • Q,K,V projections (GQA: │        │
   │     few shared K/V heads)   │        │
   │   • RoPE rotates Q,K        │        │
   │   • QK-norm                 │        │
   │   • layer is EITHER         │        │
   │     local sliding-window    │        │
   │     OR global  (≈ 5:1)      │        │
   └────┬───────────────────────┘        │
   ┌────▼─────┐                          │
   │ RMSNorm   │ (post)                   │
   └────┬─────┘                          │
        ▼                                │
       (＋)◄──────────────────────────────┘  residual add
        │
        ├──────────────────────────────┐
   ┌────▼─────┐                         │
   │ RMSNorm   │ (pre)                   │
   └────┬─────┘                         │
   ┌────▼───────────────────────┐       │
   │  FEED-FORWARD               │       │
   │   MoE: router picks         │       │
   │   top-8 of 128 experts      │       │
   │   (GeGLU experts)           │       │
   │   [31B model = 1 dense      │       │
   │    GeGLU FFN instead]       │       │
   └────┬───────────────────────┘       │
   ┌────▼─────┐                         │
   │ RMSNorm   │ (post)                  │
   └────┬─────┘                         │
        ▼                               │
       (＋)◄─────────────────────────────┘  residual add
        │
   ┌────▼────────────────────────┐
   │ + PLE (per-layer embedding)  │  looked up from flash,
   │   ~256-dim → up-proj → embed_dim│  added per layer
   └────┬────────────────────────┘
        ▼
   residual out  ──►  next block
```

## 3.2 — How to bucket the main techniques

There are three goals: save **memory**, save **compute**, improve **capability/quality**.

| Technique | Primary job | Bucket |
|---|---|---|
| GQA (grouped K/V) | Shrink the KV cache | Memory |
| Sliding Window Attention | Avoid the n² attention blowup | Compute/speed |
| RoPE | Represent position + handle long contexts | Capability |
| QK-norm | Stabilize training, steadier attention scores | Quality/stability |
| MoE | More knowledge at low per-token cost | Capacity vs compute |

One-liner: **GQA saves memory, Sliding Window saves compute, RoPE unlocks long
context, QK-norm keeps training stable, MoE adds knowledge without adding effort.**

Long-context teamwork: RoPE makes far positions *representable*, Sliding Window
makes long attention *affordable*, GQA makes the long KV cache *fit in memory*.

---

## 3.3 — RoPE — Rotary Position Embeddings

Instead of *adding* a position vector to `e` (GPT-3), RoPE **rotates** the `q` and
`k` vectors by an angle based on position, at each attention layer. The word's
meaning stays in the vector; the rotation angle encodes where it sits.

**2D rotation reminder** — spin an arrow `(x, y)` by angle `θ`:
```
x' = x·cos θ − y·sin θ
y' = x·sin θ + y·cos θ
```
Properties that matter: length preserved, angles preserved, rotations add
(`R(θ₁)·R(θ₂) = R(θ₁+θ₂)`).

**The magic**: if query at position `m` is rotated by `m·θ` and key at position `n`
by `n·θ`, their dot product depends only on the **difference `m − n`** (relative
distance), because `cos(mθ)cos(nθ) + sin(mθ)sin(nθ) = cos((m−n)θ)`.

**How the angle is determined** (NOT learned — a fixed schedule):
```
angle = m · θ_i        where   θ_i = 1 / (base^(2i / d))
```
- `m` = token position, `i` = which dimension-pair, `d` = head dim, `base` = 10,000
- The 128-dim vector is split into 64 pairs; each pair is a little 2D arrow rotated
  by its own speed `θ_i`.
- Speeds form a geometric progression: pair 0 spins fast (fine, short-range
  resolution), pair 63 spins slow (broad, long-range). Like clock hands.

**Why it scales**: angle is linear in `m`, so there's no lookup table to run out
of. Cranking `base` (e.g. 10,000 → 1,000,000) slows all the hands and extends the
representable range — this is RoPE/NTK scaling, a big part of reaching 256K tokens.

### Worked example — "The cat sat on the mat" (4-dim head, 2 pairs)

`θ_0 = 1.0` (fast), `θ_1 = 0.01` (slow). The word "the" appears at pos 0 and pos 4;
same raw vector `[1,0,1,0]` gets different fingerprints:
```
the@0 → [ 1.000,  0.000, 1.000, 0.000]   (no rotation)
the@4 → [-0.654, -0.757, 0.999, 0.040]   (fast pair swung far, slow pair barely moved)
```
Dot product between query@m and key@n collapses to `cos((m−n)·θ)` → pure distance:
- "sat"(2) ↔ "cat"(1): cos(1) ≈ 0.540  (distance 1)
- "sat"(2) ↔ "The"(0): cos(2) ≈ −0.416 (distance 2)

### Extending context after training — base scaling, PI, NTK, YaRN

**Why a RoPE model still has a limit.** RoPE extrapolates *better* than absolute
embeddings, but it's not infinite. A model trained at length `L` only ever saw rotation
angles up to `L·θ_i`. Push past `L` and the slow hands swing into angle territory the
model **never saw in training** → attention scores can blow up and quality drops abruptly.

**The key high-freq / low-freq insight** (this is the crux, and it's just the clock hands
again):
- **Fast hands** (high-freq, early pairs) complete *many* full turns even within the
  trained length → the model has effectively seen "every angle" for them → they
  **extrapolate fine** past `L`.
- **Slow hands** (low-freq, late pairs) sweep only a *small arc* within `L` → going past
  `L` pushes them into unseen angles → **these are what break**. So they're the ones that
  need help.

**The methods, in order of cleverness:**

1. **Position Interpolation (PI) — linear scaling.** Don't extrapolate; instead *squeeze*
   new positions back into the trained range by scaling indices by `L/L'` (e.g. treat
   position 8000 as "2000" for a 2K model). Downside: it compresses *all* frequencies
   uniformly, blurring the fine high-freq distinctions (tokens 1 apart start to look ¼
   step apart). Needs some fine-tuning.

2. **NTK-aware scaling — change the `base`, not the positions.** Instead of scaling every
   dimension equally, **increase the base `θ`** (e.g. 10,000 → larger). This is
   *non-uniform* by construction: high-freq hands barely change (good — they extrapolate
   fine), low-freq hands get effectively interpolated (good — they were the problem). Can
   work **without fine-tuning**. "Dynamic NTK" recomputes the base as the sequence grows.

3. **YaRN ("NTK-by-parts") — frequency-aware, the current go-to.** Explicitly bins
   dimensions: leave high-freq alone (extrapolate), interpolate low-freq, blend the
   middle — plus an **attention temperature** tweak to keep dot-product variance healthy.
   Extends to ~128K with **10× fewer tokens and 2.5× fewer training steps** than earlier
   methods.

4. **LongRoPE** and friends push to ~2M tokens via per-dimension search.

**So, how is `base` tuned in practice?** The mainstream production recipe (Llama, Gemma,
etc.) is exactly NTK's core move: **replace the rotary base 10,000 with a larger value**
(commonly 5×10⁵–1×10⁶ or higher) and do a **short fine-tune on longer text**. Bigger base
= slower low-freq hands = longer wavelengths = positions stay distinguishable much
further out. That single hyperparameter change, plus a little long-context fine-tuning, is
what turns a few-thousand-token model into a 256K one — no architecture change required.

> Lineage of context growth: APE → RoPE → PI → NTK-aware → Dynamic NTK → YaRN → LongRoPE,
> taking usable context from ~512 to 2M+ tokens, mostly with minimal fine-tuning.

---

## 3.4 — Sliding Window Attention

Full attention = every token looks at every previous token → cost grows with n²
(brutal as sequences get long). Sliding window restricts each token to its last
~512–1024 neighbors → cost grows linearly (`n × window`).

Far-away info still propagates indirectly: neighbors looked at *their* neighbors, so
information travels layer-by-layer like a game of telephone. Gemma 4 mixes mostly
sliding-window layers (fast) with occasional global-attention layers (catch anything
the telephone dropped). Small quality tradeoff for big efficiency win.

---

## 3.5 — GQA — Grouped-Query Attention

Motivation = the **KV cache**: during generation, K and V of every past token are
cached. Size ∝ `layers × heads × seq_len`, and it can exceed the model weights and
become the speed bottleneck.

- **MHA** (original): each head has its own Q, K, V → 96 K/V sets → biggest cache.
- **MQA** (extreme): all heads share ONE K and V → 96× smaller cache, some quality loss.
- **GQA** (middle): split heads into groups; each group shares one K/V → e.g. 96
  queries but only ~8–12 K/V sets. Most of MHA's quality at most of MQA's speed.

Analogy: 96 reporters (queries). MHA = 96 research assistants, MQA = 1 shared
assistant (bottleneck), GQA = 12 assistants each serving 8 reporters.

Smaller KV cache is exactly what lets Gemma 4 run on a laptop and hold 256K context.

### KV cache memory math (MHA vs GQA vs MQA)

The cache stores K and V for every past token, at every layer, for every K/V head:

```
KV cache bytes = 2 × layers × kv_heads × head_dim × seq_len × bytes_per_value × batch
                 ↑
                 K and V
```

The only term that changes between schemes is **`kv_heads`**: MHA uses all heads, GQA
uses #groups, MQA uses 1. Everything else is identical.

**Worked example.** Hypothetical model: 32 layers, 32 query heads, head_dim 128
(embed_dim 4096), FP16 (2 bytes), batch 1.

Per-token cost (`2 × layers × kv_heads × head_dim × 2 bytes`):

| Scheme | kv_heads | Per token | At 100K ctx | At 256K ctx |
|---|---|---|---|---|
| MHA | 32 | 512 KB | **~52 GB** | ~134 GB |
| GQA (groups of 4) | 8 | 128 KB | **~13 GB** | ~34 GB |
| MQA | 1 | 16 KB | **~1.6 GB** | ~4.2 GB |

The point lands hard: at 100K tokens, **MHA's KV cache alone (~52 GB) dwarfs the model
weights** and won't fit on a single consumer GPU. GQA's 4× cut (~13 GB) makes it
feasible; MQA's 32× cut (~1.6 GB) is tiny but costs some quality. GQA is the sweet spot.

(Sanity check, MHA @100K: `2 × 32 × 32 × 128 × 100,000 × 2 = 52.4e9` bytes ≈ 52 GB.)

**Tie-in:** sliding-window attention shrinks this further — for local layers the cache
only needs the last `window` tokens, not the full `seq_len`. So GQA (fewer kv_heads) and
sliding window (smaller effective seq_len) attack the *same* formula from two different
terms. Together with RoPE (making the positions representable at all), that's the full
"how do you afford 256K context" answer.

---

## 3.6 — QK-norm + RMSNorm

GPT-3 used LayerNorm. Gemma uses **RMSNorm** (cheaper — no mean subtraction) and
normalizes the query/key projections (**QK-norm**) so attention logits don't explode
at scale. Net effect: more stable training and steadier, more accurate attention.

---

## 3.7 — MoE — Mixture of Experts (the 26B-A4B variant)

Dense model = all parameters fire for every token. MoE adds a **router** that, per
token, picks only the top few "experts" (chunks of feed-forward weights) to run.

- `26B-A4B` = **26B total params**, but only **~4B active per token** → ~8× less
  compute per token than an equivalent dense model.
- Best of both: large knowledge capacity (all experts exist) + small per-token cost
  (only a few run).

### Gemma 4 26B-A4B — exact MoE config

| Property | Value |
|---|---|
| Total / active params | ~25.2B total / ~3.8B active per token (rounded to "26B-A4B") |
| Experts per MoE layer | **128 fine-grained experts** |
| Routing | **top-8** (8 of 128 experts fire per token) |
| Layers | 30 |
| Attention | Hybrid: alternating sliding-window + global, ~5:1 (or 4:1) ratio |
| KV | Shared KV cache (GQA) |
| Extras | Per-Layer Embeddings (PLE); native text + image (structured thinking, function calling) |
| Context | 256K (26B-A4B and 31B); 128K on E2B/E4B |

Notes:
- "Fine-grained" = many small experts (128) with a relatively high top-k (8), rather
  than a few big experts — gives finer routing granularity.
- No explicit *shared/always-on* expert documented; it's straight top-8 routing.
- The 26B vs 25.2B / 4B vs 3.8B differences are just marketing-rounded vs exact.
- **Pruning is real here**: community builds already exist that drop 128 → 98 experts
  per layer via contribution-based drop maps — direct confirmation of the
  "prune rarely-used experts" option below.

> Source caveat: these figures come from inference-provider and community model docs
> (vLLM recipe, Hugging Face model repos), which agree closely; treat exact counts as
> "very likely" rather than official-spec-confirmed.

**Important reality checks (correcting the "100 specialists" analogy):**
- Experts are **NOT** human-interpretable specialists. No clean "math expert" or
  "French expert" — specializations are subtle/statistical (subword patterns, token
  types), discovered during training.
- Routing is **per-token AND per-layer**, not per-task. A single sentence activates a
  different expert mix for every token at every MoE layer. "4B active" is an *average
  per token*, not a fixed extractable subset.
- So there is no stable "which 4B for task X" — it churns token by token.

**Can you squeeze out a custom/smaller expert for a specific need?** Not by cleanly
extracting one. Options, roughly best-to-worst fit:
1. **Fine-tune the whole MoE (LoRA/adapters)** — most reliable for "make it great at
   my thing"; router + experts re-specialize toward your data. Keeps full size.
2. **Distill into a small dense model** — train a new smaller model to imitate
   outputs on your task; cleanest path to a compact, fast, task-specific model.
3. **Expert pruning by usage profiling** — run your data, drop rarely-fired experts;
   shrinks the model but limited before quality drops (routing is distributed).
4. **Expert merging/compression** — research-y, quality-sensitive.

Training caveat: the router must be trained well (load balancing) so it doesn't keep
routing to the same few experts while others stay idle.

---

## 3.8 — PLE — Per-Layer Embeddings

**The problem it solves.** In a vanilla transformer, each token gets **one** embedding
at the input layer. That single vector then flows up through the whole residual stream,
and every layer just builds on top of it — the initial representation has to carry all
the semantic weight by itself.

**What PLE does.** It adds a **second embedding table for *each* decoder layer**. As a
token passes through layer N, that layer looks up its own per-layer embedding for the
token and injects it alongside the residual stream. So the model can **introduce and
refine** a token's representation at every layer, instead of relying on one initial
vector. This is the "secondary conditioning signal per layer" idea (inherited from
Gemma-3n).

**The clever memory trick (the real innovation).** These per-layer embedding tables are
big, but they're just **lookups** — no matrix multiply. So Gemma stores them in
**flash / fast local storage, NOT in VRAM**:

- VRAM is the #1 constraint for on-device inference (phones, laptops); it's what you run
  out of first.
- Each PLE entry is **low-dimensional (~256)**, then **up-projected** to the full model
  embedding size at inference time.
- The params count toward the model's *total* size but don't occupy precious VRAM —
  they're cached to disk and fetched per-token, per-layer as needed.

**Why it matters.**
- *Quality*: per-layer conditioning boosts "intelligence per active parameter."
- *Efficiency*: decouples a big chunk of embedding parameters from the VRAM/compute
  path → much smaller live memory footprint. This is a big reason the "E" (edge) sizes
  have more *total* params than their *effective* in-VRAM footprint (e.g. E2B/E4B
  naming reflects effective, not total, params).

**Mental model:** the residual stream is the main conversation; PLE is a per-layer
"cheat sheet" for each token that the model pulls from cheap storage and consults at
every floor of the building, instead of memorizing everything at the front door.

> Related on-device tricks from the Gemma-3n lineage worth a later look: MatFormer
> (nested FFNs = many model sizes in one), Conditional Parameter Loading, AltUp,
> and LAuReL.

---

## 3.9 — MTP — Multi-Token Prediction

**The problem.** Normal generation is autoregressive: predict 1 token, append it, predict
the next, repeat. Each forward pass touches every (active) weight just to emit **one**
token. A 50-token reply = 50 full passes. That per-token overhead is the main latency
bottleneck.

**Core idea.** Train the model with **extra lightweight heads that forecast multiple
future tokens** (not just the next one). Popularized by DeepSeek-V3, which added a second
loss per position (also predict token t+2, not just t+1) via small sequential heads that
still respect causal order.

MTP pays off in **two** ways:

1. **Better training/quality.** Forcing the model to predict a *horizon* of tokens gives
   a richer learning signal and strengthens "planning ahead." The extra MTP parameters
   get distilled back into the main model through gradient flow during pretraining.
2. **Faster inference** via **speculative decoding** — the trained MTP heads are
   repurposed as a "drafter."

**Speculative decoding (the inference payoff).**
- A small, cheap **drafter** proposes several candidate tokens ahead.
- The big **target** model verifies them **all in a single parallel forward pass**.
- If a prefix of the draft is accepted, the target emits all those tokens in the
  wall-clock time of **one** step → multiple tokens for the cost of one verification.
- **Lossless**: verification guarantees the output is identical to normal one-at-a-time
  decoding. So it's "faster, *same outputs*" — not an approximation.

**Gemma 4 specifics.**
- Google ships dedicated **MTP drafter** models → up to **~3× faster inference, no
  quality loss**.
- The drafter **shares the input embedding table** with the main model and reuses the
  **activations from the main model's final layer**, so extra memory is minimal and
  setup is simple.
- To draft cheaply over Gemma 4's huge **262K-token vocabulary**, the drafter uses
  **sparse decoding via clustered token lookup** (find the likely token cluster first,
  then compute logits only inside it — a two-stage retrieval on the LM head).
- Supported in Ollama, MLX, and HF Transformers.

**Mental model:** an eager intern (drafter) scribbles the next few words it thinks you'll
say; the expert (target) glances at the whole guess at once and keeps the part that's
right. When the intern guesses well, you get several words in the time it'd normally take
to write one — and the expert never lets a wrong word through.

> Contrast with PLE/GQA/MoE: those change the *model*. MTP's speculative-decoding side
> changes *how you run* an unchanged model — Google's "3× faster without changing the
> model" framing. (The training side does shape the model, though.)

---

## 3.10 — Cheat sheet: GPT-3 → Gemma 4

| Dimension | GPT-3 (2020) | Gemma 4 (2026) |
|---|---|---|
| Params vs compute | 175B dense, all active | up to 31B dense or 26B MoE (~4B active) |
| Positional encoding | learned absolute | RoPE (rotation, relative) |
| Attention | full + banded sparse | sliding-window local + periodic global, GQA |
| Normalization | LayerNorm | RMSNorm + QK-norm |
| Context window | ~2K tokens | 256K tokens |
| Modality | text only | text + image + audio |
| Alignment | base model, few-shot only | instruction-tuned + preference-optimized |

---

## 3.11 — TODO / things to explore next
- [x] Gemma 4 exact MoE config — 128 experts, top-8, 30 layers, ~3.8B active (see MoE section)
- [x] Per-Layer Embeddings (PLE) — per-layer embedding tables stored in flash, up-projected from ~256 dim (see PLE section)
- [x] Multi-Token Prediction (MTP) — drafter + speculative decoding, ~3× faster, lossless (see MTP section)
- [x] How `base` scaling (NTK-aware RoPE) is tuned in practice — raise base 10k→~1M + short long-context fine-tune (see RoPE "Extending context" subsection)
- [x] Memory math: MHA vs GQA KV cache at 100K+ context — formula + worked table (see GQA "KV cache memory math" subsection)

### Ideas for future entries (newer models / techniques)
- Multi-head Latent Attention (MLA, DeepSeek) — compress KV into a latent vector
- MatFormer / nested FFNs, Conditional Parameter Loading, AltUp, LAuReL (Gemma-3n lineage)
- Sparse/clustered LM-head decoding for huge vocabularies
- Diffusion-based text generation (DiffusionGemma)
- Quantization (QAT, AWQ, GPTQ) and its interaction with MoE routing
