# Chapter 4 — DeepSeek V4-Pro (2026): From GPT-3 to the Efficiency Frontier

DeepSeek V4-Pro was released April 24, 2026 under the MIT license. It is a
1.6-trillion-parameter Mixture-of-Experts model with 49B parameters active per token,
a 1-million-token context window, and a new hybrid attention architecture that reduces
per-token inference FLOPs by ~73% compared to V3. It achieves 80.6% on SWE-bench
Verified and 93.5 on LiveCodeBench — the highest scores of any publicly available model
at release.

This chapter covers the full lineage from GPT-3 to V4-Pro, treating each version as
a solution to the specific problems the previous version left unsolved.

---

## 4.1 — The DeepSeek lineage at a glance

| Version | Released | Total params | Active params | Key architectural leap |
|---|---|---|---|---|
| GPT-3 (baseline) | 2020 | 175B | 175B (dense) | — baseline dense transformer |
| DeepSeek V1 | Jan 2024 | 67B | 67B (dense) | Competitive open dense model |
| DeepSeek V2 | May 2024 | 236B | 21B | **MoE + MLA** — 42× efficiency vs dense V1 |
| DeepSeek V3 | Dec 2024 | 671B | 37B | Aux-loss-free MoE routing, MTP, FP8 training |
| DeepSeek V3.2 | ~May 2025 | ~671B | ~37B | Incremental: better reasoning, tool use |
| DeepSeek V4-Pro | Apr 2026 | **1.6T** | **49B** | Hybrid CSA/HCA attention, mHC, hash routing, 1M ctx |

The theme across all versions: **more total knowledge, less compute per token**. Each
generation keeps active params roughly flat while dramatically increasing total capacity
and reducing the cost of using it.

---

## 4.2 — GPT-3 (2020): the baseline problems

Chapter 1 covered GPT-3 in full. The relevant limitations that DeepSeek's lineage
addresses:

**1. Dense — all 175B params fire for every token.** Compute scales with total params,
making large models expensive. There is no way to have a 500B-param model without a
500B-param compute cost.

**2. Full attention — O(n²) cost.** Every token attends to every past token. At 2,048
tokens this is manageable. At 100K tokens it is prohibitive.

**3. Fixed KV cache — O(n) memory per layer per head.** As context grows, the cache
grows with it. GPT-3's 96 heads × 96 layers makes this very large.

**4. Context window capped at 2,048 tokens.** Absolute positional embeddings cannot
extrapolate beyond the training length. No mechanism for longer sequences.

**5. Single next-token prediction.** One forward pass → one token at inference, N passes
for N tokens, strictly sequential.

Each of these is a distinct bottleneck. DeepSeek's four generations address them in order.

---

## 4.3 — DeepSeek V2 (May 2024): MoE and MLA — the two foundational moves

V2 made two architectural changes that define the entire subsequent lineage.

### MoE — decoupling knowledge from compute

V2 replaced the dense FFN with a **Mixture-of-Experts** FFN. Instead of one large
FFN that always runs, there are many expert FFNs and a router that picks a small
subset per token.

```
Dense FFN (GPT-3):    all 175B params fire → cost ∝ total params
MoE FFN (V2):         236B total params, only 21B fire per token
                      → 11× more total knowledge, same per-token cost as a 21B dense model
```

This breaks the GPT-3 coupling between "how much the model knows" (total params) and
"how much it costs to use" (active params). V2 effectively demonstrated that you can
build a very large model that runs like a small one.

### MLA — Multi-Head Latent Attention (solving KV cache memory)

V2 introduced **MLA** to replace standard multi-head attention. The problem with
standard MHA: the KV cache size is:

```
cache = 2 × n_layers × n_heads × head_dim × seq_len
```

At 128K tokens (V2's target context) this is enormous. MLA compresses the KV cache
by projecting K and V into a **low-dimensional latent vector** rather than storing
the full head-sized K and V separately:

```
Standard MHA:   cache per token = 2 × n_heads × head_dim   (large)
MLA:            cache per token = 1 latent vector (much smaller)
                K and V are reconstructed from the latent at attention time
```

The latent vector is smaller than the full K/V by a large factor, so the cache shrinks
dramatically — enabling V2 to run at 128K context on hardware that couldn't afford
standard MHA at that length.

**Result:** V2 with 236B total / 21B active costs 42× less than an equivalent dense
model. Context extended to 128K via YaRN (RoPE scaling).

---

## 4.4 — DeepSeek V3 (December 2024): scaling MoE, removing its main flaw

V3 pushed V2's MoE to 671B total / 37B active and fixed MoE's most significant
training problem: **load imbalance**.

### The MoE load balancing problem

In standard MoE, a router assigns tokens to experts. If the router learns to always
pick the same few experts (because they were slightly better early in training), most
experts get few gradient updates and never improve — the "expert collapse" problem.
Prior solutions used an **auxiliary loss** that penalized imbalanced routing, but this
auxiliary loss competed with the main next-token loss and hurt quality.

### V3's fix: auxiliary-loss-free load balancing

V3 used a **bias term per expert** that is adjusted dynamically during training to keep
expert load balanced, without adding any term to the loss function itself. The bias
nudges the router toward underutilized experts without polluting the gradient signal.
Result: balanced routing with no quality penalty.

### Other V3 contributions

- **Multi-Token Prediction (MTP)** — extra lightweight heads predict tokens t+2, t+3
  during training, providing a stronger gradient signal. At inference, repurposed as a
  speculative decoding drafter → ~3× faster generation (covered in Chapter 3 §3.9).
- **FP8 mixed-precision training** — trained in 8-bit floating point with a 32-bit
  master copy, reducing training memory and accelerating computation.
- **Auxiliary-loss-free routing** enabled training a much larger expert pool (256
  experts, top-8 routing) without collapse.

**Result:** V3 at 671B / 37B active became the strongest open-weight model at release,
matching GPT-4 class performance at a fraction of inference cost.

---

## 4.5 — DeepSeek V4-Pro (April 2026): the 1M context problem

V3 was strong, but it inherited one fundamental problem from MLA: **at 1M tokens, even
the compressed MLA KV cache becomes a bottleneck**, and the attention computation
(scoring the query against all 1M past keys) is expensive. V4-Pro's entire architecture
is organized around making 1M-token context economically viable.

### Reference dimensions (V4-Pro)

| Symbol | Value |
|---|---|
| Total params | ~1.6T |
| Active params per token | ~49B |
| Context window | 1,000,000 tokens |
| Max output | 384,000 tokens |
| Precision | FP8 mixed + FP4/FP8 mixed |
| License | MIT |

---

## 4.6 — New technique 1: Hybrid attention (CSA + HCA)

V4-Pro replaces V3's uniform MLA with a **hybrid** of two attention types assigned to
different layers:

### CSA — Compressed Sparse Attention

Applied to most layers. CSA compresses the query and key vectors before computing
scores, reducing the dimensionality of the attention computation. Think of it as
applying a learned down-projection to Q and K before the dot product, then projecting
back — similar to how MLA compressed the KV cache, but now applied to the score
computation itself.

```
Standard attention score:  q (128-dim) · k (128-dim)   expensive at 1M keys
CSA score:                 compress(q) · compress(k)    cheaper; slightly lossy
```

Most layers use CSA — they handle local patterns efficiently at low cost.

### HCA — Heavily Compressed Attention

Applied to a small number of layers (globally spaced). HCA compresses even more
aggressively than CSA — essentially a very narrow bottleneck that forces the attention
to be highly selective. These layers handle the long-range, global dependencies that
the cheap CSA layers might miss.

```
Layer distribution (approximate):
  ~80% of layers: CSA (cheap, local/medium range)
  ~20% of layers: HCA (expensive but aggressive, long range)
```

This hybrid mirrors what Gemma 4 does with sliding-window vs global attention
(Chapter 3 §3.4), but instead of restricting the *token range*, it restricts the
*representation dimensionality*.

**Result at 1M context:** V4-Pro requires only **27% of the single-token inference
FLOPs** and **10% of the KV cache** of V3 at the same context length.

```
KV cache at 1M tokens:
  V3 (MLA):       X GB
  V4-Pro (CSA):   0.10 × X GB    ← 10% of V3's cache
```

---

## 4.7 — New technique 2: mHC — Manifold-Constrained Hyper-Connections

Every model since GPT-3 uses **residual connections**: `e' = e + sublayer(e)`. The
`+` adds the sublayer's output to the unchanged input (the "skip connection"). This is
what Chapter 1 §1.5 calls the "residual stream" — a fixed identity path that carries
information forward unchanged while sublayers write deltas.

V4-Pro replaces the plain `+` with **Manifold-Constrained Hyper-Connections (mHC)**:

```
Standard residual:   e' = e + sublayer(e)
mHC:                 e' = α·e + β·sublayer(e)   where α and β are learned per-layer
                                                  scalars constrained to a manifold
```

**What this changes:**
- **`α` and `β` are learnable** — the model can decide how much to preserve the existing
  representation vs how much to accept the new sublayer's update, per layer.
- **Manifold constraint** — α and β are not free scalars; they are constrained to lie on
  a curved surface (a manifold) that prevents degenerate solutions (e.g. α=0, β=0 which
  would kill the residual stream).

In GPT-3 (and V3), `α = β = 1` always — the identity path and the update are always
added with equal weight. mHC lets early layers preserve more of the original embedding
(higher α, lower β) and later layers accept more radical updates (lower α, higher β),
adapting the information flow to what each layer actually needs.

**Practical effect:** Reportedly reduces training instability and improves quality per
parameter, especially at the very large scales V4-Pro operates at.

---

## 4.8 — New technique 3: Hash routing for early MoE layers

Standard MoE routing (V2, V3) uses a **learned router** — a small neural network that,
given a token, decides which experts to send it to. This works well but has a cold-start
problem: in the very first MoE layers (early in the block stack), the token
representations are still raw and uninformative. The router has little signal to route
well, so early routing is essentially random — wasted computation.

V4-Pro uses a **static hash routing** for the first few MoE layers:

```
Early layers (hash routing):
  expert_id = hash(token_id) % n_experts    ← determined by token identity alone
  No learned router, no cold-start problem

Later layers (learned routing):
  expert_id = router(token_representation)  ← standard learned routing
```

The hash is computed from the token's integer ID, not its embedding — so the routing
decision is fixed, deterministic, and requires no learned parameters for those layers.
By the time the representations are rich enough for learned routing to be meaningful,
the model switches to it.

**Practical effect:** More stable training in early layers, better expert utilization
from the very first tokens of each sequence.

---

## 4.9 — The technology gap: GPT-3 to V4-Pro in one table

| Problem | GPT-3 answer | DeepSeek's solution (version) |
|---|---|---|
| Dense compute cost | All 175B fire per token | MoE: 49B active from 1.6T total (V2+) |
| KV cache at long context | Full K,V per head per token | MLA: compressed latent (V2), CSA/HCA: 10% of V3's cache (V4) |
| Context window hard limit | 2,048 (absolute position table) | YaRN RoPE scaling → 128K (V2), 1M (V4) |
| Full O(n²) attention | All tokens attend to all tokens | CSA + HCA hybrid: compressed dims, selective global layers (V4) |
| MoE training instability | Auxiliary loss hurts quality | Aux-loss-free bias adjustment (V3), hash routing for early layers (V4) |
| Single-token generation | 1 pass → 1 token | MTP + speculative decoding → ~3× (V3+) |
| Fixed residual stream | α=β=1 always | mHC: learned α, β per layer, manifold-constrained (V4) |
| Reasoning mode separate | Base model only | Hybrid thinking mode built in (V4) |

---

## 4.10 — Why V4-Pro matters: cost at scale

The practical consequence of all the above can be summarized in one number:

```
V4-Pro requires ~27% of V3's FLOPs per token at 1M-token context
```

This is the difference between 1M-token inference being economically feasible or not.
At 73% fewer FLOPs per token:
- A provider that could serve 100 users simultaneously on V3 can serve ~370 on V4-Pro
- A developer who found V3 unaffordable for long-context work can now use V4-Pro

The 1.6T total params (vs V3's 671B) means more knowledge capacity, but the 49B active
params means the per-token cost is only modestly higher than V3's 37B — knowledge grew
2.4× but compute grew only ~1.3×.

---

## 4.11 — Cheat sheet: GPT-3 → V4-Pro

| Dimension | GPT-3 (2020) | DeepSeek V2 (2024) | DeepSeek V3 (2024) | DeepSeek V4-Pro (2026) |
|---|---|---|---|---|
| Total / active params | 175B / 175B | 236B / 21B | 671B / 37B | 1.6T / 49B |
| Context window | 2,048 | 128K | 128K | **1M** |
| Attention | Full MHA | MLA | MLA | **CSA + HCA hybrid** |
| Positional encoding | Absolute table | RoPE + YaRN | RoPE + YaRN | RoPE + YaRN |
| FFN / MoE | Dense FFN | MoE (fine-grained) | MoE (aux-loss-free) | MoE (hash early + learned) |
| Residual connections | Fixed `e + delta` | Fixed | Fixed | **mHC (learned α, β)** |
| Multi-token prediction | None | None | MTP heads | MTP heads |
| Reasoning mode | None | None | Separate R-series | **Built-in hybrid thinking** |
| Training precision | FP16 | BF16 | FP8 | FP8 + FP4/FP8 mixed |
| Open weights | No | Yes | Yes (MIT) | Yes (MIT) |

---

## 4.12 — What V4-Pro does NOT change from prior chapters

Understanding what stayed the same is as important as what changed:

- **The 96-block sequential structure** — blocks still run in sequence, one after another.
  mHC changes how residuals are added but not the sequential dependency.
- **Token-by-token autoregressive generation** — still one position at a time at
  inference (MTP's speculative decoding provides speedup but not a fundamental change
  to the AR paradigm).
- **Backprop + AdamW training** — still trained with gradient descent, same basic loop
  as Chapter 2.
- **The Q/K/V attention mechanism** — CSA and HCA are modifications of Q/K/V attention
  (compressed dimensions, selective layers), not a replacement of the fundamental
  mechanism.
- **Causal masking** — still required during training/prefill; still a no-op during
  decoding (Chapter 1 §1.3 step 5).

The GPT-3 forward pass you learned in Chapter 1 is still the skeleton. V4-Pro is that
skeleton with six specific joints replaced by stronger, cheaper parts.

---

*Sources: DeepSeek V4 technical report (April 2026), Hugging Face model documentation,
[deepinfra.com V4-Pro overview](https://deepinfra.com/blog/deepseek-v4-pro-model-overview),
[artificialanalysis.ai V4 analysis](https://artificialanalysis.ai/articles/deepseek-is-back-among-the-leading-open-weights-models-with-v4-pro-and-v4-flash).
Architecture figures paraphrased from public technical documentation.*
