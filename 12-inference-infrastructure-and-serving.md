# Chapter 12: Inference Infrastructure — KV Cache, Batching & Serving at Scale

## Why This Chapter Matters

Chapter 9 explained *why* LLMs are fast or slow. This chapter explains what production systems do about it — the engineering that turns a research model into a service handling thousands of concurrent users.

---

## KV Cache: The Most Important Data Structure in LLM Serving

### Quick Recap

During autoregressive decoding, each new token must attend to all previous tokens. Without caching, you'd recompute attention for the entire sequence at every step:

```
Token 1: Compute attention over 1 token
Token 2: Compute attention over 2 tokens (re-doing token 1's computation)
Token 3: Compute attention over 3 tokens (re-doing tokens 1+2)
...
Token 1000: Compute attention over all 1000 tokens
```

The **KV cache** stores the Key and Value vectors for all previously processed tokens in every layer. At each decode step, you only compute K and V for the *new* token, then attend against the cached values.

```
Without KV cache: O(n²) compute for n output tokens
With KV cache:    O(n) compute for n output tokens
```

### KV Cache Memory Math

For a typical 70B model (like Llama 2 70B):
```
Per token per layer: 2 × hidden_dim × sizeof(dtype)
  = 2 × 8192 × 2 bytes (FP16) = 32 KB per token per layer

Across all layers (80 layers):
  = 32 KB × 80 = 2.56 MB per token

For a 4K context sequence:
  = 2.56 MB × 4096 = ~10 GB of KV cache PER REQUEST

For a 128K context:
  = 2.56 MB × 131072 = ~327 GB PER REQUEST
```

This is why long-context models are expensive to serve. The KV cache alone can exceed the GPU memory available for a single request.

---

## KV Cache Management at Scale

When you're serving 1000 concurrent users, each with their own KV cache, memory management becomes the dominant engineering challenge.

### Problem: Memory Fragmentation

Naive allocation:
```
Request A: allocates 4096 slots → uses 2000 → 2096 slots wasted
Request B: allocates 4096 slots → uses 3500 → 596 slots wasted
Request C: arrives → not enough contiguous memory → rejected (even though total free > needed)
```

### Solution: Paged Attention (vLLM)

**Paper:** *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., 2023)

Instead of allocating contiguous memory per sequence, allocate in **pages** (fixed-size blocks):

```
Physical memory: [Page 0][Page 1][Page 2][Page 3][Page 4][Page 5]...

Request A: Pages 0, 3, 5 (non-contiguous, that's fine)
Request B: Pages 1, 2, 4
Request C: New pages allocated as needed

Each page holds KV states for a fixed number of tokens (e.g., 16 tokens per page).
```

**Benefits:**
- Near-zero memory waste (allocate exactly what you need)
- No fragmentation (pages don't need to be contiguous)
- Enables memory sharing (same prefix → share pages via copy-on-write)
- 2-4x more concurrent requests vs. naive allocation

### KV Cache Eviction Strategies

When memory is full, which KV entries do you evict?

| Strategy | How it works | Tradeoff |
|---|---|---|
| **LRU (Least Recently Used)** | Evict tokens attended least recently | Simple, but some old tokens are important |
| **Attention-based** | Evict tokens that received lowest attention scores | Better quality, more compute to track |
| **Window + Sink** | Keep first N tokens + last M tokens, evict middle | Fast, works well for many tasks |
| **H2O (Heavy Hitters)** | Keep tokens that consistently get high attention | Best quality, most complex |

**StreamingLLM** (Xiao et al., 2023) showed that keeping just the first few tokens ("attention sinks") plus the most recent tokens preserves most of the model's quality even for very long contexts. The middle tokens contribute relatively little.

### KV Cache Reuse Across Requests

If multiple requests share a common prefix (system prompt, shared context):

```
System prompt (1000 tokens) — shared across ALL requests
  → Compute KV cache ONCE
  → All concurrent requests reference the same cached KV pages

Request A: [shared prefix pages] + [unique suffix pages A]
Request B: [shared prefix pages] + [unique suffix pages B]
```

This is what **prompt caching** (Chapter 11) looks like at the infrastructure level. The provider precomputes and stores KV states for common prefixes.

---

## Prefill vs. Decode: Two Different Optimization Problems

### Prefill Phase: Compute-Bound

Processing the input prompt is **embarrassingly parallel** — all input tokens are processed simultaneously through the transformer layers.

```
Input: 5000 tokens
GPU utilization: 90%+ (large matrix multiplications saturate compute)
Bottleneck: raw FLOPS (floating point operations per second)
```

**Optimization strategies for prefill:**
- Tensor parallelism (split matrices across GPUs)
- Flash Attention (memory-efficient attention algorithm)
- Quantized prefill (use lower precision during this phase)

### Decode Phase: Memory-Bound

Generating one token at a time means tiny matrix multiplications that don't saturate the GPU:

```
Output: 1 token at a time
GPU utilization: 10-30% (tiny matmul, waiting for memory reads)
Bottleneck: memory BANDWIDTH (reading model weights + KV cache)
```

**The decode bottleneck is reading data, not computing with it.** Each decode step reads the entire model's weights from GPU memory (HBM) to compute units, but only processes a single token. The compute units are starved waiting for data.

**Optimization strategies for decode:**
- Batching (amortize weight reads across many sequences)
- Speculative decoding (process multiple draft tokens in one pass)
- Quantization (smaller weights = faster reads)
- Paged KV cache (efficient memory access patterns)

### Why They Optimize Differently

| Dimension | Prefill | Decode |
|---|---|---|
| Parallelism | High (all tokens at once) | Low (one token at a time) |
| GPU utilization | High | Low |
| Bottleneck | Compute (FLOPS) | Memory bandwidth |
| Batching helps? | Somewhat | Dramatically |
| Quantization helps? | Moderate | Significant (less memory to read) |
| Latency metric | Time to first token (TTFT) | Tokens per second (TPS) |

### Disaggregated Prefill and Decode

Some serving systems now separate prefill and decode onto different hardware:

```
Prefill cluster: GPUs optimized for compute (high FLOPS)
  → Process input prompts, generate KV caches
  → Transfer KV caches to decode cluster

Decode cluster: GPUs optimized for memory bandwidth
  → Generate output tokens using cached KV states
  → Optimized for high throughput batching
```

This is called **prefill-decode disaggregation** and allows each phase to use hardware best suited to its bottleneck.

---

## Continuous Batching

### The Problem with Static Batching

Traditional batching groups requests and processes them together:

```
Static batch: [Request A (50 tokens), Request B (200 tokens), Request C (10 tokens)]

Request C finishes after 10 tokens → sits idle for 190 more tokens
Request A finishes after 50 tokens → sits idle for 150 more tokens
Only Request B uses the full batch time
```

GPU utilization drops as shorter requests finish and their slots sit empty.

### Continuous Batching (Iteration-Level Scheduling)

Instead of waiting for the entire batch to complete, swap finished requests out and new requests in **at every decode step:**

```
Step 1:  [A, B, C, D] — all active
Step 10: [A, B, _, D] — C finished, slot available
Step 11: [A, B, E, D] — E (new request) takes C's slot immediately
Step 50: [_, B, E, D] — A finished, slot available
Step 51: [F, B, E, D] — F joins the batch
```

**Result:** Near-100% GPU utilization. No request sits idle. The batch is always full.

**Implemented in:** vLLM, TensorRT-LLM, TGI (Text Generation Inference)

### Preemption and Priority

In production, some requests are more important than others:

```
Priority 1 (real-time chat): Low latency required
Priority 2 (batch processing): High throughput, latency flexible
Priority 3 (background tasks): Best-effort

When a P1 request arrives and the batch is full:
  → Preempt a P3 request (swap its KV cache to CPU, resume later)
  → Schedule P1 immediately
```

This is **preemptive scheduling** — the same concept as operating system process scheduling, applied to LLM inference.

---

## Speculative Decoding

### The Core Idea

A small, fast "draft" model generates several candidate tokens. The large "target" model then verifies them all in a single forward pass (which is just as fast as generating one token, since verification is parallel like prefill).

```
Draft model (1B): Generates tokens "The quick brown fox" (4 tokens, very fast)
Target model (70B): Verifies all 4 in one pass
  → Accepts "The quick brown" (3 tokens match)
  → Rejects "fox" (target would have said "dog")
  → Generates "dog" as the corrected token

Net result: 4 tokens generated in ~1.5 forward passes of the target model
  (vs. 4 forward passes without speculation)
```

### When Speculative Decoding Works

| Condition | Effectiveness |
|---|---|
| Draft model closely matches target | High (most tokens accepted) |
| Easy/predictable text (code, structured) | High (drafts are usually right) |
| Creative/novel text | Low (drafts frequently rejected) |
| Long sequences | Compounds savings |

### Practical Speedup

Typical: **2-3x** tokens per second improvement for well-matched draft models.

**Variants:**
- **Self-speculative decoding:** Use early exit from the same model as the draft
- **Medusa:** Add multiple prediction heads to the target model itself
- **Eagle:** Trained draft heads that share the target model's KV cache
- **Lookahead decoding:** Use Jacobi iteration to decode multiple positions simultaneously

---

## Quantization for Serving

### Why Quantize?

The decode phase is memory-bandwidth-bound. Smaller weights = faster reads from GPU memory = faster token generation.

```
FP16 model (70B): 140 GB → needs multiple GPUs
INT8 model (70B):  70 GB → fits on fewer GPUs, 2x faster memory reads
INT4 model (70B):  35 GB → fits on single high-end GPU, 4x faster reads
```

### Quantization Formats

| Format | Bits | Quality Impact | Use Case |
|---|---|---|---|
| **FP16/BF16** | 16 | None (baseline) | Training, highest quality serving |
| **FP8** | 8 | Minimal | H100 native support, good tradeoff |
| **INT8** | 8 | Very small | Standard serving, broad hardware support |
| **INT4 (GPTQ)** | 4 | Small to moderate | Cost-optimized serving |
| **INT4 (AWQ)** | 4 | Small (better than GPTQ) | Activation-aware, preserves quality better |
| **INT4 (GGUF)** | 4 | Varies by method | CPU inference, llama.cpp |
| **INT3/INT2** | 2-3 | Significant | Extreme edge cases only |

### AWQ vs. GPTQ vs. Standard INT8

**GPTQ (GPT Quantization):** Post-training quantization using calibration data. Quantizes weights by minimizing reconstruction error layer-by-layer. Fast inference, some quality loss on hard tasks.

**AWQ (Activation-Aware Quantization):** Observes which weight channels carry the most important activations and protects them (keeps them at higher precision). Typically better quality than GPTQ at the same bit width.

**When quantization hurts quality:**
- Complex reasoning chains (errors compound across tokens)
- Rare/unusual tokens (quantization biases toward common patterns)
- Tasks requiring precise numerical computation
- Low-resource languages (less data during calibration)

**Rule of thumb:** INT8 is almost free (negligible quality loss). INT4 costs 1-3% on benchmarks. Below INT4, quality degrades significantly for general-purpose models.

---

## Distillation vs. Quantization

Two different approaches to making models smaller/faster:

| Dimension | Quantization | Distillation |
|---|---|---|
| What it does | Same model, fewer bits per weight | Smaller model trained to mimic larger one |
| Training needed | Calibration only (minutes) | Full training (hours/days) |
| Quality loss | Small (INT8) to moderate (INT4) | Depends on student size |
| Speed gain | 2-4x (memory bandwidth) | Arbitrary (smaller model = faster) |
| Flexibility | Any model post-hoc | Need training pipeline |
| Combination | Can quantize a distilled model | ✓ Common in practice |

**When to choose which:**
- **Quick deployment:** Quantize your existing model (minutes of work)
- **Maximum compression:** Distill into a small model, then quantize it
- **Task-specific optimization:** Fine-tune a small model on your task (often better than distilling a general model)

---

## Putting It All Together: A Production Serving Stack

```
┌──────────────────────────────────────────────────────────────┐
│                      Load Balancer                             │
├──────────────────────────────────────────────────────────────┤
│                    Request Router                              │
│  (route by model, priority, context length)                   │
├─────────────────────────┬────────────────────────────────────┤
│   Prefill Workers       │        Decode Workers               │
│   (compute-optimized)   │     (bandwidth-optimized)           │
│                         │                                     │
│   Flash Attention       │     Continuous Batching             │
│   Tensor Parallelism    │     Paged Attention                 │
│   Chunked Prefill       │     Speculative Decoding            │
│                         │     Quantized Weights (INT8/FP8)    │
├─────────────────────────┴────────────────────────────────────┤
│                    KV Cache Manager                            │
│  (Paged allocation, eviction, prefix sharing, CPU offload)    │
├──────────────────────────────────────────────────────────────┤
│                    GPU Cluster                                 │
│  (H100s with NVLink, high-bandwidth interconnect)             │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Metrics for Inference Infrastructure

| Metric | What it measures | Good target |
|---|---|---|
| TTFT (Time to First Token) | Prefill speed + queue time | <500ms for chat |
| TPS (Tokens per Second) | Decode speed per request | >50 TPS for chat |
| Throughput | Total tokens/sec across all requests | Maximize |
| GPU utilization | How well hardware is used | >80% |
| KV cache hit rate | Prefix reuse effectiveness | >60% for API workloads |
| Request rejection rate | Memory pressure indicator | <1% |
| P99 latency | Tail latency worst case | <3x median |

---

## Key Takeaways

1. **KV cache is the dominant memory consumer** — for long contexts, it can exceed the model weights themselves
2. **Paged Attention solves memory fragmentation** — 2-4x more concurrent users via virtual memory for KV cache
3. **Prefill is compute-bound, decode is memory-bound** — they need different optimization strategies
4. **Continuous batching keeps GPUs full** — swap requests in/out at every decode step
5. **Speculative decoding trades draft model compute for 2-3x speedup** — works best when text is predictable
6. **INT8 quantization is nearly free** — INT4 costs some quality but enables single-GPU serving of large models
7. **Production stacks combine all of these** — paged attention + continuous batching + speculative decoding + quantization is standard

---

## Key Papers & Resources

- **PagedAttention / vLLM** (Kwon et al., 2023) — paged KV cache management
- **Flash Attention 2** (Dao, 2023) — memory-efficient exact attention
- **Speculative Decoding** (Leviathan et al., 2022; Chen et al., 2023) — draft-verify acceleration
- **AWQ: Activation-aware Weight Quantization** (Lin et al., 2023) — quality-preserving INT4
- **StreamingLLM** (Xiao et al., 2023) — infinite-length generation with attention sinks
- **Efficient Memory Management for LLM Serving** (vLLM team) — continuous batching + paged attention

---

## What's Next

The model generated output — but is that output actually *correct*? Structured outputs, function calling, and schema validation are where generation meets deterministic contracts. Chapter 13.
