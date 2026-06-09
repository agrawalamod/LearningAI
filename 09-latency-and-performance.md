# Chapter 9: Latency & Performance — What Makes LLMs Fast or Slow

## Why This Matters for You

You said you're building a smaller language model. Understanding latency lets you make informed design choices: how many parameters, how long a context, what hardware to target.

---

## The Two Phases of LLM Inference

LLM inference has two distinct phases with completely different performance characteristics:

### Phase 1: Prefill (Processing the Input)

The model reads your entire input prompt at once.

```
Input: "Explain quantum computing in simple terms" (7 tokens)

What happens: All 7 tokens are processed SIMULTANEOUSLY through all layers.
This is a single forward pass — parallel computation.
```

**Latency of prefill:** Proportional to input length, but parallelized on GPU.
- 100 tokens → fast (milliseconds)
- 10,000 tokens → slower (hundreds of milliseconds)
- 100,000 tokens → noticeable (seconds)

### Phase 2: Decode (Generating the Output)

The model generates tokens ONE AT A TIME.

```
Output: "Quantum computing uses quantum bits..." 

Token 1: "Quantum"    → one forward pass through all layers
Token 2: "computing"  → one forward pass through all layers
Token 3: "uses"       → one forward pass through all layers
...
```

**Latency of decode:** Proportional to OUTPUT length. Each token requires a full forward pass.

---

## The Key Insight: Output Length Dominates Latency

```
Short input + short output = fast
Long input + short output = moderate (prefill cost)
Short input + long output = slow (many decode steps)
Long input + long output = slowest
```

**This is why Claude "thinking" takes time:** Extended thinking generates hundreds or thousands of tokens before giving you the answer. Each thinking token is one forward pass through the entire model.

A simple question like "How are you?" might generate:
```
[Thinking: 50 tokens of reasoning about appropriate response]
[Answer: 10 tokens]
= 60 forward passes total
```

A hard math problem might generate:
```
[Thinking: 2000 tokens of step-by-step reasoning]
[Answer: 50 tokens]
= 2050 forward passes total ← THIS is why it's slow
```

---

## What Determines the Speed of Each Forward Pass?

### Factor 1: Model Size (Parameters)

More parameters = more computation per forward pass.

```
7B parameters  → ~14 TFLOPS per forward pass → fast on consumer GPU
70B parameters → ~140 TFLOPS per forward pass → needs powerful hardware
405B parameters → ~810 TFLOPS per forward pass → needs multiple GPUs
```

**Rule of thumb:** A 70B model is ~10x slower than a 7B model per token (all else equal).

### Factor 2: Architecture Depth and Width

| Dimension | Effect on latency |
|---|---|
| **More layers (depth)** | More sequential computation → directly slower |
| **Wider layers (hidden dim)** | More parallel computation → slower but GPU-friendly |
| **More attention heads** | More parallel work per layer → moderate impact |
| **Vocabulary size** | Affects final projection → usually minor |

### Factor 3: Context Length (KV Cache)

At each decode step, the model must attend to ALL previous tokens. This requires storing key-value pairs for every previous token in every layer — the **KV cache**.

```
Context of 1000 tokens: Small KV cache → fast attention
Context of 100,000 tokens: Huge KV cache → slow attention, lots of memory
```

**This is where input length matters:** More input tokens = larger KV cache = each decode step is slightly slower because attention has more to attend to.

But the effect is sublinear — going from 100 to 1000 input tokens does NOT make each output token 10x slower. It's more like 1.2x slower (depends on architecture).

### Factor 4: Hardware

| Hardware | Typical speed (tokens/sec for 7B model) |
|---|---|
| CPU (laptop) | 5-20 tokens/sec |
| Consumer GPU (RTX 4090) | 50-150 tokens/sec |
| Server GPU (A100) | 100-300 tokens/sec |
| Server GPU (H100) | 200-500 tokens/sec |
| Multiple GPUs | Scales further |

---

## Time to First Token (TTFT) vs Tokens Per Second (TPS)

These are the two key latency metrics:

### TTFT (Time to First Token)
How long until the first output token appears.

```
TTFT = Prefill time (processing input) + first decode step
```

This is what you feel as "thinking time." For models with extended thinking, it includes all the reasoning tokens generated before the visible answer starts.

### TPS (Tokens Per Second)
How fast output tokens appear once they start flowing.

```
After first token, each subsequent token appears every 1/TPS seconds.
At 50 TPS: one new token every 20ms (feels like smooth typing)
At 10 TPS: one new token every 100ms (noticeable delay between words)
```

---

## Why 5 Tokens vs 5000 Tokens Input (Your Question)

**Direct answer:** It matters, but less than you'd think.

```
5 input tokens → TTFT: ~50ms, each output token: ~20ms
5000 input tokens → TTFT: ~500ms, each output token: ~25ms
```

The prefill is proportionally longer, and each decode step is slightly slower (bigger KV cache). But the dominant factor for total response time is **how many output tokens are generated**, not input length.

**Exception:** At very long contexts (100K+ tokens), the KV cache becomes a significant bottleneck for both memory and compute.

---

## Batched Inference: Why It's Different

When serving many users simultaneously:

**Single request:** GPU is often underutilized (the matrix multiplications don't saturate the hardware)

**Batched requests:** Multiple requests processed together → GPU is fully utilized → higher throughput

```
Single request:  50 tokens/sec, GPU at 30% utilization
Batch of 32:     40 tokens/sec per request, but 1280 tokens/sec total, GPU at 95%
```

**Key tradeoff:**
- Batching increases *throughput* (total tokens/sec across all users)
- But can increase *latency* per individual request (waiting to be batched, sharing compute)

This is why API services sometimes feel faster or slower depending on load.

---

## Design Choices for Your Smaller Model

Since you're building a smaller model, here are the tradeoffs:

### Model Size
| Size | Quality | Speed | Hardware needed |
|---|---|---|---|
| 1-3B | Basic tasks | Very fast | Phone/laptop CPU |
| 7B | Good for many tasks | Fast | Consumer GPU |
| 13B | Strong for focused tasks | Moderate | Good GPU (24GB VRAM) |
| 70B | Near-frontier for open models | Slow | Multiple GPUs / cloud |

### Context Length
| Max context | Memory cost | Use case |
|---|---|---|
| 2K tokens | Low | Simple Q&A, short conversations |
| 8K tokens | Moderate | Most conversations, short documents |
| 32K tokens | High | Long documents, complex tasks |
| 128K+ tokens | Very high | Full codebases, books |

### Width vs Depth
- **Deeper models** (more layers): Better at complex reasoning chains
- **Wider models** (larger hidden dim): Better at storing knowledge, more parallelizable

For a small model focused on speed:
- Fewer layers (24-32) with moderate width
- Short context (4K-8K) unless you specifically need long context
- Efficient attention (grouped query attention, sliding window)

---

## Why Claude "Thinks" Before Simple Questions

You specifically asked about this. Multiple reasons:

1. **Extended thinking is always-on:** The model generates thinking tokens for every query (even simple ones), because the policy is trained to reason before responding.

2. **Safety reasoning:** Even "How are you?" might trigger brief reasoning about context, appropriateness, etc.

3. **Network latency:** Some of what feels like "thinking time" is actually network round-trip between your device and the server.

4. **Batching queues:** Your request might wait briefly to be batched with others.

5. **Model size:** Claude is a very large model (likely hundreds of billions of parameters). Even with fast hardware, each forward pass takes time, and extended thinking generates many passes.

---

## Optimization Techniques (How Production Models Stay Fast)

| Technique | What it does | Speed gain |
|---|---|---|
| **KV Cache** | Store previous attention keys/values instead of recomputing | 10-100x for long sequences |
| **Quantization** | Use 4-bit or 8-bit numbers instead of 16-bit | 2-4x speed, some quality loss |
| **Speculative decoding** | Small model drafts tokens, big model verifies in batch | 2-3x speed |
| **Flash Attention** | Memory-efficient attention algorithm | 2-4x for long contexts |
| **Tensor parallelism** | Split model across multiple GPUs | Scales with GPU count |
| **Continuous batching** | Add/remove requests from batch dynamically | Higher throughput |
| **Paged attention (vLLM)** | Efficient KV cache memory management | More concurrent requests |

---

## Summary: What Defines LLM Latency

```
Total latency = Prefill time + (Number of output tokens × Time per token)

Where:
  Prefill time ∝ input_length × model_size / hardware_speed
  Time per token ∝ model_size × f(context_length) / hardware_speed
  
And: output_length >> input_length usually dominates
```

---

## Key Takeaways

1. **Output length dominates latency** — more generated tokens = slower
2. **Input length matters but less** — affects prefill and slightly slows each decode step
3. **Model size is the biggest single factor** — 70B is ~10x slower than 7B
4. **"Thinking time" = generating reasoning tokens** — each one costs a full forward pass
5. **Batching helps throughput but not individual latency**
6. **For your small model:** Choose size based on task needs, optimize context length, use quantization for deployment

---

## Key Papers & Resources

- **Efficient Transformers: A Survey** (Tay et al., 2022)
- **FlashAttention: Fast and Memory-Efficient Exact Attention** (Dao et al., 2022)
- **Scaling Data-Constrained Language Models** (Muennighoff et al., 2023)
- **LLM Inference Performance Engineering** (various blog posts from vLLM, TensorRT-LLM teams)

---

## What's Next

Final chapter: the theoretical limits of what these models can achieve, and whether we can trust them in production. Chapter 10.
