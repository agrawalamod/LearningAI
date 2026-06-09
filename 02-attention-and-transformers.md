# Chapter 2: What Is Attention? What Are Transformers?

## The Problem Before Transformers

Before 2017, language models used **Recurrent Neural Networks (RNNs)** — specifically LSTMs and GRUs. These processed text one word at a time, left to right, like reading a sentence letter by letter.

**The problems:**
1. **Sequential bottleneck** — You can't parallelize. Word 50 must wait for words 1–49 to be processed first.
2. **Forgetting** — By the time the model reaches word 100, it has largely forgotten what word 5 said. Information "decays" over distance.
3. **No direct connections** — If word 3 and word 97 are related (e.g., a pronoun and its referent), the information must survive through 94 sequential steps.

**Example of the problem:**

```
"The cat, which had been sitting on the windowsill watching birds all 
morning while the rain poured down outside, finally jumped."
```

An RNN processing "jumped" has to remember "cat" across ~20 words of intervening text. In practice, this is hard.

---

## The Key Insight: Attention

**Attention** answers a simple question: *"When processing this word, which other words in the sentence should I look at?"*

Instead of passing information sequentially through a chain, attention lets **every word look directly at every other word**.

### Intuitive Example

```
"The animal didn't cross the street because it was too tired."
```

When processing the word **"it"**, which word should the model attend to?
- "animal" ✓ (because "it" refers to the animal)
- "street" ✗ (not what "it" refers to)

Attention learns to assign a **high weight** to "animal" and a **low weight** to "street" when processing "it".

---

## How Attention Works (The Mechanics)

Attention uses three vectors for each token: **Query (Q)**, **Key (K)**, and **Value (V)**.

Think of it like a search engine:
- **Query** = "What am I looking for?" (the current word's question)
- **Key** = "What do I contain?" (each other word's label)
- **Value** = "What information do I actually carry?" (each other word's content)

### Step-by-step:

1. Each token computes its own Q, K, and V vectors (by multiplying its embedding by learned matrices)
2. To decide how much to attend to another token, compute: `score = Q · K` (dot product)
3. Normalize scores with softmax (so they sum to 1)
4. Multiply each Value by its attention score and sum them up

### Simplified Numerical Example

Say we have three tokens: ["The", "cat", "sat"]

Processing "sat", its Query asks: "Who did the sitting?"

```
Attention scores (before softmax):
  "The" → 0.1  (not very relevant)
  "cat" → 0.8  (very relevant — the cat is doing the sitting)
  "sat" → 0.3  (somewhat relevant — self-reference)

After softmax normalization:
  "The" → 0.08
  "cat" → 0.62
  "sat" → 0.30

Final output for "sat" = 0.08 × Value("The") + 0.62 × Value("cat") + 0.30 × Value("sat")
```

So "sat" now carries information that is heavily influenced by "cat" — it has learned **who** is doing the sitting.

---

## Multi-Head Attention: Looking at Multiple Things at Once

A single attention mechanism can only capture one type of relationship. But language has many simultaneous relationships:
- Syntactic: subject ↔ verb
- Semantic: pronoun ↔ referent
- Positional: adjacent words
- Topical: words about the same concept

**Multi-head attention** runs multiple attention mechanisms in parallel (typically 12–96 "heads"), each learning to capture different relationships.

```
Head 1 might learn: subject-verb agreement
Head 2 might learn: adjective-noun connections
Head 3 might learn: coreference (pronouns → nouns)
Head 4 might learn: position-based patterns
...
```

Their outputs are concatenated and projected back to the model dimension.

---

## The Transformer Architecture

The **Transformer** (Vaswani et al., 2017) is the architecture that uses attention as its core mechanism. It replaced RNNs entirely.

### A Single Transformer Block

```
Input tokens (as vectors)
       │
       ▼
┌─────────────────┐
│  Multi-Head     │ ← Each token attends to all other tokens
│  Self-Attention │
└────────┬────────┘
         │ + Residual connection + Layer Norm
         ▼
┌─────────────────┐
│  Feed-Forward   │ ← Each token processed independently
│  Network (FFN)  │   (2-layer neural net, adds "thinking")
└────────┬────────┘
         │ + Residual connection + Layer Norm
         ▼
Output (same shape as input)
```

### A Full LLM = Many Blocks Stacked

```
Input → [Block 1] → [Block 2] → ... → [Block 96] → Predict next token
```

GPT-3 has 96 blocks. Each block refines the representation. Early blocks capture simple patterns (grammar, syntax). Later blocks capture complex patterns (semantics, reasoning, world knowledge).

---

## Why Attention Changed the Game

| Property | RNNs | Transformers |
|---|---|---|
| Processing | Sequential (slow) | Parallel (fast) |
| Long-range dependencies | Hard (information decays) | Easy (direct connections) |
| Training speed | Slow | Fast (parallelizable on GPUs) |
| Scalability | Plateaus | Scales with compute |

### The concrete impact:

1. **Parallelism**: All tokens can be processed simultaneously during training. This made it practical to train on trillions of tokens.

2. **No forgetting**: Token 1 and token 10,000 are equally connected. The model can reference information from anywhere in its context.

3. **Scalability**: Because training parallelizes across GPUs, you can scale to enormous models. This enabled the "scaling laws" discoveries.

4. **Rich representations**: Multi-head attention captures multiple types of relationships simultaneously, creating richer token representations than RNNs ever could.

---

## Self-Attention vs. Cross-Attention

**Self-attention** (used in GPT-style models): Each token attends to other tokens *in the same sequence*.

**Cross-attention** (used in encoder-decoder models like the original Transformer, T5): Tokens in the output attend to tokens in the input. Used in translation:

```
Encoder input:  "The cat sat on the mat"  (English)
Decoder output: "Le chat s'est assis"     (French)

When generating "chat", the decoder cross-attends to "cat" in the English input.
```

Modern LLMs like GPT and Claude are **decoder-only** (only self-attention), but the original Transformer paper proposed both.

---

## Causal Masking: Why LLMs Can Only Look Backward

In a decoder-only LLM, when predicting the next token, the model can only attend to **previous** tokens, not future ones. This is enforced with a **causal mask**.

```
Sentence: "The cat sat on the mat"

When processing "sat":
  Can attend to: "The", "cat", "sat" ✓
  Cannot attend to: "on", "the", "mat" ✗ (future tokens are masked)
```

This makes intuitive sense: when generating text left-to-right, you can't peek at words you haven't generated yet.

---

## Positional Encoding: How the Model Knows Word Order

Attention itself has no notion of position — "cat sat" and "sat cat" would look identical to raw attention. So transformers add **positional information** to each token.

Modern approaches:
- **Rotary Position Embeddings (RoPE)** — used in LLaMA, encodes relative position into the Q and K vectors
- **Learned positional embeddings** — used in original GPT, a learned vector for each position
- **ALiBi** — adds a bias to attention scores based on distance

---

## Putting It All Together: How a Transformer Processes Text

```
Input: "The cat sat on the"
Task:  Predict next word

1. Tokenize → [The] [cat] [sat] [on] [the]
2. Embed each token → 5 vectors of dimension 4096 (for example)
3. Add positional encoding → now position-aware
4. Pass through Block 1:
   - Self-attention: each token gathers info from previous tokens
   - FFN: each token is independently transformed
5. Pass through Block 2... Block 3... ... Block 96
6. Final output for last position → probability distribution over vocabulary
7. Highest probability: "mat" → output "mat"
```

---

## Key Takeaways

1. **Attention** lets every token directly look at every other relevant token — no sequential bottleneck
2. **Transformers** are the architecture built around attention (+ feed-forward layers + residual connections)
3. **Multi-head attention** captures multiple types of relationships simultaneously
4. The ability to **parallelize training** is what made scaling to trillions of tokens possible
5. **Causal masking** ensures the model only looks at past tokens when generating

---

## The Key Paper

**Attention Is All You Need** (Vaswani et al., 2017) — The paper that introduced the Transformer architecture. Perhaps the most influential ML paper of the decade.

---

## What's Next

Now you understand the architecture and how it processes text. But the deeper question remains: if it's just learning patterns from text, how can it generate genuinely new content? That's Chapter 3.
