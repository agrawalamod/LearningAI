# Chapter 11: Context Engineering & Prompt Architecture

## The Shift: From Prompt Engineering to Context Engineering

The industry evolved fast. In 2023, "prompt engineering" meant crafting clever instructions. By 2025, that's table stakes. The real discipline is **context engineering** — designing the entire information environment the model operates in.

---

## What Context Engineering Actually Is

Prompt engineering: "How do I phrase this question to get a good answer?"

Context engineering: "What information does the model need, in what structure, at what time, to reliably produce correct output across thousands of requests?"

```
Prompt engineering:
  "You are a helpful assistant. Answer the user's question about our product."

Context engineering:
  System prompt (role, constraints, output format)
  + Retrieved documentation (RAG results, ranked by relevance)
  + User history (recent interactions, preferences)
  + Tool definitions (available actions, schemas, examples)
  + Few-shot examples (calibrated for this task type)
  + Guard instructions (what NOT to do, edge cases)
  + Dynamic state (current time, user tier, feature flags)
```

The difference is architectural, not editorial.

---

## The Anatomy of a Production Context Window

A real production context for an AI assistant might look like:

```
┌───────────────────────────────────────────────────────────┐
│ System Prompt (500 tokens)                                │
│   Role definition, output constraints, safety rules       │
├───────────────────────────────────────────────────────────┤
│ Tool Definitions (2,000 tokens)                           │
│   Function schemas, parameter descriptions, examples      │
├───────────────────────────────────────────────────────────┤
│ Retrieved Context (4,000 tokens)                          │
│   RAG results: docs, code, knowledge base articles        │
├───────────────────────────────────────────────────────────┤
│ Conversation History (8,000 tokens)                       │
│   Previous turns, summarized older turns                  │
├───────────────────────────────────────────────────────────┤
│ Current User Message (200 tokens)                         │
│   The actual question                                     │
├───────────────────────────────────────────────────────────┤
│ Remaining budget for output (~115,000 tokens)             │
└───────────────────────────────────────────────────────────┘
Total: ~130K context window
```

Every token of context has a cost (literal dollar cost, latency cost, attention dilution cost). Context engineering is about maximizing signal-to-noise in that window.

---

## Harness Engineering: The Meta-Discipline

Your model is one component in a **harness** — the surrounding system that orchestrates calls, manages context, validates outputs, and handles failures.

```
┌─────────────────────────────────────────────────────┐
│                    HARNESS                            │
│                                                      │
│  ┌─────────┐   ┌───────┐   ┌──────────────────┐   │
│  │ Router  │──→│ Model │──→│ Output Validator  │   │
│  └────┬────┘   └───────┘   └────────┬─────────┘   │
│       │                              │              │
│  ┌────┴────┐                   ┌─────┴─────┐      │
│  │Context  │                   │ Retry/     │      │
│  │Builder  │                   │ Fallback   │      │
│  └─────────┘                   └────────────┘      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

The harness handles:
- **Context assembly** — what goes into the prompt and in what order
- **Model routing** — which model handles this request
- **Output parsing** — extracting structured data from model output
- **Validation** — does the output meet the schema/constraints?
- **Retry logic** — what to do when the model fails
- **Fallback chains** — graceful degradation when primary model is down
- **Observability** — logging tokens, latency, errors, costs

**The key insight:** Most production failures aren't model failures — they're harness failures. The model produced something, but the harness didn't handle it correctly.

---

## Context Window Management Strategies

### Strategy 1: Sliding Window with Summarization

```
Turn 1-5:   Full conversation (recent, high value)
Turn 6-15:  Summarized to key facts (medium value)
Turn 16+:   Dropped or compressed to 1-2 sentences (low value)
```

This preserves recent context at full fidelity while keeping older context compressed.

### Strategy 2: Retrieval-Augmented Context

Instead of stuffing everything into the prompt, retrieve only what's relevant:

```python
def build_context(user_message, conversation_history):
    # Retrieve relevant docs based on current message
    relevant_docs = rag_search(user_message, top_k=5)
    
    # Retrieve relevant past conversations
    relevant_history = semantic_search(user_message, conversation_history, top_k=3)
    
    # Assemble context with priority ordering
    context = [
        system_prompt,           # Always included
        tool_definitions,        # Always included
        relevant_docs,           # Dynamically retrieved
        relevant_history,        # Dynamically retrieved  
        recent_turns[-3:],       # Last 3 turns always included
        user_message             # Current message
    ]
    return context
```

### Strategy 3: Context Budgeting

Assign token budgets to each context component:

| Component | Budget | Priority | Eviction strategy |
|---|---|---|---|
| System prompt | 500 tokens | Never evict | Fixed |
| Tool defs | 2,000 tokens | Never evict | Fixed |
| RAG results | 4,000 tokens | Evict lowest-ranked | By relevance score |
| Conversation | 8,000 tokens | Evict oldest | FIFO with summarization |
| Few-shot examples | 1,000 tokens | Evict if budget tight | Drop least relevant |

When the total exceeds the context window, evict from the lowest-priority components first.

---

## Prompt Caching vs. Semantic Caching

Two different caching strategies that solve different problems:

### Prompt Caching (KV Cache Reuse)

**What it is:** The model provider caches the computed KV (key-value) attention states for a prefix of your prompt. If your next request shares the same prefix, the prefill computation is skipped.

```
Request 1: [System prompt + Tool defs + User: "What's the weather?"]
            ↑ This prefix is computed and cached

Request 2: [System prompt + Tool defs + User: "What time is it?"]
            ↑ Cache HIT — prefix KV states reused, only new tokens computed
```

**When it saves money and latency:**
- You have a large, stable system prompt (same across requests)
- Tool definitions don't change between requests
- RAG results are the same (rare)

**Tradeoff:** Only works for *exact prefix matches*. Change one token in the prefix → cache miss.

**Anthropic's implementation:** You pay to write the cache (1.25x input cost), but cache reads are 0.1x input cost. Break-even after ~4 cache reads of the same prefix.

### Semantic Caching

**What it is:** Cache full responses keyed by the *semantic meaning* of the query, not the exact text.

```
Request 1: "What's the return policy for electronics?"
Response:   "Electronics can be returned within 30 days..."
            → Cache: embed(query) → response

Request 2: "Can I return my laptop? What's the policy?"
            → embed(query) matches Request 1 with similarity > 0.95
            → Return cached response (no model call at all)
```

**When it works:**
- High volume of semantically similar questions (customer support, FAQ)
- Answers don't depend on user-specific context
- Freshness isn't critical

**When it fails:**
- Queries look similar but need different answers (context-dependent)
- Stale cached responses (information changed)
- Multi-turn conversations (context makes "same question" have different answers)

### The Tradeoff Table

| Dimension | Prompt Caching | Semantic Caching |
|---|---|---|
| What it caches | KV states for exact prefix | Full responses for similar queries |
| Cache hit condition | Exact byte-level prefix match | Semantic similarity threshold |
| Latency savings | Reduces prefill time (~50-80%) | Eliminates model call entirely |
| Cost savings | Input token cost only | Full request cost |
| Freshness risk | None (still generates new response) | High (stale cached response) |
| Quality risk | None | Response may not fit context exactly |
| Best for | Stable system prompts, tools | High-volume FAQ, commodity queries |

**The production pattern:** Use both. Prompt caching for the stable prefix, semantic caching for repeated commodity questions, full model calls for novel or context-dependent queries.

---

## Common Context Engineering Failures

### Failure 1: Attention Dilution

Stuffing too much irrelevant context degrades performance:

```
System prompt: 500 tokens of role definition
+ 3,000 tokens of irrelevant docs (retrieved but not useful)
+ User question: "What's 2+2?"

The model attends to the irrelevant docs, gets distracted, gives a worse answer
than it would with just the system prompt + question.
```

**Fix:** Aggressively filter retrieved context. Less but more relevant > more but noisy.

### Failure 2: Instruction Conflicts

Multiple instructions that contradict each other:

```
System: "Always respond in JSON format"
System: "Be conversational and friendly"
System: "If you don't know, say so"
User: "I don't think the API is working"

Model is confused: JSON? Conversational? Admit uncertainty? All three conflict.
```

**Fix:** Clear priority ordering. Explicit conflict resolution rules.

### Failure 3: Context Ordering Effects

Models are sensitive to where information appears in the context:

```
Research shows: information at the beginning and end of context 
is recalled better than information in the middle ("lost in the middle" effect).
```

**Fix:** Put critical instructions at the start AND end. Put retrieved docs in relevance order with most relevant first.

---

## Key Takeaways

1. **Context engineering > prompt engineering** — it's about the entire information environment, not just the question
2. **Harness engineering is the real discipline** — the model is one component in a system that routes, validates, retries, and falls back
3. **Prompt caching saves on prefill cost for stable prefixes** — exact match only, no quality risk
4. **Semantic caching eliminates model calls for repeated queries** — saves more money, but risks staleness and context mismatch
5. **Context has a signal-to-noise ratio** — more isn't better; relevant is better
6. **Production context management requires budgeting, eviction strategies, and priority ordering**
7. **Most "model failures" are actually harness failures** — bad context assembly, not bad generation

---

## Key Papers & Resources

- **Lost in the Middle** (Liu et al., 2023) — how LLMs struggle with information in the middle of long contexts
- **Prompt Caching** (Anthropic, 2024) — technical documentation on KV cache reuse
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020) — RAG foundation
- **Many-Shot In-Context Learning** (Agarwal et al., 2024) — scaling few-shot to many-shot with caching

---

## What's Next

Context gets assembled, but how does the inference engine actually process it? The next chapter dives into the internals: KV cache management, prefill vs. decode optimization, and how serving infrastructure handles thousands of concurrent requests. Chapter 12.
