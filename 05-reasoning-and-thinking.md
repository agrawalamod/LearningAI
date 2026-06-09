# Chapter 5: Reasoning & Thinking — What It Actually Means Inside an LLM

## The Core Question

> "How did we go from predicting the next word to reasoning and thinking? What does reasoning even mean in LLM language?"

This is one of the most important questions in AI right now. Let's build up the answer layer by layer.

---

## What Reasoning Looks Like From the Outside

When you ask Claude: "If all roses are flowers and all flowers need water, do roses need water?"

It answers: "Yes, roses need water."

That looks like logical reasoning. But what's actually happening internally?

---

## The Naive View (Wrong)

"The model just memorized that roses need water from its training data."

This is wrong. You can verify by asking increasingly obscure questions:
- "If all blickets are daxes and all daxes are feps, are blickets feps?"
- The model answers correctly with made-up words it has never seen.

So it's not memorization. Something more is going on.

---

## What's Actually Happening: Learned Reasoning Patterns

During training, the model saw millions of examples of logical chains in text:
- Textbooks with step-by-step proofs
- Forum posts where people work through problems
- Scientific papers with logical arguments
- Code with if/then/else logic

From this, the model learned **abstract patterns of reasoning** — templates of how conclusions follow from premises. Not specific conclusions, but the *structure* of inference itself.

**Analogy:** A student who reads 1000 geometry proofs doesn't memorize them. They learn the *technique* of proof: how to chain axioms, how to use contradiction, how to apply previously proven lemmas. An LLM does the same — at a statistical level.

---

## The Role of Chain-of-Thought

A breakthrough discovery (Wei et al., 2022): if you ask a model to **"think step by step,"** it performs dramatically better on reasoning tasks.

**Without chain-of-thought:**
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have?
A: 11 ✓ (sometimes) or 9 ✗ (often)
```

**With chain-of-thought:**
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have? Think step by step.
A: Roger starts with 5 balls. He buys 2 cans. Each can has 3 balls. 
   2 × 3 = 6 new balls. 5 + 6 = 11. The answer is 11. ✓ (reliably)
```

### Why does this work?

**Key insight:** Each token the model generates becomes part of its input for the next token. By generating intermediate steps, the model is essentially **giving itself a scratchpad**.

Without chain-of-thought: the model must go from problem → answer in a single forward pass (a fixed amount of computation).

With chain-of-thought: the model gets additional forward passes (one per intermediate token), each building on previous reasoning.

```
Single forward pass = limited computation = simple problems only
N forward passes (via N reasoning tokens) = N× computation = complex problems
```

**This is why reasoning "takes time."** It's not thinking in the background — it's literally generating more tokens, and each token gives it more computation to work with.

---

## What Reasoning IS in LLM Terms

Reasoning in an LLM is: **the serial generation of intermediate tokens that progressively constrain and refine the probability distribution toward a correct final answer.**

More concretely:

```
Problem: "What is 347 × 28?"

Without reasoning (single hop):
  P(answer | problem) → very spread out, might get wrong answer

With reasoning (multiple hops):
  P("347 × 28" | problem) → starts the decomposition
  P("= 347 × 20 + 347 × 8" | previous) → breaks it down
  P("= 6940 + 2776" | previous) → intermediate results
  P("= 9716" | previous) → final answer, heavily constrained by previous steps
```

Each step **narrows the distribution** for the next step. By the time you reach the final answer, the probability distribution is sharply peaked at the correct answer because all the intermediate constraints make alternatives nearly impossible.

---

## Extended Thinking / "Deep Reasoning" Models

Models like Claude with extended thinking, OpenAI's o1/o3, and DeepSeek-R1 take this further:

1. **They generate many more reasoning tokens** (sometimes thousands before answering)
2. **They explore multiple approaches** ("Let me try this... no, that doesn't work. What if instead...")
3. **They self-correct** ("Wait, I made an error in step 3. Let me redo...")
4. **They verify** ("Let me check: does my answer satisfy the original constraints?")

### What "thinking time" actually is:

```
Simple question: "What's 2+2?"
→ Model barely needs computation → almost no thinking tokens → fast

Hard question: "Prove that there are infinitely many primes"
→ Model needs many reasoning steps → generates many thinking tokens → slow
```

**The time is not the model "processing" — it's the model generating tokens.** More tokens = more time. Complex reasoning requires more tokens.

---

## Why Does Claude "Think" Even For Simple Questions?

You noticed this. Here's why:

1. **The model doesn't always know in advance how hard a question is.** It may start generating reasoning tokens as a default policy.

2. **Extended thinking is often always-on** for safety/quality: even "How are you?" might trigger the model to briefly reason about what kind of response is appropriate.

3. **It's a design choice:** Models trained with reasoning turned on tend to give better answers across the board, so it's left on even when unnecessary.

4. **The tradeoff:** A small amount of unnecessary thinking on easy questions is worth the large improvement on hard questions.

---

## Is It "Branching and Converging"?

You hypothesized: "Is it going in different branches and seeing which ones converge with higher probability?"

**Sort of!** There are two ways this happens:

### Within a single generation (sequential reasoning):
The model generates a chain, realizes it's stuck, backtracks, tries another path:
```
"Let me try approach A... hmm, that gives a contradiction.
 Let me try approach B... yes, this works because..."
```
This is serial exploration — one branch at a time.

### Across multiple generations (best-of-N / tree search):
Some systems (like o1) may internally:
1. Generate multiple reasoning chains in parallel
2. Score each chain
3. Select the most consistent/highest-scoring one

This IS branching-and-converging, but it's done by generating multiple complete chains, not within a single forward pass.

---

## Why Some Models Reason and Others Don't

| Model Type | Reasoning | Why |
|---|---|---|
| Base model (no fine-tuning) | Minimal | Never trained to show work |
| Instruction-tuned | Some | Seen examples of step-by-step work |
| Reasoning-optimized (o1, DeepSeek-R1) | Strong | Specifically trained with RL to reason well |

### It's BOTH architecture and data:

**Architecture contribution:**
- More parameters = more "capacity" for reasoning patterns
- Longer context = more room for chains of thought
- But a small model trained specifically on reasoning can outperform a larger general model

**Data/Training contribution (more important):**
- Models trained on math proofs, code, logical arguments learn reasoning better
- RLHF/RL specifically rewards correct final answers, incentivizing models to develop reasoning chains
- Process reward models (PRMs) reward *each step* being correct, not just the final answer

**The biggest factor:** Models that are trained with **reinforcement learning on reasoning tasks** (where they get reward for correct answers) learn to generate useful intermediate steps — because doing so leads to more reward. This is how o1 and DeepSeek-R1 were trained.

---

## A Concrete Example: How Reasoning Emerges

Imagine training a model with RL on math problems:

**Round 1:** Model outputs answers directly. Gets 30% correct.

**Round 2:** Some random chains include intermediate steps. Those chains get more correct answers. RL reinforces them.

**Round 3:** Model starts consistently producing intermediate steps. Accuracy rises to 60%.

**Round 100:** Model has learned sophisticated strategies — decomposition, verification, backtracking. Accuracy reaches 90%.

The model wasn't told to "reason." It discovered that reasoning *works* because it leads to more reward. This is emergent behavior from optimization pressure.

---

## What Reasoning IS NOT

- It is NOT the model "understanding" in a philosophical sense
- It is NOT running a logic engine or symbolic solver internally
- It is NOT guaranteed to be correct (it can make reasoning errors)
- It is NOT the same as human reasoning (the mechanism is different, the outcome is similar)

What it IS:
- Learned statistical patterns of logical structure
- Amplified by chain-of-thought (more computation per problem)
- Optimized by RL to produce chains that lead to correct answers
- Surprising in how far it can go

---

## The Limits of LLM Reasoning (Preview of Chapter 10)

- Struggles with very long chains (>20 steps) without errors compounding
- Cannot reason about things outside its knowledge
- Can "sound" logical while being wrong (confident hallucination)
- Arithmetic becomes unreliable at large numbers (no calculator module)
- Novel proof techniques that have no analog in training data are very hard

---

## Key Takeaways

1. "Reasoning" = generating intermediate tokens that constrain the final answer
2. Chain-of-thought gives the model a scratchpad — more tokens = more computation = harder problems
3. "Thinking time" IS the generation of reasoning tokens — not background processing
4. Reasoning ability comes from training on logical text + RL optimization for correct answers
5. Some models reason better because they were specifically trained to (data + RL), not just architecture
6. It's learned patterns of logic, not true understanding — but the practical results are remarkable

---

## Key Papers

- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., 2022)
- **Let's Verify Step by Step** (Lightman et al., 2023) — process reward models for math reasoning
- **Scaling LLM Test-Time Compute** (Snell et al., 2024) — using more compute at inference for reasoning
- **DeepSeek-R1** (DeepSeek, 2025) — RL training for reasoning without supervised chain-of-thought data

---

## What's Next

We mentioned RL several times — "models trained with reinforcement learning to reason better." But what IS reinforcement learning in the context of LLMs? How does it actually work? That's Chapter 6.
