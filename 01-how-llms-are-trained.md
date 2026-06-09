# Chapter 1: How Are LLMs Trained?

## The Core Idea in One Sentence

An LLM is trained by showing it massive amounts of text and asking it, over and over: **"Given these words, what word comes next?"**

That's it. The entire foundation of GPT, Claude, LLaMA, and every other large language model is built on this deceptively simple task: **next-token prediction**.

---

## Step 1: Tokenization — Breaking Text Into Pieces

Before training begins, all text must be broken into **tokens**. Tokens are not exactly words — they're sub-word units.

**Example:**

```
Input:  "Understanding transformers is fascinating"
Tokens: ["Under", "standing", " transform", "ers", " is", " fascinating"]
```

Why not just use whole words? Because:
- There are too many unique words (every name, every typo, every language)
- Sub-word tokens let the model handle words it has never seen by composing them from known pieces

Common tokenizers: **BPE (Byte-Pair Encoding)**, used in GPT models, and **SentencePiece**, used in LLaMA.

A typical LLM has a vocabulary of 32,000–100,000 tokens.

---

## Step 2: The Training Objective — Next-Token Prediction

The model sees a sequence of tokens and must predict the next one.

**Example:**

```
Input:  "The cat sat on the"
Target: "mat"
```

The model outputs a **probability distribution** over all possible next tokens:

```
"mat"    → 0.12
"floor"  → 0.09
"roof"   → 0.04
"table"  → 0.06
"dog"    → 0.001
...
(32,000 tokens each get some probability)
```

The training signal says: "The correct answer was 'mat' — adjust your internal parameters so that 'mat' gets a higher probability next time you see this context."

This adjustment happens through **backpropagation** and **gradient descent** — the same math that trains all neural networks.

---

## Step 3: Self-Supervised Learning — No Human Labels Needed

This is what makes LLM training so powerful: **you don't need humans to label anything**.

In traditional machine learning:
- You need humans to label images as "cat" or "dog"
- You need humans to mark emails as "spam" or "not spam"

With LLMs:
- The text itself provides the labels
- Every sentence is automatically a training example
- "The cat sat on the **mat**" — the word "mat" is both the answer and something that exists naturally in the data

This is called **self-supervised learning**. It means you can train on essentially unlimited data without human annotation cost.

---

## Step 4: Scale — The Three Ingredients

Training a modern LLM requires enormous amounts of three things:

| Ingredient | Example Scale (GPT-4 class) |
|---|---|
| **Data** | Trillions of tokens (books, web, code, papers) |
| **Parameters** | Hundreds of billions of learned weights |
| **Compute** | Thousands of GPUs running for months |

**Key insight:** The "scaling laws" (Kaplan et al., 2020) showed that model performance improves **predictably** as you increase data, parameters, and compute. This is what sparked the race to build bigger models.

**Paper:** *Scaling Laws for Neural Language Models* (Kaplan et al., 2020)

---

## Step 5: The Training Pipeline (Three Stages)

Modern LLMs go through multiple stages:

### Stage 1: Pretraining
- Train on massive text corpus (internet, books, code)
- Objective: next-token prediction
- Result: A model that can complete text, but is not particularly helpful or safe
- This is the most expensive stage (months of GPU time)

### Stage 2: Supervised Fine-Tuning (SFT)
- Train on curated examples of helpful conversations
- Human writers create examples of ideal assistant behavior
- "When a user asks X, here's what a good response looks like"
- Result: A model that acts more like an assistant

### Stage 3: Reinforcement Learning from Human Feedback (RLHF)
- Humans rank model outputs from best to worst
- A reward model learns what humans prefer
- The LLM is optimized to produce outputs the reward model scores highly
- Result: A model that is more helpful, harmless, and honest

*(We'll cover RLHF in detail in Chapter 5)*

---

## A Concrete Example of the Full Flow

Let's trace through what happens:

**Pretraining data might include:**
```
"The theory of relativity was proposed by Albert Einstein in 1905..."
"Machine learning models are trained using gradient descent..."
"To make pasta, boil water and add salt..."
```

**After pretraining**, the model can complete text:
```
Input:  "The theory of relativity"
Output: "was developed by Einstein and fundamentally changed our understanding of space and time."
```

But it might also produce:
```
Input:  "How do I make a bomb?"
Output: "First, you need to gather..." (unhelpful and dangerous)
```

**After fine-tuning + RLHF**, the same model instead says:
```
Input:  "How do I make a bomb?"
Output: "I can't help with that. If you're interested in chemistry, I'd be happy to discuss safe experiments..."
```

---

## Key Takeaways

1. LLMs learn by predicting the next word, billions of times
2. No human labeling is needed for pretraining — the text labels itself
3. The model learns a compressed representation of patterns in language
4. Three stages: pretraining → fine-tuning → RLHF
5. Scale (data + parameters + compute) is what makes them powerful

---

## Key Papers

- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) — showed performance scales predictably
- **Language Models are Few-Shot Learners** (Brown et al., 2020) — the GPT-3 paper, demonstrated emergent abilities from scale
- **Training language models to follow instructions with human feedback** (Ouyang et al., 2022) — the InstructGPT/RLHF paper

---

## What's Next

This chapter covered *what* LLMs are trained to do. But *how* does the model actually process text and make predictions? That's where **attention** and **transformers** come in — covered in Chapter 2.
