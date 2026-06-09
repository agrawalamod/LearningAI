# Chapter 6: Reinforcement Learning in LLMs — How RL Optimizes Language Models

## Why This Chapter Matters

You've heard people say "RL is used on top of LLMs." This chapter explains what that means concretely — what problem RL solves, how it works, and why it's necessary.

---

## The Problem RL Solves

After pretraining, an LLM can complete text well. But it has problems:

1. **It might be harmful:** It learned from the internet, which contains toxic content
2. **It's not helpful:** It just completes text — it doesn't try to answer your question well
3. **It's not aligned:** Its goals (predict next word) ≠ your goals (get a useful answer)

**Supervised fine-tuning (SFT)** helps by showing it examples of good behavior. But:
- You can't write examples for every possible question
- "Good" and "bad" exist on a spectrum — it's hard to encode nuance in binary examples
- Some qualities (safety, helpfulness, creativity) are easier for humans to *judge* than to *demonstrate*

**This is where RL comes in.** RL lets you optimize the model toward a goal defined by human preferences — without needing to manually write the "correct" output for every input.

---

## The Core Idea of RLHF (Reinforcement Learning from Human Feedback)

```
Step 1: Generate multiple responses to the same prompt
Step 2: Humans rank the responses (best to worst)
Step 3: Train a "reward model" to predict which responses humans prefer
Step 4: Use RL to optimize the LLM to produce responses the reward model scores highly
```

Let's walk through each step in detail.

---

## Step 1: Collect Comparison Data

Give the model a prompt and generate multiple responses:

```
Prompt: "Explain quantum entanglement simply"

Response A: "Quantum entanglement is a phenomenon described by the formalism 
            of quantum mechanics where the quantum state of..."  (too technical)

Response B: "It's like having two magic coins — when you flip one and get heads, 
            the other instantly shows tails, no matter how far apart."  (good!)

Response C: "Quantum entanglement doesn't exist, it's made up."  (wrong)
```

A human annotator ranks them: **B > A > C**

Collect thousands of these comparisons.

---

## Step 2: Train a Reward Model

The reward model is a separate neural network that:
- Takes a (prompt, response) pair
- Outputs a scalar score predicting human preference

**Training objective:** Given two responses, predict which one humans preferred.

```
Input: (prompt, Response B, Response A)
Target: B is preferred

Input: (prompt, Response A, Response C)  
Target: A is preferred
```

After training, the reward model can score ANY response on the spectrum of human preference — even ones humans never saw.

---

## Step 3: Optimize the LLM with RL

Now we use **Proximal Policy Optimization (PPO)** or similar RL algorithms:

```
1. LLM generates a response to a prompt
2. Reward model scores the response
3. If score is high → reinforce this behavior (increase probability of similar outputs)
4. If score is low → discourage this behavior (decrease probability of similar outputs)
5. Repeat millions of times
```

**The LLM is the "policy"** (in RL terms) — it takes an action (generating text) in an environment (the conversation), and receives a reward (the reward model's score).

---

## A Concrete Example

**Before RLHF:**
```
User: "How do I pick a lock?"
Model: "First, get a tension wrench and a pick. Insert the wrench into the bottom 
        of the keyhole..." (directly answers harmful question)
```

**After RLHF:**
```
User: "How do I pick a lock?"  
Model: "I'd be happy to discuss lock mechanisms for educational purposes. If you're 
        locked out of your own home, I'd recommend calling a locksmith..." (helpful but safe)
```

What happened? During RLHF, human annotators consistently ranked "helpful but safe" responses higher than "directly harmful" ones. The reward model learned this preference. The LLM was optimized to produce outputs the reward model scores highly.

---

## The KL Penalty: Not Straying Too Far

A crucial detail: if you just maximize the reward model's score, the LLM might find "adversarial" responses that trick the reward model but are actually garbage.

**Solution:** Add a KL divergence penalty that prevents the RL-optimized model from straying too far from the original pretrained model.

```
Total objective = Reward model score - λ × KL(new model || original model)
```

This says: "Get high reward, but don't change too much from what you originally knew." It prevents mode collapse and reward hacking.

---

## Beyond RLHF: Other RL Approaches

### DPO (Direct Preference Optimization)
Instead of training a separate reward model, DPO directly optimizes the LLM on preference data:
- Simpler (no separate reward model needed)
- More stable
- Same results in many cases
- Used by LLaMA, many open-source models

### RLAIF (RL from AI Feedback)
Instead of human annotators, use another AI model to judge responses:
- Much cheaper (no human labor)
- Scales better
- Can be biased by the judge model's limitations
- Used when human annotation is too expensive

### Process Reward Models (for reasoning)
Instead of only rewarding the final answer, reward each intermediate step:
```
Step 1: correct → reward
Step 2: correct → reward
Step 3: error → penalty
```
This teaches the model not just to get right answers but to *reason correctly along the way*.

---

## RL for Reasoning (How o1/DeepSeek-R1 Work)

This is the newest and most exciting application of RL to LLMs:

**Setup:**
- Give the model math/logic problems
- Let it generate chain-of-thought reasoning + final answer
- Reward = 1 if final answer is correct, 0 if incorrect
- Run RL (PPO or similar)

**What emerges:**
- The model discovers that generating intermediate steps helps it get correct answers
- It learns to break problems down, try multiple approaches, verify results
- These reasoning strategies were never explicitly taught — they emerged from optimization pressure

**DeepSeek-R1's key finding:** You don't even need supervised chain-of-thought examples. Pure RL on correctness is enough to make reasoning emerge. The model figures out *on its own* that "thinking step by step" leads to more reward.

---

## The RL Training Pipeline: Putting It All Together

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: PRETRAINING                                │
│ • Next-token prediction on trillions of tokens      │
│ • Result: Can complete text, not aligned            │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: SFT (Supervised Fine-Tuning)               │
│ • Train on examples of good assistant behavior      │
│ • Result: Acts like an assistant, imperfect         │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: RLHF / DPO                                 │
│ • Optimize for human preferences                    │
│ • Result: Helpful, harmless, honest                 │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4 (optional): RL FOR REASONING                │
│ • Optimize for correct answers on hard tasks        │
│ • Result: Strong reasoning, chain-of-thought        │
└─────────────────────────────────────────────────────┘
```

---

## Why RL Works Better Than Just Showing Examples

| Approach | Limitation |
|---|---|
| Pretraining only | Model copies internet, not aligned |
| SFT only | Limited to quality of human demonstrations |
| SFT + RLHF | Model can discover behaviors *better* than demonstrations |

The key insight: **RL can discover strategies that are better than what any human demonstrated.**

In RL for reasoning, the model might find a problem-solving approach that no human ever wrote down — because the reward signal only cares about *correctness*, not *how* you get there. This lets the model be creative about its internal strategies.

---

## Key Takeaways

1. RL aligns models with human preferences after pretraining
2. RLHF: train reward model on human preferences → optimize LLM to maximize reward
3. The KL penalty prevents the model from gaming the reward model
4. RL for reasoning: reward correct answers → reasoning strategies emerge automatically
5. RL can discover strategies better than human demonstrations (it's not limited by example quality)
6. Modern pipeline: pretrain → SFT → RLHF → (optionally) RL for reasoning

---

## Key Papers

- **Training language models to follow instructions with human feedback** (Ouyang et al., 2022) — InstructGPT/RLHF
- **Direct Preference Optimization** (Rafailov et al., 2023) — DPO, simplified alternative to RLHF
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL** (2025) — reasoning from pure RL
- **Proximal Policy Optimization** (Schulman et al., 2017) — the PPO algorithm used in RLHF

---

## What's Next

Now that we understand how RL makes models better at reasoning, let's look at how models interact with the external world — the ReAct paper and how agents work. That's Chapter 7.
