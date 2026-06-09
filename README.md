# How LLMs Work: From Fundamentals to AI Engineering

A practitioner's guide to understanding and building with Large Language Models — written progressively as a learning journal.

## What This Is

A 16-chapter book that starts with how LLMs are trained and builds all the way up to production AI engineering: inference infrastructure, structured output, evals, security, and system design tradeoffs.

Written in a direct, example-heavy style. Each chapter builds on the previous ones.

## Chapters

### Part I: Foundations

| # | Chapter | Core Topic |
|---|---------|-----------|
| 1 | [How Are LLMs Trained?](01-how-llms-are-trained.md) | Next-token prediction, self-supervised learning, training pipeline |
| 2 | [Attention & Transformers](02-attention-and-transformers.md) | Self-attention, multi-head attention, transformer architecture |
| 3 | [Generative AI Landscape](03-generative-ai-landscape.md) | LLMs vs GANs vs Diffusion — when to use which |
| 4 | [Generalization & Novelty](04-generalization-and-novelty.md) | Compositional generalization, manifold intuition, hard limits |
| 5 | [Reasoning & Thinking](05-reasoning-and-thinking.md) | Chain-of-thought, extended thinking, why reasoning takes time |
| 6 | [Reinforcement Learning in LLMs](06-reinforcement-learning-in-llms.md) | RLHF, DPO, process reward models, RL for reasoning |

### Part II: Application

| # | Chapter | Core Topic |
|---|---------|-----------|
| 7 | [ReAct & Agents](07-react-and-agents.md) | Thought-Action-Observation loops, tool use, agent architectures |
| 8 | [Multimodal & World Models](08-multimodal-models-and-world-models.md) | Vision encoders, video understanding, egocentric data, physics |

### Part III: Systems

| # | Chapter | Core Topic |
|---|---------|-----------|
| 9 | [Latency & Performance](09-latency-and-performance.md) | Prefill vs decode, TTFT, TPS, batching, model size tradeoffs |
| 10 | [Limits, Determinism & Trust](10-limits-determinism-and-trust.md) | Reasoning ceilings, production reliability, guardrail patterns |

### Part IV: AI Engineering

| # | Chapter | Core Topic |
|---|---------|-----------|
| 11 | [Context Engineering & Prompt Architecture](11-context-engineering-and-prompt-architecture.md) | Harness engineering, context budgeting, prompt vs semantic caching |
| 12 | [Inference Infrastructure](12-inference-infrastructure-and-serving.md) | KV cache, paged attention, continuous batching, speculative decoding, quantization |
| 13 | [Structured Output & Function Calling](13-structured-output-and-function-calling.md) | Schema validation, repair loops, tool contracts, agent guardrails, model routing |
| 14 | [Evals, Observability & Cost](14-evals-observability-and-cost.md) | Golden sets, LLM-as-judge, RAG evals, traces, drift detection, cost attribution |
| 15 | [Safety, Security & Multi-Tenancy](15-safety-security-and-multi-tenancy.md) | Prompt injection, data leakage, tenant isolation, permission boundaries |
| 16 | [Tradeoffs & Production Decisions](16-tradeoffs-and-production-decision-making.md) | Fine-tune vs RAG vs ICL vs distillation, 4-axis tradeoff space, decision framework |

## PDF

A compiled PDF of all chapters is included for mobile reading: **[AI-Engineering-Book.pdf](AI-Engineering-Book.pdf)**

## Status

This is a living document. New chapters are added as I learn new topics.
