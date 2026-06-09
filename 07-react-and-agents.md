# Chapter 7: The ReAct Paper & How Agents Work

## Why This Matters

Your friend said: "ReAct is essentially the basis of how agents are made." They're right. This paper fundamentally changed how we use LLMs — from passive question-answerers to active problem-solvers.

---

## The Problem Before ReAct

LLMs had two modes, both limited:

**Mode 1: Direct Answer (no tools)**
```
Q: "What's the population of the capital of the country that won the 2022 World Cup?"
A: "Argentina won the 2022 World Cup. The capital is Buenos Aires. 
    The population is approximately 3.1 million." ← Might be outdated or wrong
```
Problem: The model is relying on memorized knowledge that might be stale or incorrect.

**Mode 2: Action Only (tools but no reasoning)**
```
Q: Same question
Action: search("2022 World Cup winner") → "Argentina"
Action: search("Argentina capital") → "Buenos Aires"  
Action: search("Buenos Aires population") → "15.6 million (metro)" 
Answer: "15.6 million" ← Took the wrong number (metro vs city)
```
Problem: The model takes actions mechanically without reasoning about what the results mean.

---

## ReAct: Synergy of Reasoning + Acting

**Paper:** *ReAct: Synergizing Reasoning and Acting in Language Models* (Yao et al., 2022)

The key insight: **Interleave reasoning traces with actions.** Let the model *think* about what it's seeing and what to do next.

```
Q: "What's the population of the capital of the country that won the 2022 World Cup?"

Thought: I need to find who won the 2022 World Cup first.
Action: search("2022 FIFA World Cup winner")
Observation: Argentina defeated France in the final.

Thought: Argentina won. Now I need the capital of Argentina.
Action: search("capital of Argentina")
Observation: Buenos Aires is the capital of Argentina.

Thought: Now I need the population of Buenos Aires. I should look for the city 
         proper population, not the metro area, since the question asks about "the capital."
Action: search("Buenos Aires city population 2023")
Observation: The city of Buenos Aires has a population of approximately 3.1 million.

Thought: I have all the information. The answer is about 3.1 million.
Answer: Approximately 3.1 million people.
```

### What Changed?

The **Thought** steps let the model:
1. Plan what to do next
2. Interpret results (not just blindly forward them)
3. Catch potential errors ("city vs metro — I should be specific")
4. Decide when it has enough information to stop

---

## The ReAct Pattern: Thought → Action → Observation → Repeat

```
┌─────────┐     ┌─────────┐     ┌─────────────┐
│ Thought │ ──→ │ Action  │ ──→ │ Observation │ ──→ (loop back to Thought)
│ (reason)│     │ (do)    │     │ (see result)│
└─────────┘     └─────────┘     └─────────────┘
```

- **Thought:** The model reasons about what to do (generated text, not executed)
- **Action:** The model calls a tool (search, calculator, code execution, API call)
- **Observation:** The tool returns a result (injected back into the model's context)

The model decides when to stop the loop and give a final answer.

---

## Why This Is the Foundation of All Agents

An **AI agent** is just: an LLM in a ReAct-style loop with access to tools.

Every agent framework (LangChain, AutoGPT, Claude's tool use, OpenAI's function calling) follows this same pattern:

```python
# Simplified agent loop (pseudocode)
while not done:
    # The LLM sees: the original task + all previous thoughts/actions/observations
    response = llm.generate(context)
    
    if response.is_final_answer:
        return response.answer
    
    if response.is_thought:
        context.append(response.thought)
    
    if response.is_action:
        result = execute_tool(response.action)
        context.append(f"Observation: {result}")
```

That's it. Every agent is this loop with different tools plugged in.

---

## What Kinds of Tools Can Agents Use?

| Tool Type | Examples |
|---|---|
| **Search** | Web search, document retrieval, database queries |
| **Code execution** | Run Python, JavaScript, SQL |
| **APIs** | Send email, create calendar events, update tickets |
| **File system** | Read files, write files, list directories |
| **Other models** | Image generation, speech-to-text |
| **Browser** | Navigate web pages, click buttons, fill forms |

The tool set is what defines an agent's capabilities. Same LLM + different tools = different agent.

---

## Concrete Example: A Coding Agent

```
Task: "Fix the failing test in auth_service.py"

Thought: I should first see what the test error is.
Action: run_tests("tests/test_auth.py")
Observation: FAILED - test_login_expired_token - AssertionError: 
             expected 401, got 200

Thought: The test expects a 401 when the token is expired, but the service 
         returns 200. I should look at the token validation code.
Action: read_file("src/auth_service.py")
Observation: [file contents showing token validation logic]

Thought: I see the issue — the expiry check uses `<` instead of `<=`, so tokens 
         expiring at exactly the current time are treated as valid.
Action: edit_file("src/auth_service.py", line=47, 
         old="if token.exp < current_time:",
         new="if token.exp <= current_time:")
Observation: File edited successfully.

Thought: Let me verify the fix by running the tests again.
Action: run_tests("tests/test_auth.py")
Observation: ALL TESTS PASSED

Answer: Fixed the issue. The token expiry check on line 47 used strict less-than 
        instead of less-than-or-equal. Tokens expiring at exactly the current time 
        were incorrectly treated as valid.
```

This is exactly how Claude Code (the tool you're talking to right now) works.

---

## The Evolution from ReAct to Modern Agents

### Level 1: Single ReAct Loop (2022)
- One LLM, one loop, simple tools
- Works for straightforward tasks

### Level 2: Multi-step Planning (2023)
- Agent creates a plan first, then executes steps
- Can recover from failed steps

### Level 3: Multi-agent Systems (2024-2025)
- Multiple specialized agents collaborating
- One "orchestrator" agent delegates to sub-agents
- Example: a research agent, a coding agent, and a review agent working together

### Level 4: Autonomous Agents with Memory (2025+)
- Agents that remember across sessions
- Can be given long-running tasks
- Monitor systems, respond to events

---

## Why ReAct Works So Well

1. **Reasoning grounds actions:** The model doesn't blindly act — it thinks about what it's doing, reducing errors

2. **Actions ground reasoning:** The model doesn't just hallucinate facts — it looks things up, reducing confabulation

3. **Self-correction:** When an observation doesn't match expectations, the Thought step lets the model adjust its approach

4. **Transparency:** You can see *why* the agent did what it did by reading its Thought traces

---

## The Key Design Choices for Agents

When building an agent, you decide:

| Choice | Options |
|---|---|
| **Which LLM** | Smarter = better reasoning but slower/costlier |
| **Which tools** | More tools = more capable but harder to choose correctly |
| **How much autonomy** | Fully autonomous vs. human-in-the-loop |
| **When to stop** | After N steps? When confident? When human approves? |
| **Memory** | Forget after each task vs. persist across tasks |

---

## Limitations of ReAct-Style Agents

1. **Error propagation:** One bad Thought or wrong tool call can derail the whole chain
2. **Cost:** Each step requires an LLM call (expensive for complex tasks)
3. **Latency:** Each step takes time (sequential, not parallel)
4. **Tool choice errors:** Model might use the wrong tool or wrong parameters
5. **Infinite loops:** Agent might keep trying the same failing approach

---

## Key Takeaways

1. **ReAct** = interleave Thought (reasoning) with Action (tool use) and Observation (results)
2. This pattern is the foundation of **all** AI agents
3. Reasoning without tools → hallucination. Tools without reasoning → mechanical errors. Together → powerful.
4. Every agent is just an LLM in a loop with tools — the tools define its capabilities
5. Modern agents build on ReAct with planning, multi-agent systems, and persistent memory

---

## Key Papers

- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022) — the foundational paper
- **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023) — self-taught tool use
- **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., 2023) — agents that learn from failures
- **The Landscape of Emerging AI Agent Architectures** (Masterman et al., 2024) — survey of agent designs

---

## What's Next

We've covered how LLMs reason and act. Now let's tackle the multimodal world — how models handle video, egocentric data, and the path toward world models. Chapter 8.
