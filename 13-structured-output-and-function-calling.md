# Chapter 13: Structured Output, Function Calling & Tool Reliability

## The Core Tension

LLMs generate free-form text. Production systems need **deterministic, machine-readable output**. The entire challenge of this chapter is bridging that gap reliably.

```
What the model produces:  "The temperature is around 72 degrees Fahrenheit"
What your system needs:   {"temperature": 72, "unit": "fahrenheit"}

What the model produces:  "I'll search for that"
What your system needs:   {"tool": "search", "args": {"query": "...", "limit": 10}}
```

---

## Part 1: Structured Output

### The Problem

Ask a model to respond in JSON, and you'll get JSON. Most of the time. The failures are insidious:

```
❌ Trailing comma:        {"name": "Alice", "age": 30,}
❌ Unquoted key:          {name: "Alice"}
❌ Extra explanation:     Here's the JSON: {"name": "Alice"}
❌ Markdown wrapping:     ```json\n{"name": "Alice"}\n```
❌ Missing required field: {"name": "Alice"}  (age was required)
❌ Wrong type:            {"name": "Alice", "age": "thirty"}
❌ Truncation:            {"name": "Alice", "ag  (context limit hit)
```

At 100 requests, this is annoying. At 1 million requests per day, even a 0.1% failure rate means 1,000 broken responses.

### Solution 1: Constrained Decoding (Grammar-Based)

Force the model's output to conform to a grammar at generation time. At each token position, mask out tokens that would violate the schema.

```
Schema: {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

At position 0: Only allow "{" (must start JSON object)
At position 1: Only allow '"' (must start a key)
After "name": ": Only allow string token starts
After value: Only allow "," or "}" (must be valid JSON continuation)
```

**Result:** 100% syntactically valid JSON. Every time. The model literally cannot produce invalid output because invalid tokens are masked to probability zero.

**Implemented in:** OpenAI's `response_format`, Anthropic's tool use, vLLM with Outlines, llama.cpp with GBNF grammars.

**Limitation:** Guarantees syntax, not semantics. The model can still produce `{"name": "Alice", "age": -5}` — syntactically valid JSON but semantically wrong.

### Solution 2: Schema Validation + Retry

Generate normally, then validate against the schema:

```python
import jsonschema

for attempt in range(max_retries):
    response = model.generate(prompt)
    try:
        parsed = json.loads(response)
        jsonschema.validate(parsed, schema)
        return parsed
    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
        prompt += f"\nYour previous output was invalid: {e}\nPlease try again."

raise StructuredOutputFailure("Max retries exceeded")
```

**Tradeoff:** Works with any model, but costs extra tokens/latency on retries. Typically 1-5% of requests need a retry.

### Solution 3: Output Repair

Don't retry the whole generation — repair the broken output:

```python
def repair_json(raw_output):
    # Strip markdown code fences
    cleaned = re.sub(r'^```json\s*', '', raw_output)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # Fix trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Try parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: ask a model to fix it
        return model.generate(f"Fix this JSON: {raw_output}")
```

### The Fallback Chain Pattern

Production systems layer these approaches:

```
Step 1: Constrained decoding (grammar-enforced)
  ↓ If unavailable (model doesn't support it)
Step 2: Parse response, validate against schema
  ↓ If fails
Step 3: Attempt automated repair (regex, heuristics)
  ↓ If fails
Step 4: Retry with error feedback (up to 3 times)
  ↓ If fails
Step 5: Return structured error + fall back to human/rule-based system
```

---

## Part 2: Function Calling (Tool Use)

### How Function Calling Works

The model doesn't actually call functions. It generates a structured output describing which function to call and with what arguments. Your harness executes the function.

```
User: "What's the weather in San Francisco?"

Model output (structured):
{
  "tool_calls": [{
    "function": "get_weather",
    "arguments": {"city": "San Francisco", "units": "fahrenheit"}
  }]
}

Harness: Executes get_weather("San Francisco", "fahrenheit")
Result: {"temperature": 62, "condition": "foggy"}

Harness injects result back into context, model continues:
"The weather in San Francisco is 62°F and foggy."
```

### Tool Contracts: Defining What the Model Can Call

A tool definition is a contract between the model and your system:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city. Only supports US cities.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "City name, e.g. 'San Francisco'"
      },
      "units": {
        "type": "string",
        "enum": ["fahrenheit", "celsius"],
        "default": "fahrenheit"
      }
    },
    "required": ["city"]
  }
}
```

**Contract responsibilities:**
- Description tells the model *when* to use the tool
- Parameter schema tells the model *how* to call it
- Enum constraints prevent invalid values
- Required fields prevent incomplete calls

### Function Calling Failure Modes

| Failure | Example | Mitigation |
|---|---|---|
| **Wrong tool chosen** | Uses `search` when `database_query` was needed | Better descriptions, few-shot examples |
| **Hallucinated tool** | Calls `send_email` which doesn't exist | Constrained decoding on tool names |
| **Invalid arguments** | `{"city": 123}` instead of string | Schema validation + type coercion |
| **Missing required args** | `{}` when `city` is required | Validation + retry with error |
| **Extra arguments** | `{"city": "SF", "country": "US"}` | Strip unknown fields (lenient parsing) |
| **Malformed JSON** | `{"city": "San Francisco` | Repair or retry |
| **Correct call, wrong time** | Searches before reading available context | Better system prompt, ReAct pattern |

### Argument Validation Beyond Schema

Schema validation catches type errors. Production needs more:

```python
def validate_tool_call(tool_name, arguments):
    # Schema validation (catches type/format errors)
    schema_errors = validate_schema(tool_name, arguments)
    if schema_errors:
        return Retry(reason=schema_errors)
    
    # Business logic validation (catches semantic errors)
    if tool_name == "transfer_money":
        if arguments["amount"] > 10000:
            return RequireConfirmation("Large transfer requires approval")
        if arguments["to_account"] == arguments["from_account"]:
            return Reject("Cannot transfer to same account")
    
    if tool_name == "delete_record":
        if not arguments.get("confirm"):
            return Reject("Delete requires explicit confirmation")
    
    # Rate limiting
    if rate_limiter.exceeded(tool_name):
        return Reject("Tool call rate limit exceeded")
    
    return Execute(tool_name, arguments)
```

### Idempotency: When Tools Get Called Twice

In a retry loop, the same tool might be called multiple times. Some tools are safe to repeat (idempotent), others aren't:

| Tool | Idempotent? | Risk of duplicate call |
|---|---|---|
| `get_weather("SF")` | Yes | None — same result |
| `search("query")` | Yes | None — same results |
| `send_email(to, body)` | **No** | Duplicate email sent |
| `create_order(items)` | **No** | Duplicate order created |
| `increment_counter()` | **No** | Counter incremented twice |
| `set_status("active")` | Yes | Same final state |

**Mitigation patterns:**
- **Idempotency keys:** Attach a unique ID to each tool call. If the same ID is seen again, return the cached result.
- **At-most-once semantics:** Track which tool calls have been executed. Never execute the same call twice.
- **Destructive action confirmation:** Require explicit confirmation for non-idempotent actions before execution.

---

## Part 3: Agent Guardrails

### Loop Budgets

Agents in a ReAct loop can get stuck — retrying the same failing approach or calling tools endlessly:

```python
class AgentGuardrails:
    def __init__(self):
        self.max_iterations = 25          # Hard stop on loop count
        self.max_tool_calls = 50          # Total tool invocations
        self.max_tokens_generated = 50000 # Total output tokens
        self.max_wall_time = 300          # 5 minutes wall clock
        self.max_cost = 5.00              # Dollar cost cap
        
        self.tool_call_count = 0
        self.iteration_count = 0
        self.total_tokens = 0
        self.start_time = time.time()
    
    def check_budget(self, tokens_this_step):
        self.iteration_count += 1
        self.total_tokens += tokens_this_step
        
        if self.iteration_count > self.max_iterations:
            return Terminate("Max iterations reached")
        if self.tool_call_count > self.max_tool_calls:
            return Terminate("Tool call budget exhausted")
        if self.total_tokens > self.max_tokens_generated:
            return Terminate("Token budget exhausted")
        if time.time() - self.start_time > self.max_wall_time:
            return Terminate("Time budget exceeded")
        if self.estimate_cost() > self.max_cost:
            return Terminate("Cost budget exceeded")
        
        return Continue()
```

### Tool Budgets (Per-Tool Limits)

Some tools are expensive or dangerous. Limit them independently:

```python
tool_budgets = {
    "web_search": {"max_calls": 10, "cooldown_seconds": 2},
    "code_execute": {"max_calls": 20, "timeout_per_call": 30},
    "send_email": {"max_calls": 1, "requires_confirmation": True},
    "database_write": {"max_calls": 5, "requires_confirmation": True},
    "file_delete": {"max_calls": 3, "requires_confirmation": True},
}
```

### Termination Conditions

When should an agent stop?

```python
termination_conditions = [
    # Success conditions
    "model outputs final_answer",
    "task objective achieved (verified by assertion)",
    "user indicates satisfaction",
    
    # Failure conditions  
    "budget exhausted (any budget)",
    "same tool called 3 times with same arguments (stuck loop)",
    "3 consecutive errors from tools",
    "model outputs 'I cannot complete this task'",
    
    # Safety conditions
    "attempted action on blocklist",
    "output contains PII/secrets",
    "cost exceeds threshold",
]
```

### Stuck Loop Detection

```python
def detect_stuck_loop(history):
    # Check if the last 3 actions are identical
    recent_actions = [h for h in history[-6:] if h.type == "action"]
    if len(recent_actions) >= 3:
        if all(a.tool == recent_actions[0].tool and 
               a.args == recent_actions[0].args 
               for a in recent_actions[-3:]):
            return StuckLoop(
                action=recent_actions[0],
                suggestion="Try a different approach or tool"
            )
    return None
```

---

## Part 4: Model Routing and Fallback Logic

### Why Route Between Models?

Not every request needs the most powerful (and expensive) model:

```
"What's 2+2?"                    → Small model (fast, cheap)
"Summarize this email"           → Medium model (good enough)
"Debug this concurrency bug"     → Large model (needs deep reasoning)
"Generate a creative story"      → Large model with high temperature
```

### Routing Strategies

```python
def route_request(request):
    # Complexity-based routing
    estimated_complexity = classify_complexity(request)
    
    if estimated_complexity == "simple":
        return Model("claude-3-haiku", max_tokens=500)
    elif estimated_complexity == "medium":
        return Model("claude-3-5-sonnet", max_tokens=2000)
    else:
        return Model("claude-opus-4", max_tokens=8000)

def classify_complexity(request):
    # Heuristics:
    # - Short factual questions → simple
    # - Multi-step reasoning, code → complex
    # - Long input requiring analysis → complex
    # Can also use a cheap classifier model for this decision
    pass
```

### Graceful Fallback Logic

When the primary model fails (rate limit, error, timeout), degrade gracefully:

```python
async def generate_with_fallback(prompt, models=None):
    models = models or [
        {"model": "claude-opus-4", "timeout": 30},
        {"model": "claude-sonnet-4", "timeout": 20},
        {"model": "claude-3-haiku", "timeout": 10},
    ]
    
    for config in models:
        try:
            response = await call_model(
                model=config["model"],
                prompt=prompt,
                timeout=config["timeout"]
            )
            if validate_response(response):
                return response
        except (RateLimitError, TimeoutError, ServerError) as e:
            log.warning(f"{config['model']} failed: {e}")
            continue
    
    # All models failed — return degraded response
    return DegradedResponse(
        message="Service temporarily limited. Please try again.",
        fallback_used=True
    )
```

### Degraded-Mode UX

When falling back to a weaker model, the user experience should adapt:

```
Normal mode (primary model available):
  → Full reasoning, complex tool use, detailed responses

Degraded mode (fell back to simpler model):
  → Shorter responses, fewer tool calls
  → Surface warning: "Currently operating in limited mode"
  → Disable complex features (multi-step agents, code execution)
  → Queue non-urgent requests for when primary recovers
```

---

## Key Takeaways

1. **Constrained decoding guarantees syntax** — the model physically cannot produce invalid JSON when grammar-enforced
2. **Schema validation catches type errors, but business logic validation catches semantic errors** — you need both
3. **Fallback chains handle failures gracefully** — constrained decoding → validation → repair → retry → error
4. **Tool contracts define what models can call** — descriptions control *when*, schemas control *how*
5. **Idempotency matters** — non-idempotent tools need deduplication and confirmation gates
6. **Agent guardrails prevent runaway costs** — loop budgets, tool budgets, stuck loop detection, and termination conditions
7. **Model routing matches request complexity to model capability** — save money and latency on simple requests
8. **Graceful degradation > hard failures** — always have a fallback path

---

## Key Papers & Resources

- **Outlines: Structured Text Generation** (Willard & Louf, 2023) — grammar-constrained decoding
- **Gorilla: Large Language Model Connected with Massive APIs** (Patil et al., 2023) — function calling accuracy
- **ToolBench** (Qin et al., 2023) — benchmarking tool use
- **Anthropic Tool Use Documentation** (2024) — production tool calling patterns
- **OpenAI Function Calling Documentation** (2024) — structured output and function schemas

---

## What's Next

The model generates structured output and calls tools reliably. But how do we know if the entire system is working correctly at scale? Evals, observability, and cost tracking are the discipline that catches regressions before users do. Chapter 14.
