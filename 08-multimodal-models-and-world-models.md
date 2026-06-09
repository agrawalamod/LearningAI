# Chapter 8: Multimodal Models, Video Understanding & World Models

## The Big Picture

You asked several connected questions:
1. How do video-language models (like Qwen) work?
2. Can they annotate activities they've never seen (like egocentric farming)?
3. What are world models?
4. Can probability-based systems ever "understand" physics?
5. How important is egocentric data?

Let's build this up from foundations.

---

## Part 1: How Multimodal Models Work

### The Core Architecture: Vision Encoder + LLM

A multimodal model combines:
- A **vision encoder** (typically a Vision Transformer / ViT) that turns images/video into vectors
- A **language model** (standard transformer) that processes those vectors alongside text

```
Image → [Vision Encoder] → Visual tokens (vectors)
Text  → [Tokenizer]     → Text tokens (vectors)

Both → [LLM Transformer] → Output text
```

**The key insight:** The visual tokens are projected into the same embedding space as text tokens. To the LLM, an image is just another sequence of "tokens" — it processes them with the same attention mechanism.

### Example: How Qwen-VL Processes a Video

```
Input: [Frame 1 tokens][Frame 2 tokens]...[Frame N tokens] + "What is happening?"

The model sees visual tokens as context, just like it would see text context.
It attends to the visual tokens and generates: "A person is chopping vegetables."
```

### Training Data for Multimodal Models

These models are trained on massive datasets of:
- Image-caption pairs (billions from the internet)
- Video-caption pairs (millions of videos with descriptions)
- Visual question-answering (image + question → answer)
- Interleaved image-text documents (web pages, articles)

---

## Part 2: Can Qwen Annotate Egocentric Video It Has Never Seen?

This connects directly to the generalization question from Chapter 4.

### What Qwen HAS likely seen:
- Third-person cooking videos with captions ("chef is dicing onions")
- YouTube first-person videos (some egocentric content exists online)
- Action recognition datasets (people doing various activities)
- The *concepts* of farming, cleaning, cooking described in text

### What it might NOT have seen:
- Your friend's specific egocentric dataset of rare activities
- Activities from cultures or contexts with minimal internet presence
- Novel tool usage that has no analog in existing video data

### The Answer: A Spectrum

| Activity Type | Can Qwen annotate it? | Why |
|---|---|---|
| Cooking pasta (first-person) | Yes, well | Seen from many angles, concept well-understood |
| Farming rice (first-person) | Probably yes, roughly | Concept known, though angle might be unusual |
| A completely novel craft unique to one village | Poorly | Might describe hand movements but not the activity semantics |
| Abstract activities with no visual analog in training | No | Would hallucinate or give generic descriptions |

### The Key Principle:

**If the underlying concept exists in training data (even from a different angle/modality), the model can transfer.** Changing viewpoint (third-person → first-person) is a relatively easy generalization because the *activity* is the same — only the camera position changed.

But if the concept itself is absent, no amount of viewpoint robustness helps.

---

## Part 3: What Are World Models?

A **world model** is a system that can:
1. Predict what happens next in a physical environment
2. Understand cause and effect
3. Simulate "what would happen if..."
4. Reason about physics, space, and time

### The Dream:
An AI that understands that:
- If you push a glass off a table, it falls and breaks
- If you pour water into a cup, the cup fills up
- If you throw a ball, it follows a parabolic trajectory

### Why LLMs Aren't World Models (Yet):
LLMs know *text about* physics. They can describe what happens when you drop a glass. But their "understanding" comes from text descriptions, not from directly modeling physical dynamics.

```
LLM: Knows "glass falls → breaks" as a text pattern
World model: Actually simulates the physics of falling and shattering
```

---

## Part 4: How Are World Models Being Built?

### Approach 1: Video Prediction Models

Train a model to predict the next frames of a video:

```
Given: Frames 1-10 of a ball being thrown
Predict: Frames 11-20 (where the ball will be)
```

If the model can predict correctly, it must have some internal representation of physics (trajectory, gravity, occlusion).

**Examples:**
- **Sora** (OpenAI) — generates physically plausible video from text descriptions
- **Genie 2** (DeepMind) — generates interactive 3D worlds from images
- **UniSim** (research) — learns physics simulation from video

### Approach 2: Learned Simulators

Train a neural network to replace physics engines:

```
Input: Current state of a physical system (positions, velocities)
Output: Next state of the system
```

These are faster than traditional physics simulators and can learn from real-world data.

### Approach 3: Embodied Learning

Robots learn world models by interacting with the real world:

```
Action: Push the cup
Observation: Cup moves, water spills
Update model: "Pushing a cup with liquid → spilling"
```

This is why companies collect **egocentric data** — it captures the causal relationship between actions and outcomes.

---

## Part 5: Why Egocentric Data Is Important for World Models

**Third-person video** shows *what happened*: "The person picked up the cup."

**Egocentric (first-person) video** shows *what it's like to do it*:
- What you see when you reach for a cup
- How objects move relative to your hands
- The causal link between your actions and the world's response

### This matters because:

```
Third-person: Observing physics from outside
Egocentric: Experiencing physics from inside (action → consequence)
```

For a world model to be useful for **robots** or **embodied AI**, it needs the egocentric perspective — because that's how an agent will interact with the world.

### Your friend's intuition is right:
Egocentric data is valuable because:
1. It captures **action-outcome pairs** (I pushed → it moved)
2. It's **agent-centric** (what I see → what I should do)
3. It's **causal** (my action caused this effect)
4. It's **underrepresented** in existing datasets (most video is third-person)

**Key datasets:**
- **Ego4D** (Meta) — thousands of hours of egocentric video
- **Epic-Kitchens** — first-person cooking videos with annotations

---

## Part 6: Can Probability-Based Systems Ever Understand Physics?

This is your deepest question. Here are the perspectives:

### Argument: "No, probabilities aren't enough"

**Yann LeCun's position:** Current LLMs operate in "token space" (text). Real understanding requires operating in a continuous representation of the physical world. You can't learn physics just from reading about physics.

He proposes **JEPA** (Joint Embedding Predictive Architecture) — models that predict in representation space rather than pixel space. The idea: learn abstract representations of how the world works, without generating pixels.

### Argument: "Yes, if you scale enough"

**The scaling hypothesis:** If you train on enough video (trillions of frames), the model must develop internal representations of physics to make accurate predictions. If it can predict where a ball will be in the next frame, it has *implicitly* learned gravity — even if it can't articulate Newton's laws.

**Evidence for this:**
- Sora (video generation) shows understanding of reflections, shadows, basic physics
- Language models trained on enough physics text can solve novel physics problems
- Internal probing of LLMs reveals spatial representations that weren't explicitly taught

### Argument: "It's the wrong question"

**Pragmatic view:** Maybe the question isn't "do they understand?" but "can they predict accurately?" If a model reliably predicts physical outcomes, does it matter whether it "truly understands" physics in some philosophical sense?

### The Current Reality (2025):

```
Text LLMs:        Know ABOUT physics (from text). Limited simulation ability.
Video models:     Learn some implicit physics. Inconsistent. Break on edge cases.
Physics engines:  Perfect simulation. Zero generalization to new scenarios.
World models:     The goal — combine generalization with physical accuracy. Not yet achieved.
```

---

## Part 7: The Path Forward

The consensus in the field (as of 2025):

1. **Pure text is insufficient** for world models — you need grounding in sensory data
2. **Pure video prediction is insufficient** — you need action-conditioned models (what happens IF I do X)
3. **Egocentric + action data is critical** — it provides the causal link between actions and outcomes
4. **We're in early days** — current models show flickers of physical understanding but break easily

The gap between "can generate a plausible-looking video of a ball bouncing" and "truly understands physics" is still large. But it's shrinking.

---

## Key Takeaways

1. Multimodal models project images/video into the same space as text, then process with standard attention
2. They CAN annotate novel viewpoints of known activities (egocentric cooking → works if concept is known)
3. They CANNOT annotate truly unseen concepts (same limit as text LLMs)
4. World models aim to understand physics and causality, not just describe them
5. Egocentric data is crucial because it captures action→outcome causality
6. Whether probabilities can "understand" physics is open — current evidence suggests partial but incomplete understanding
7. The field is moving toward: video prediction + action conditioning + egocentric data = world models

---

## Key Papers & Resources

- **Video PreTraining (VPT)** (Baker et al., 2022) — learning from internet video of gameplay
- **Ego4D** (Grauman et al., 2022) — massive egocentric video dataset
- **A Path Towards Autonomous Machine Intelligence** (LeCun, 2022) — JEPA and world model architecture
- **World Models** (Ha & Schmidhuber, 2018) — early influential work on learned world simulators
- **Sora Technical Report** (OpenAI, 2024) — video generation with implicit physics

---

## What's Next

Now let's talk about something practical and important: latency. If you're building your own model, you need to understand what makes it fast or slow. Chapter 9.
