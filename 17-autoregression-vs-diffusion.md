# Chapter 17: Autoregression vs Diffusion, and Sparsity via MoE

## The Question This Chapter Answers

Why is an LLM called "autoregressive," why is diffusion not, and why does everyone claim diffusion generates text faster when it clearly does more total computation?

The short answer, which the rest of this chapter unpacks: **both paradigms are iterative, but they iterate along different axes.** Autoregression iterates over sequence position. Diffusion iterates over noise level. Everything else, including the speed argument, falls out of that one distinction.

---

## Part 1: Why "Autoregressive" Is the Correct Technical Term

The word comes from time-series statistics, not from deep learning. An autoregressive process is one where the current value is a function of its own previous values. "Auto" means self, "regression" means predicting from inputs. You are regressing the sequence on itself.

For a sequence of tokens, the chain rule of probability gives you an exact factorization of the joint distribution:

```
P(x₁, x₂, ..., xₙ) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(xₙ|x₁...xₙ₋₁)
```

An LLM models exactly one term of this product at a time: `P(xₜ | x₁...xₜ₋₁)`. Generation walks the factorization left to right, feeding each sampled token back in as input to the next conditional. The model's own output becomes its own input. That self-feeding is what makes it autoregressive.

Three consequences follow directly from this factorization, and they explain most of what people find frustrating about LLMs:

**The factorization is exact, so the likelihood is tractable.** You can compute the exact log-probability of any sequence. This is why perplexity is a meaningful metric for LLMs and why training is a clean maximum-likelihood problem with cross-entropy loss. No approximation, no variational bound.

**The ordering is imposed and permanent.** The chain rule is valid for any ordering of the variables, but you must pick one, and standard LLMs pick left to right. Once `xₜ` is sampled, every subsequent conditional treats it as fixed ground truth. There is no mechanism to revise it. This is the source of the "stuck with a bad token" problem.

**Sequential depth equals sequence length.** Generating N tokens requires N forward passes that cannot be reordered or parallelized, because pass `t+1` needs the output of pass `t`. This is a hard serial dependency, not an implementation limitation.

Note that training does not have this problem. During training the entire target sequence is known, so causal masking lets you compute all N conditionals in a single parallel forward pass. Autoregression is parallel at training time and strictly serial at inference time. That asymmetry is the root of the entire LLM serving problem from Chapter 12.

---

## Part 2: Why Diffusion Is Not Autoregressive

Diffusion does not factorize over sequence position at all. It factorizes over a noise schedule.

The forward process progressively corrupts data toward pure noise, and the model learns to reverse it:

```
Forward  (fixed):    x₀ → x₁ → x₂ → ... → x_T   (data to noise)
Reverse  (learned):  x_T → ... → x₂ → x₁ → x₀   (noise to data)
```

Each `xₜ` here is a **complete sequence or image at noise level t**, not a single token at position t. The subscript indexes corruption level, not position. This is the single most common point of confusion when people move between the two paradigms.

So the factorization looks like this:

```
P(x₀) = ∫ P(x_T) · Π P(x_{t-1} | x_t) dx_{1..T}
```

At every reverse step the model sees the whole object and updates the whole object. There is no notion of "the part I have already committed to" versus "the part I have not reached yet." Every position is under revision at every step until the process terminates.

### The Comparison That Matters

| | Autoregressive | Diffusion |
|---|---|---|
| Iterates over | sequence position | noise level |
| State at step t | first t tokens, final | all N positions, provisional |
| Conditioning | past tokens only (causal) | entire sequence (bidirectional) |
| Sequential depth | N (one per token) | K (one per denoising step), independent of N |
| Can revise earlier output | no | yes, that is the entire mechanism |
| Likelihood | exact, tractable | variational lower bound (ELBO) |
| Output length | dynamic, stops when it emits EOS | fixed by canvas size, decided up front |

That last row is an underappreciated practical cost of diffusion. An AR model decides when to stop by emitting an end-of-sequence token. A diffusion model must be allocated a canvas before it starts generating, so variable-length output requires extra machinery. We will see how DiffusionGemma solves this, and the solution is where the hybrids come in.

### Why This Makes Diffusion Good at Constraint Satisfaction

Bidirectional attention over a provisional canvas is qualitatively different from causal attention over committed history. Consider Sudoku. A digit in the first cell is constrained by cells in the last row. An AR model generating left to right must commit cell 1 before it has any representation of what cells 40 through 81 will contain. It cannot backtrack.

Google's DiffusionGemma release uses precisely this example. The [developer guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/) reports that the base model essentially never solves Sudoku puzzles, but a light supervised fine-tune on Sudoku data reaches roughly 80% correctness while also needing fewer denoising steps. The relevant point is not the accuracy number, it is that the *architecture* permits global constraint propagation at all. Every canvas position attends to every other one on each pass, so information about a conflict flows in both directions in a single step.

This generalizes to any task where the correct output has non-local structural constraints: code that must satisfy a type signature declared later, structured output where a closing bracket constrains earlier content, planning where a goal state constrains intermediate steps.

---

## Part 3: Discrete Diffusion — Making Diffusion Work on Text

Image diffusion adds Gaussian noise to continuous pixel or latent values. You cannot add Gaussian noise to token ID 4,721. Tokens are discrete symbols, not points on a continuum, so "slightly noisy" has no obvious meaning. Three families of solutions emerged.

### Approach 1: Diffuse in Continuous Embedding Space

Map tokens to their embedding vectors, run standard continuous diffusion in that space, and round back to the nearest token at the end. This was the earliest approach (Diffusion-LM and similar work).

It mostly failed to scale. The embedding space is not smooth in the way pixel space is. Small perturbations to an embedding frequently cross a decision boundary into a semantically unrelated token, so the rounding step at the end is lossy and unstable. Trying to keep the process near valid embeddings requires awkward regularization that fights the diffusion objective.

### Approach 2: Masked Diffusion

Define corruption as replacement with a special `[MASK]` token. The forward process progressively masks more of the sequence; the reverse process predicts what is behind each mask. This is essentially BERT's objective generalized to a full schedule of masking ratios rather than a single fixed 15%.

This works, scales, and produced the first credible diffusion language models (LLaDA and similar 8B-class models). It has one structural weakness, which Google's [technical explainer](https://ai.google.dev/gemma/docs/diffusiongemma/explained) states directly: once a mask is resolved into a concrete token, that token is locked. There is no path back to the masked state. So masked diffusion recovers parallelism but partially loses the self-correction property that made diffusion attractive in the first place. It is monotonic, and monotonic decoding is closer to autoregression than it first appears.

### Approach 3: Uniform State Diffusion

This is what DiffusionGemma uses, and it is the cleaner solution. Noise is not a special token, it is **a random token sampled from the actual vocabulary**.

The forward process replaces real tokens with random vocabulary entries. The reverse process starts from a canvas of pure random tokens and repeatedly asks, for every position simultaneously, "does this token look like signal or like noise given everything else on the canvas?" High-confidence tokens are retained. Positions whose confidence falls below threshold get re-noised with a fresh random token and revisited on a later pass.

The critical property is that **corruption and generation live in the same space**. There is no absorbing `[MASK]` state to get trapped in, so any position can be revised at any step. Confident regions stabilize early and act as context that helps resolve their neighbors, and the sequence progressively snaps into coherence rather than being filled in monotonically.

### How DiffusionGemma Is Actually Built

Worth understanding concretely, because the architecture is more pragmatic than the pure paradigm suggests.

The model is a 26B Mixture of Experts on the Gemma 4 backbone that activates about 3.8B parameters per token, released under Apache 2.0. A single backbone toggles between two attention modes rather than using two separate models:

```
Prefill / Incremental Prefill  →  CAUSAL attention
    Ingests prompt, writes KV cache.
    Runs once for the prompt, then once per completed block.

Denoising                      →  BIDIRECTIONAL attention
    Iteratively refines the 256-token canvas.
    Every canvas position attends to every other position, plus the KV cache.
```

Two additional mechanisms matter:

**Self-conditioning.** After each denoising pass, the model multiplies its output probability distribution by the token embedding table to produce a vector summarizing what it just predicted and how confident it was, then feeds that into the next step. The denoiser gets memory of its own previous guess, which stabilizes the trajectory.

**Multi-canvas block sampling.** A canvas is fixed at 256 tokens, so long outputs are produced by chaining canvases. Denoise a full 256-token block, commit it to the KV cache, initialize a fresh canvas conditioned on that committed history, repeat.

That last mechanism is not a detail. It is a hybrid, and it deserves its own section.

---

## Part 4: Why Autoregression *Is* Used for Images (Correcting a Common Claim)

The widely repeated claim that autoregression does not work for images was true in 2021 and is false now. Getting this right matters because the actual failure mode was misdiagnosed for years.

### What Actually Went Wrong Early On

Early autoregressive image models (PixelRNN, PixelCNN, the original DALL-E, Parti) tokenized images into a grid and generated tokens in **raster-scan order**: left to right, top to bottom, like reading a page. This produced three problems, none of which is inherent to autoregression:

**Raster order is a lie about image structure.** In text, left-to-right is the true generative order of the data. In an image, the pixel to the left and the pixel above are both immediate neighbors, but raster ordering makes one of them "the recent past" and the other "distant past." The imposed causal order actively destroys the 2D locality that matters.

**Token counts explode quadratically.** A 1024×1024 image at 16×16 patches is 4,096 tokens for one image. At one forward pass per token, this is hopeless.

**Errors accumulate along a meaningless direction.** A mistake in the top-left corner propagates through thousands of subsequent conditionals with no way to revise it, and the propagation direction has no relationship to image semantics.

### The Fix: Change the Ordering, Not the Paradigm

[Visual Autoregressive Modeling (VAR)](https://arxiv.org/abs/2404.02905) kept autoregression and replaced raster ordering with **next-scale prediction**. The model generates a coarse low-resolution token map first, then conditions on it to generate the next resolution, and so on up to full resolution. Within any single scale, all tokens are produced in one parallel pass. The autoregression is over *scale*, not over *position*.

The results ended the debate. On ImageNet 256×256, VAR improved FID from 18.65 to 1.80 and Inception Score from 80.4 to 356.4 over the raster-scan AR baseline, with roughly 20x faster inference, and beat Diffusion Transformers on quality, speed, data efficiency, and scaling behavior.

The lesson is that autoregression is a factorization strategy, and factorization strategies are only as good as the ordering they impose. Coarse-to-fine is a natural generative order for images. Left-to-right is not.

### The Punchline: The Two Paradigms Are Not Cleanly Distinct

Once you see next-scale AR working, the boundary starts dissolving. Recent theoretical work makes this explicit. ["Scale-Wise VAR is Secretly Discrete Diffusion"](https://arxiv.org/html/2509.22636v1) shows that VAR equipped with a Markovian attention mask is mathematically equivalent to a discrete diffusion process. A companion result, ["Multi-scale Autoregressive Models are Laplacian, Discrete, and Latent Diffusion Models in Disguise"](https://arxiv.org/html/2510.02826v1), reaches the same conclusion from the Laplacian pyramid direction.

This should be intuitive by now. Coarse-to-fine generation *is* progressive refinement. Diffusion's noise schedule is a particular way of defining "coarse." Both are iterative refinement procedures; they differ in what they use as the corruption operator and whether the refinement steps are causally masked. "Autoregressive versus diffusion" is better understood as a spectrum parameterized by choice of iteration axis than as two separate species.

### Where Video Sits

Video is the case where hybrids are obviously correct rather than merely interesting, because video has two genuinely different axes. Space within a frame has no natural causal order. Time across frames has a very strong one: frame `t+1` is caused by frame `t`.

So the sensible architecture is autoregressive over frames or chunks of frames, diffusion within each frame. This is roughly what current video generation systems do, and it is a direct consequence of matching each iteration axis to the structure actually present in that axis.

---

## Part 5: Hybrids — What "Applying Autoregression to Diffusion" Means

Two distinct things get called hybrids. Worth separating them.

### Hybrid Type 1: Block Autoregressive Diffusion (Outer AR, Inner Diffusion)

This is what DiffusionGemma does and what your question was pointing at.

```
Block 1: [256-token canvas]  ← K parallel denoising steps
              ↓ commit to KV cache
Block 2: [256-token canvas]  ← K parallel denoising steps, conditioned on block 1
              ↓ commit to KV cache
Block 3: [256-token canvas]  ← ...
```

Autoregression across blocks, diffusion within a block. This is not a compromise for its own sake; it solves three concrete problems that pure diffusion has:

**Variable length.** Pure diffusion requires committing to output length before generating. Chaining blocks lets the model keep going until it decides to stop, recovering AR's dynamic length behavior.

**Bounded attention cost.** Bidirectional attention over the full canvas is quadratic in canvas size. Fixing the canvas at 256 tokens caps that cost, and cross-block context arrives through the KV cache at linear cost instead.

**Long-range coherence.** Committed blocks are stable context. Pure diffusion over a very long canvas has to hold global structure in a fully provisional state throughout, which is unstable.

**Where it helps:** long-form generation, code, streaming output, anything where you want low latency but do not know the output length up front.

**Where it does not help:** tasks whose constraints span more than one block. A Sudoku grid fits in one canvas, so bidirectional refinement can see the whole problem. A constraint linking token 50 to token 4,000 crosses a block boundary, and once block 1 is committed to the KV cache it is as frozen as any AR output. The hybrid inherits AR's irreversibility at block granularity. It buys you parallelism inside a window; it does not buy you global revisability.

### Hybrid Type 2: Diffusion Inside an Autoregressive Frame (Outer AR, Inner Diffusion, Different Axis)

The video case above. AR over the axis with real causal structure (time), diffusion over the axis without it (space). Also describes VAR-style coarse-to-fine when you view scale as the AR axis.

**Where it helps:** any data with a genuine causal axis plus one or more non-causal axes. Video, audio with temporal structure, sequential decision making with high-dimensional actions.

**Where it does not help:** data that is uniformly non-causal (a single image has no axis that deserves AR treatment, which is why pure diffusion works well there) or uniformly causal with low-dimensional steps (pure AR text generation, where per-step dimensionality is one token and there is nothing to parallelize within a step).

### The Design Rule

Match your iteration axis to the structure in your data. Use autoregression along axes with real causal ordering. Use diffusion along axes without one. Most interesting data has both kinds of axis, which is why hybrids keep winning.

---

## Part 6: The Speed Question — Why "Diffusion Is Faster" Is Both True and Misleading

This is the part most explanations get wrong, so let us do the accounting explicitly.

### Count the Work Honestly

Generate N = 256 tokens. Diffusion uses K = 32 denoising steps.

```
AUTOREGRESSIVE
  Sequential steps:        256   (one per token, strictly ordered)
  Token-forward-passes:    256   (with KV cache, one new token per pass)

DIFFUSION (256-token canvas, 32 steps)
  Sequential steps:         32   (one per denoising pass)
  Token-forward-passes:  8,192   (32 passes × 256 positions each)
```

Diffusion performs **32 times more arithmetic** and yet finishes in **8 times fewer sequential steps**. Both facts are true simultaneously. If you were counting FLOPs, diffusion looks catastrophically worse. If you were counting serial round trips, it looks much better.

Which one determines wall-clock time depends entirely on whether the hardware is starved for compute or starved for memory bandwidth.

### The Actual Reason: Decode Is Memory-Bound

Chapter 12 established this and it is the whole explanation. During autoregressive decode, each step must stream the model's weights from HBM into the compute units in order to process a single token. The arithmetic per step is trivial; the memory traffic is not. The tensor cores sit idle waiting on loads.

Google's explainer makes the consequence unusually clear: because weights are loaded once per step regardless of batch size, generating a token takes nearly the same wall-clock time for 1 user as for 256 users batched together.

Read that again, because it is the crux. In autoregressive serving, **batching is close to free**. That is a statement about spare capacity. The hardware can do 256 users' worth of work in the time it takes to do one user's worth, which means when you serve a single user you are wasting 255/256 of the available compute.

Diffusion spends that waste. Instead of computing 1 token for 256 users, compute 256 tokens for 1 user. Same weight loads, same memory traffic, same wall-clock per step, 256x more useful output for the single user. The bottleneck shifts from memory bandwidth to compute, which is the regime GPUs are actually designed for.

Reported numbers for DiffusionGemma: up to 4x faster token generation, 700+ tokens/sec on an RTX 5090 and 1000+ tokens/sec on a single H100.

### The Catch, Which Is Large

The speedup is **single-user latency**, funded entirely by idle compute. It is not a throughput improvement, and under load it evaporates.

```
Single user, AR:          compute mostly idle  →  diffusion converts idle compute into speed  ✓
256 users batched, AR:    compute already saturated  →  no idle capacity left to spend  ✗
```

With a large batch, autoregression has already filled the GPU. There is no free compute for diffusion to claim, and now diffusion's 32x FLOP overhead is a real cost paid in real time. Under high concurrency, autoregression should win on tokens per second per dollar.

There is a visible tell in Google's own recommended vLLM configuration: `--max-num-seqs 4`. That caps concurrent sequences at four. A production autoregressive deployment would run that number in the hundreds. The configuration is telling you what regime this architecture is for.

### So Who Is Actually Faster

Neither, unconditionally. The honest answer is that they optimize different metrics:

| Scenario | Winner | Why |
|---|---|---|
| Single user, local or on-device | Diffusion | Converts idle compute into latency reduction |
| Interactive latency-critical, low concurrency | Diffusion | Fewer serial round trips |
| High-concurrency API serving | Autoregressive | Compute already saturated by batching; FLOP overhead becomes real |
| Compute-constrained hardware | Autoregressive | Cannot afford 32x FLOPs |
| Memory-bandwidth-constrained hardware | Diffusion | Precisely the bottleneck it sidesteps |
| Long outputs, unknown length | Block hybrid | Needs AR's dynamic termination |
| Global constraint satisfaction | Diffusion | Bidirectional refinement, revisability |

When someone says "diffusion generates text faster," the accurate version is: **diffusion trades a large amount of extra parallel computation for a large reduction in sequential depth, which is a winning trade exactly when your compute would otherwise sit idle waiting on memory.** That condition holds for single-user and on-device inference. It fails for batched serving.

This is the same structural bargain as speculative decoding from Chapter 12. Speculative decoding spends draft-model compute to verify several tokens in one target-model pass. Diffusion spends canvas-width compute to resolve many positions per pass. Both are converting surplus FLOPs into reduced serial depth. Diffusion is simply the more aggressive version, and it is architectural rather than bolted on.

---

## Part 7: Where This Is Heading

Three observations that follow from the above.

**Quality parity is not yet established.** Frontier text models remain autoregressive. Diffusion language models are competitive at their size class but have not demonstrated they match the best AR models on hard reasoning at scale. The exact-likelihood training objective of AR is a real advantage that diffusion's variational bound does not have, and it may matter more than the parallelism.

**The taxonomy is collapsing.** Once VAR is provably equivalent to discrete diffusion, and once DiffusionGemma is autoregressive at block level and diffusive within blocks, "AR versus diffusion" stops being a useful binary. The productive question is not which paradigm but which axes you iterate over, what corruption operator you use, and where you place causal masks.

**Hardware trends favor diffusion.** Compute per chip has been growing faster than memory bandwidth for years. Every year that gap widens, the pool of idle compute during autoregressive decode grows, and the trade diffusion offers gets cheaper. If that trend holds, parallel-refinement decoding becomes more attractive over time regardless of which paradigm currently produces better text.

---

## Part 8: Mixture of Experts, and the Two Different Ways a Model Can Be "Smaller Than It Looks"

DiffusionGemma is built on the Gemma 4 26B-A4B Mixture of Experts backbone, so MoE belongs in this chapter. But there is a naming trap in the Gemma 4 family that is worth clearing up first, because two completely different techniques both produce a parameter count smaller than the headline number, and they are frequently conflated.

### First, a Correction Worth Making Explicitly

**E2B and E4B are not Mixture of Experts models.** Google's [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) lists them in the *Dense Models* table. The only MoE model in the family is the 26B-A4B.

The two letters encode different things:

```
E = Effective parameters    (E2B, E4B)      →  DENSE model, memory-side trick
A = Active parameters       (26B-A4B)       →  SPARSE model, compute-side trick
```

`E` is about where parameters *live*. `A` is about which parameters *run*. Getting these confused leads to the wrong deployment reasoning, so we will take them separately.

### What Mixture of Experts Actually Is

In a standard dense transformer block, every token passes through the same feed-forward network (FFN). The FFN is typically about two thirds of the model's parameters, and every single one of them is multiplied for every single token.

MoE replaces that one FFN with many smaller FFNs called experts, plus a small learned **router** that decides which experts each token visits.

```
DENSE BLOCK
  token → attention → [ one big FFN ]              → out
                        all params used

MoE BLOCK
  token → attention → router picks top-k of N      → out
                      ┌──────────────────────────┐
                      │ expert 1   ...  expert N │   only k of N run
                      └──────────────────────────┘
```

The router is usually just a linear layer producing a score per expert. Take the top-k scoring experts, run the token through only those, and combine their outputs weighted by the router's scores. In Gemma 4's MoE, k is 8 and N is 128, so each token touches roughly 6% of the available experts.

The load-bearing property: **total parameter count and per-token compute are now decoupled.** You can grow the model's capacity by adding experts without increasing the FLOPs any individual token costs. Parameters store knowledge, activated parameters cost compute, and MoE lets you buy the former without paying for the latter.

Three mechanisms make this work in practice, and they are where implementations differ:

**Load balancing.** Left alone, routers collapse. A few experts get chosen constantly, most are never selected and never receive gradient, and you have effectively trained a small dense model wearing a large model's parameter count. Training adds an auxiliary loss that penalizes imbalanced routing to force utilization across all experts.

**Shared experts.** Gemma 4's MoE uses 8 active out of 128 total **plus 1 shared expert**. The shared expert processes every token unconditionally. The reasoning is that some computation is genuinely universal across all tokens, and forcing the routed experts to each rediscover it wastes their capacity. Pinning the common part in a shared expert frees the routed ones to specialize.

**Routing is learned, not semantic.** A persistent misconception is that experts specialize into human-legible domains, a "Python expert" and a "French expert." Interpretability work does not support this. Specialization tends to be distributed and shallow, often tracking token-level or syntactic features rather than topics. Do not reason about MoE behavior as if you could name the experts.

### The Catch That Determines Your Deployment

MoE reduces compute. It does **not** reduce memory.

```
Gemma 4 26B-A4B
  Parameters resident in memory:  25.2B   ← you must hold all of them
  Parameters computed per token:   3.8B   ← only these cost FLOPs
```

Any token might route to any expert, so every expert's weights have to be available. You get the inference speed of roughly a 4B dense model while paying the memory footprint of a 25B model. Google's model card states the practical upshot plainly: the 26B-A4B is an excellent choice for fast inference relative to the dense 31B because it runs nearly as fast as a 4B model.

This is why MoE is a *server and workstation* technique, not a phone technique. It converts abundant memory into quality at fixed latency. On a device where memory is the binding constraint, MoE gives you nothing and costs you plenty.

Which is exactly why Gemma 4's edge models do something else entirely.

### What E2B and E4B Actually Do: Per-Layer Embeddings

Official numbers from the model card:

| | Gemma 4 E2B | Gemma 4 E4B |
|---|---|---|
| Effective parameters | 2.3B | 4.5B |
| Total with embeddings | 5.1B | 8B |
| Layers | 35 | 42 |
| Sliding window | 512 tokens | 512 tokens |
| Context length | 128K | 128K |
| Vocabulary | 262K | 262K |
| Modalities | text, image, audio | text, image, audio |
| Architecture class | **Dense** | **Dense** |

Note the gap between the two parameter rows. E2B carries 5.1B parameters but behaves like a 2.3B model. That gap is Per-Layer Embeddings, and the mechanism is different in kind from MoE.

Normally a transformer has one embedding table at the input. A token is looked up once, converted to a vector, and that single vector carries all of the token's identity information through every subsequent layer. PLE instead gives **each decoder layer its own small embedding table for every token**. As a token's representation moves up the stack, each layer injects a fresh layer-specific signal for that token alongside the residual stream.

The efficiency argument is about the nature of the operation. With a 262K vocabulary and 35 layers, those per-layer tables are enormous in raw parameter count. But an embedding table is only ever used for **lookup**. There is no matrix multiplication against it and no gradient-time cost comparable to a weight matrix during inference. So:

```
Embedding tables:  huge in parameter count, trivially cheap to use (indexed lookup)
                   → can be kept in cheap memory, or offloaded off the accelerator

Compute weights:   what actually gets multiplied every forward pass
                   → this is the number that determines speed, hence "effective"
```

Because the tables are lookup-only, they do not need to sit in fast accelerator memory. Offload them, keep only the ~2.3B of genuine compute weights on the accelerator, and you get a model that runs at 2B speed and 2B accelerator footprint while retaining the representational richness that 5.1B parameters bought during training. That is the entire trick, and it is why Google reports these running on phones and Raspberry Pi class hardware.

The lineage is worth noting: PLE came from the earlier Gemma 3n family, which paired it with **MatFormer** (a Matryoshka Transformer, nesting a fully functional smaller model inside a larger one by making the small FFN weight matrices literal sub-matrices of the large ones). Gemma 4's model card describes PLE for the E-series and does not claim MatFormer, so treat nesting as a Gemma 3n property rather than assuming it carried forward.

### The Comparison That Resolves the Confusion

| | MoE (26B-A4B) | PLE (E2B / E4B) |
|---|---|---|
| Reduces | compute per token | accelerator memory footprint |
| Does not reduce | memory footprint | compute per token |
| Mechanism | route each token to a subset of experts | give each layer its own lookup-only embedding table |
| Sparsity is | conditional, depends on the token | structural, fixed regardless of token |
| Requires a router | yes | no |
| Model class | sparse | dense |
| Failure mode | expert collapse, imbalanced routing | offload bandwidth becomes the bottleneck |
| Right deployment | servers, workstations, plenty of VRAM | phones, edge, memory-constrained |
| The letter means | **A**ctive: what runs | **E**ffective: what counts toward speed |

The one-line version: MoE spends memory to save compute. PLE spends parameter count to save accelerator memory. They solve opposite problems, which is precisely why Google shipped both in the same family targeting different hardware.

For completeness, the rest of the Gemma 4 lineup: 12B Unified (dense, encoder-free, projecting raw image patches and audio waveforms straight into the embedding space rather than using dedicated modality encoders) and 31B dense. All models interleave local sliding-window attention with full global attention, always ending on a global layer, and the global layers use unified keys and values plus Proportional RoPE to cut long-context KV cache cost.

### Why MoE and Diffusion Are Unexpectedly Well Matched

This connects back to the speed argument in Part 6, and it is the most interesting thing in this section.

Recall that MoE's weakness in autoregressive decode is arithmetic intensity. To generate a single token at batch size 1, you route to 8 experts, load those experts' weights out of memory, and then use them to process exactly one token. The weight loads are almost entirely wasted. MoE makes the memory-bound problem *worse* per useful token, which is why MoE models are usually justified by throughput at high batch sizes rather than single-user latency.

Now put a 256-token diffusion canvas through the same block. Every denoising step routes 256 positions simultaneously. Each expert that gets loaded is used by many tokens instead of one, so the weight loads amortize properly and arithmetic intensity rises sharply.

```
MoE + autoregressive decode, batch 1
  load 8 experts' weights  →  process 1 token     →  terrible amortization

MoE + diffusion canvas, 256 positions
  load experts' weights    →  process ~256 tokens →  weights amortized
```

Diffusion supplies exactly the parallel token volume that MoE needs to be efficient, and MoE supplies the parameter capacity that keeps quality high without inflating the per-step FLOP cost that diffusion multiplies by K denoising steps. The pairing is complementary rather than coincidental. NVIDIA's [FP8 DiffusionGemma card](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) reports over 1,100 tokens per second on an H100 at low batch sizes, and "high speed specifically at low batch size" is the signature of having fixed an arithmetic-intensity problem.

The same reasoning explains the deployment envelope. A 26B MoE with 3.8B active quantizes into roughly 18GB of VRAM, which is a single consumer or workstation GPU serving one user at very low latency. That is the regime where AR decode leaves the most compute idle and where MoE's memory cost is affordable. Every piece of the design points at the same target.

---

## Key Takeaways

1. **"Autoregressive" means the model conditions on its own previous outputs**, factorizing the joint distribution by the chain rule along sequence position. This gives exact likelihoods, an imposed permanent ordering, and sequential depth equal to output length.
2. **Diffusion iterates over noise level, not position.** Each step revises the entire object. This is why it gets bidirectional context and self-correction, and why it needs a fixed canvas size.
3. **Text diffusion needs a discrete corruption operator.** Masked diffusion works but locks tokens once resolved. Uniform state diffusion, which uses random vocabulary tokens as noise, keeps every position revisable throughout.
4. **Autoregression works fine for images once you fix the ordering.** VAR's next-scale prediction beat Diffusion Transformers. Raster-scan order was the failure, not autoregression.
5. **The two paradigms are provably related.** Scale-wise VAR is equivalent to discrete diffusion under a Markovian mask. Treat them as a spectrum of iterative refinement, not a dichotomy.
6. **Hybrids match the iteration axis to the data's structure.** AR along genuinely causal axes (time, scale, block order), diffusion along non-causal ones (space, within-block position).
7. **Block autoregressive diffusion buys parallelism within a window, not global revisability.** Once a block is committed to the KV cache it is as frozen as any AR token.
8. **The speed claim is a single-user latency claim.** Diffusion spends roughly 32x more FLOPs to cut sequential depth about 8x. That is free only because autoregressive decode leaves compute idle waiting on memory. Under heavy batching the idle compute is gone and the advantage inverts.
9. **MoE decouples parameter count from per-token compute** by routing each token to a small subset of experts. It reduces compute, not memory, because every expert must stay resident. This makes it a server technique, not an edge technique.
10. **E2B and E4B are dense models, not MoE.** "E" is effective parameters from Per-Layer Embeddings, a lookup-only memory trick that can be offloaded off the accelerator. "A" in 26B-A4B is active parameters from expert routing. MoE spends memory to save compute; PLE spends parameter count to save accelerator memory. Opposite problems.
11. **MoE and diffusion are complementary.** MoE has poor arithmetic intensity in batch-1 autoregressive decode because experts get loaded to serve one token. A 256-position canvas amortizes those loads across many tokens, which is why DiffusionGemma is fast precisely at low batch size.

---

## Key Papers & Resources

- [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905) (Tian et al., 2024) — AR surpassing diffusion transformers on images
- [Scale-Wise VAR is Secretly Discrete Diffusion](https://arxiv.org/html/2509.22636v1) — equivalence proof between next-scale AR and discrete diffusion
- [Multi-scale Autoregressive Models are Laplacian, Discrete, and Latent Diffusion Models in Disguise](https://arxiv.org/html/2510.02826v1) — the same equivalence from the Laplacian pyramid view
- [Diffusion in Text Generation Explained](https://ai.google.dev/gemma/docs/diffusiongemma/explained) (Google, 2026) — masked vs uniform state diffusion, DiffusionGemma architecture
- [DiffusionGemma: The Developer Guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/) (Google, 2026) — block autoregressive denoising, throughput numbers, Sudoku case study
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/) (Google DeepMind) — the earlier experimental text diffusion model
- **Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)** (Austin et al., 2021) — foundational discrete diffusion framework
- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020) — continuous diffusion baseline (see also Chapter 3)

On Mixture of Experts and sparsity:

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) (Google, 2026) — authoritative source for E2B/E4B dense classification, PLE, and 26B-A4B MoE configuration
- [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) (Gemma Team, 2026)
- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** (Shazeer et al., 2017) — the paper that made MoE practical in deep learning
- **Switch Transformers** (Fedus et al., 2021) — simplified top-1 routing, scaling MoE to trillion parameters
- **MatFormer: Nested Transformer for Elastic Inference** (Devvrit et al., NeurIPS 2024) — the Matryoshka nesting used in Gemma 3n
- [Introducing Gemma 3n: developer guide](https://developers.googleblog.com/introducing-gemma-3n-developer-guide/) (Google, 2025) — origin of PLE and the effective-parameter framing

*Content from Google and NVIDIA model documentation was paraphrased for compliance with licensing restrictions.*

---

## What's Next

Chapter 3 introduced the generative families as distinct species. This chapter showed the boundary between two of them is thinner than it looks, and that the deciding factor in practice is hardware economics rather than paradigm purity. That theme, where the bottleneck sits determines the architecture, is worth carrying into anything you build.
