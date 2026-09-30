# Optimization guide

The techniques that actually move TTFT and tokens/sec, roughly in the order to try them.

## 1. Serve on the right engine

The engine is the biggest lever for self-hosters: **vLLM** (PagedAttention + continuous batching) and **SGLang** (RadixAttention + disaggregated prefill/decode) are the 2026 baselines; **TensorRT-LLM** is NVIDIA's max-perf path on Hopper/Blackwell; **llama.cpp**/**Ollama** win on CPUs, Apple Silicon, and edge. Switching from naive Hugging Face `generate()` to vLLM is routinely a multi× throughput jump before any other tuning.

## 2. Quantize the weights

Decode is memory-bandwidth-bound: halving weight precision nearly doubles tok/s.

- **FP8** — the datacenter default on Hopper+; near-lossless on most models, served natively by vLLM/SGLang/TensorRT-LLM.
- **AWQ / GPTQ / Marlin** — 4-bit weight-only quantization for memory-constrained GPUs; Marlin kernels are the fast INT4 path in vLLM.
- **GGUF K-quants** — the llama.cpp ecosystem standard (2–8 bit + imatrix); the way small models run fast on laptops and edge.
- Rule of thumb: quantize until quality evals (not vibes) say stop.

## 3. Speculative decoding

A small draft model proposes several tokens; the big model verifies them in one parallel pass. Accepted tokens are free throughput — typical gains 1.5–3× on tok/s with **zero quality change**.

- **EAGLE / EAGLE-2 / EAGLE-3** — the current state of the art in draft heads; supported in SGLang and vLLM.
- **Medusa** — the earlier multi-head approach that popularized the idea.
- Works best when the draft model is well-matched (same family, distilled drafter) and on latency-sensitive single-stream workloads.

## 4. Shrink and reuse the KV-cache

Long contexts die by KV-cache: its memory footprint caps batch size, and batch size caps throughput.

- **Prefix / prompt caching** (explicit or RadixAttention-style automatic): reuse the system-prompt prefix across requests — cuts TTFT and prefill compute for agents.
- **KV-cache quantization**: store KV in INT8/FP8 — bigger batches, higher throughput, small quality cost.
- **Architectural**: MLA (DeepSeek) compresses KV heads; sliding-window attention bounds cache growth; eviction policies (H2O, StreamingLLM) keep infinite-ish contexts tractable.

## 5. Disaggregate prefill from decode

Prefill is compute-bound, decode is memory-bound — one hardware pool can't be optimal for both. **Disaggregated serving** (SGLang's specialty, also in vLLM Production Stack / llm-d) splits them: chunky compute for prefill, bandwidth-optimized pools for decode, with KV-cache transferred between. Reported multi× gains on mixed workloads.

## 6. Batch continuously

**Continuous (in-flight) batching** — admitting new requests the moment any slot frees — replaced static batching years ago, but it's still the first thing to verify: if your server waits for full batches, you're leaving throughput on the floor. PagedAttention made it memory-efficient; everything in this repo's engine list does it.

## 7. Pick a fast architecture

Some speed is baked into the model: **MoE** (few active params/token), **MLA**, small dense models, and distilled variants are fast *by construction*. When choosing between two models at similar quality, the one with fewer active parameters per token is the faster one.

## 8. Go to inference silicon (hosted)

When GPUs aren't enough: Groq's **LPU**, Cerebras' **wafer-scale engine**, and SambaNova's **RDU** trade generality for transformer throughput — 500–3,000 tok/s on supported models with sub-100 ms TTFT. The cost is model choice: you run what they support.

## Measurement discipline

Every optimization above interacts with batch size, sequence length, and quantization. Change one thing at a time, measure **p50/p99 TTFT and output tok/s on your own prompts**, and distrust any single headline number — including the ones in this repo. See [Understanding speed metrics](understanding-speed-metrics.md) for the comparison caveats.
