# Glossary

Speed terms that recur across model cards, provider docs, and serving-engine READMEs.

- **TTFT (time to first token)** — latency from request to the first output token. The interactive-latency metric; dominated by prefill. See [Understanding speed metrics](understanding-speed-metrics.md).
- **TPOT (time per output token)** — average gap between generated tokens; 1/TPOT = output tokens/sec.
- **Throughput (tokens/sec)** — sustained generation rate. Always check *which* throughput: output, input (prefill), total, or per-stream.
- **Prefill / decode** — the two phases of inference: processing the prompt (compute-bound) vs generating tokens one by one (memory-bandwidth-bound). Optimizations target them separately.
- **KV-cache** — the stored key/value tensors that let decode skip recomputation. Its size grows with context length × layers; managing it is half of inference engineering.
- **PagedAttention** — vLLM's virtual-memory-style KV-cache management: non-contiguous blocks, near-zero fragmentation, the enabler of efficient continuous batching.
- **RadixAttention** — SGLang's radix-tree prefix cache: shares KV-cache across requests with common prefixes (system prompts, few-shot examples).
- **Continuous (in-flight) batching** — scheduling new requests into the batch the moment a slot frees, instead of waiting for the whole batch to finish. The single biggest throughput lever in modern serving.
- **Disaggregated prefill/decode** — running prefill and decode on separate hardware pools, each tuned for its phase. Lifts throughput at the cost of moving KV-cache between pools.
- **Speculative decoding** — a small draft model proposes tokens; the large model verifies them in parallel. Raises throughput with no quality change. Variants: Medusa, EAGLE, Hydra, REST.
- **Quantization** — shrinking weights (FP16 → INT8/INT4/FP8) to cut memory traffic and raise tok/s. Formats: GPTQ, AWQ, Marlin, FP8, GGUF K-quants.
- **LPU** — Groq's Language Processing Unit: deterministic, compiler-scheduled tensor streaming architecture built for low-latency transformer inference.
- **Wafer-scale engine** — Cerebras' approach: an entire silicon wafer as one chip, keeping model weights on-chip for extreme memory bandwidth and tok/s.
- **RDU** — SambaNova's Reconfigurable Dataflow Unit: dataflow architecture serving models at full 16-bit precision with high throughput.
- **Prefix caching** — reusing the KV-cache of a repeated prompt prefix across requests. Cuts TTFT and input cost for agentic loops with long system prompts.
- **FlashAttention** — IO-aware exact attention that tiles computation through SRAM, cutting the memory traffic of attention. v1/v2/v3 target successive GPU generations; table stakes in every modern engine.
- **MLA (multi-head latent attention)** — DeepSeek's compressed KV representation: shrinks the KV-cache dramatically, trading compute for memory — great for long-context throughput.
- **MoE (mixture of experts)** — only a subset of parameters ("experts") activate per token. Fewer *active* params per token → higher tok/s at a given quality bar (e.g. DeepSeek-V3/V4, Step 3.7 Flash, GPT-OSS).
- **OpenAI-compatible endpoint** — an API speaking the OpenAI chat-completions wire format, so clients switch providers by changing base URL + key. Nearly every speed-focused provider in this list offers one.
- **Single-stream vs batched** — the comparison caveat: headline tok/s is usually measured at high batch sizes; single interactive streams run slower. Compare like with like.
