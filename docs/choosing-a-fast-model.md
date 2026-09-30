# Choosing a fast model

A decision framework for picking a model when **speed is the constraint**. (For cost-per-task math, see the sibling list [awesome-flash-llms](https://github.com/dakotac1994/awesome-flash-llms).)

## 1. Name your latency budget first

| Workload | Budget | What to optimize |
|---|---|---|
| Chat / copilot / voice | TTFT < 300 ms, TPOT < 20 ms | prefill speed, single-stream tok/s |
| Agents (multi-step) | TTFT < 1 s per step | prefix caching, fast small models |
| Bulk generation / evals | none per request | batched total tok/s |
| Long-context RAG | TTFT on 50K+ tokens | prefill throughput, RadixAttention-style caching |

If you can't state the budget, measure your current p50/p99 TTFT and TPOT first — then shop.

## 2. Pick the serving substrate

- **Need the absolute fastest single stream?** Inference-chip providers: Cerebras (wafer-scale) and Groq (LPU) top independent tok/s leaderboards on the models they support. Model selection is narrower than GPU clouds.
- **Need a specific open model fast?** GPU-cloud speed tiers: Fireworks, Together, DeepInfra, Nebius, Novita, Parasail — check Artificial Analysis per-model, per-provider tok/s before committing.
- **Need full control / zero per-token cost?** Self-host with vLLM or SGLang on rented GPUs; llama.cpp/Ollama for local. Your tok/s is a function of GPU memory bandwidth and quantization — see the [optimization guide](optimization-guide.md).
- **Need one key, many providers?** OpenRouter or Hugging Face Inference Providers with speed-based routing and failover.

## 3. Shrink the model before scaling the hardware

Speed follows *active* parameters per token more than total parameters:

- **Small dense models** (3B–14B) on fast chips routinely beat 70B models on raw tok/s at "good enough" quality for classification, extraction, and simple agents.
- **MoE models** (few active params per token) give large-model quality at small-model speed — the reason DeepSeek-V4, Step 3.7 Flash, and GPT-OSS appear on speed leaderboards.
- **Distilled / quantized builds** (FP8, AWQ, GGUF K-quants) trade a little quality for large tok/s gains, especially on memory-bandwidth-bound decode.

## 4. Compare on independent leaderboards, not vendor pages

- Use **Artificial Analysis** per-model pages: median output tok/s + TTFT across providers, measured independently.
- Vendor tok/s figures are real but measured on vendor-favorable setups (short prompts, high batch, best quantization). Treat them as upper bounds.
- Always compare **same model, same quantization, same batch regime**.

## 5. Mind the billed-token trap

"Thinking"/reasoning variants emit long internal token streams billed as output. A fast tok/s figure with heavy thinking can still mean slow *task completion* and high cost — measure end-to-end task latency, not just tok/s.

## Related

- [Understanding speed metrics](understanding-speed-metrics.md) — TTFT vs TPOT vs throughput.
- [Optimization guide](optimization-guide.md) — speculative decoding, quantization, KV-cache tricks.
- [Machine-readable catalog](../data/fast-llms.json) — every entry with speed-verification status.
