# Understanding speed metrics

The three numbers that matter in LLM inference speed, what they measure, and how to compare them.

## TTFT — time to first token

How long from sending your prompt until the **first** output token arrives. This is *latency* — what a human feels in chat, and what an agent feels before it can start acting on a response.

- Dominated by **prefill**: the model must process your entire prompt before generating anything. Long prompts → higher TTFT.
- Inference-chip providers (Groq, Cerebras, SambaNova) optimize TTFT aggressively; 50–100 ms TTFT on 7B-class models is their headline territory.
- TTFT grows with prompt length and model size. A "90 ms TTFT" claim measured on a 100-token prompt says nothing about your 50K-token RAG prompt.

## TPOT — time per output token (inter-token latency)

The average gap between consecutive tokens during generation. This is the *streaming smoothness* metric — the inverse of it is output throughput.

- **tokens/sec (output)** = 1 / TPOT. A provider quoting "1,000 tok/s" is quoting TPOT = 1 ms.
- TPOT is dominated by **decode**: each token requires a full forward pass over the model with the KV-cache. Memory bandwidth is the bottleneck — this is why smaller models, quantization, and custom silicon push tok/s up.
- Reasoning/thinking models emit long internal token streams, so per-request *time* can be high even when TPOT is excellent.

## Throughput (tokens/sec) — and which throughput

"Throughput" is quoted loosely; pin down which one:

| Metric | Measures | Matters for |
|---|---|---|
| **Output tok/s** (1/TPOT) | generation speed per stream | chat, agents, interactive UX |
| **Input tok/s** (prefill rate) | prompt-processing speed | long-context ingestion, RAG |
| **Total tok/s** (input+output) | aggregate server throughput | capacity planning, cost per task |
| **Requests/sec** | concurrent request rate | bulk/API workloads |

Vendor leaderboards (e.g. Artificial Analysis) typically publish **output tokens/sec** and **TTFT** side by side — the two numbers you need for interactive use.

## The batching caveat — the big one

A provider's headline tok/s is usually measured at **high batch sizes** (many concurrent requests sharing the GPU). Your single-stream chat request sees lower throughput. When comparing:

1. **Same batch size** — single-stream vs single-stream, or saturated vs saturated.
2. **Same sequence lengths** — tok/s collapses as context grows.
3. **Same quantization** — FP8/INT4 numbers aren't comparable to FP16 numbers.
4. **Same model** — tok/s is meaningless across different parameter counts without normalizing.

Artificial Analysis publishes its methodology (median output tok/s across providers per model, with TTFT) — that's why it's the reference leaderboard in this repo.

## Latency vs throughput — pick your metric by workload

- **Interactive chat / agents:** minimize TTFT first, then TPOT. A 200 ms TTFT with 60 tok/s feels instant; a 2 s TTFT with 200 tok/s feels broken.
- **Bulk generation / evals / distillation:** maximize total tokens/sec (batch throughput). TTFT is irrelevant; sustained tok/s per dollar is the game.
- **Long-context RAG:** watch prefill (input tok/s) and TTFT on long prompts — this is where prefix caching and disaggregated prefill earn their keep.

## Related

- [Glossary](glossary.md) — PagedAttention, speculative decoding, KV-cache, LPU…
- [Choosing a fast model](choosing-a-fast-model.md) — matching models to latency budgets.
- [Optimization guide](optimization-guide.md) — the techniques that move these numbers.
