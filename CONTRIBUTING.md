# Contributing

Thanks for helping keep this the most current directory of fast LLMs and the infrastructure behind them!

## Scope: speed, not price

This list tracks **inference speed** — latency (TTFT) and throughput (tokens/sec) — across fast models, speed-focused providers, serving engines, optimization techniques, benchmarks, and accelerators. Cost-performance belongs in the sibling list [awesome-flash-llms](https://github.com/dakotac1994/awesome-flash-llms): an entry can appear in both lists only if it is genuinely notable on *both* axes.

## Adding an entry

1. **Check it fits:** something notable for LLM inference speed — a model that tops speed leaderboards, a provider/chip that serves models unusually fast, a serving engine, an optimization technique, a speed benchmark, or an inference accelerator. A PR must point at a primary source: the vendor's docs/pricing page, the project's repo, a benchmark leaderboard, or the paper.
2. **Add to the right section** of `README.md`:
   - Fast API models → models notable for served speed (tok/s or TTFT standouts)
   - Speed-focused inference providers → hosted/serverless platforms with a speed angle (the providers table)
   - Fast open-weights models → small/distilled/quantized models notable for speed
   - Inference engines & serving stacks → vLLM, SGLang, TensorRT-LLM, Ollama, llama.cpp and friends
   - Optimization techniques → speculative decoding, quantization, KV-cache tricks, disaggregated serving…
   - Speed benchmarks & leaderboards → Artificial Analysis and similar, with methodology notes
   - Hardware accelerators → LPU, wafer-scale, RDUs, inference GPUs/TPUs…
   - Archived / deprecated → retired models or shut-down providers with the date and successor
3. **One entry = one bullet** (or one table row for providers). Format:
   `- [Name](https://official-site-or-docs) — ` one-line description + the speed figure inline with its source and date.
   Every speed claim carries a confidence tag:
   - `✅ verified 2026-09-29 · vendor-reported` — you read the figure on the vendor's official page yourself, with the date you saw it.
   - `✅ verified 2026-09-29 · independent` — you read the figure on an independent benchmark (e.g. Artificial Analysis) yourself, with the measurement date.
   - `⚠️ unverified` — third-party, stale, or unconfirmed. **Never guess a speed figure.**
4. **Add the matching record** to `data/fast-llms.json` with these exact fields:

| field | type | values |
|---|---|---|
| `name` | string | model, provider, engine, technique, benchmark, or accelerator name |
| `vendor` | string | vendor / organization |
| `url` | string | official https:// URL (docs, repo, or benchmark page) |
| `description` | string | one sentence |
| `speed_claim` | string | e.g. `"≈3,000 tok/s"` or `"unverified"` |
| `speed_verified` | bool | `true` only if you verified the figure on an official or benchmark page |
| `speed_url` | string | the page carrying the speed figure, or `""` |
| `measured_date` | string | `YYYY-MM-DD` you saw the figure, or `"unknown"` |
| `claim_type` | string | `vendor-reported` / `independent` |
| `status` | string | `active` / `maintenance` / `archived` / `commercial` |
| `category` | string | `model` / `provider` / `engine` / `technique` / `benchmark` / `hardware` |
| `features` | string[] | 3–6 key capabilities |

5. **Status changes:** if a model is retired, a provider shuts down, or a headline speed figure changes materially, update its README entry *and* add a row (newest-first) to `docs/status-changes.md`.

## Style rules

- Link the **official site** (vendor docs/pricing page, project repo, benchmark leaderboard, or paper), never a blog post or reseller.
- Facts that can change (tok/s figures, TTFT, release versions) get "as of" context in prose or are omitted — link the source instead of hard-coding numbers that rot.
- Vendor claims stay labeled as vendor claims ("vendor-reported", "vendor claims").
- Speed figures in the README are stamped with their verification date; never guess a figure.
- Keep README descriptions to one entry per bullet; put depth in `docs/`.

## CI

Every PR runs:
- **Link check** (lychee) over all markdown files — no dead links.
- **JSON validation** — `data/fast-llms.json` must parse, every record must have the required fields, and `status`/`category`/`claim_type` must be from the allowed sets above. Verified speeds require an https `speed_url` and a real `measured_date`.

Run locally before pushing:

```bash
python3 -c "import json; d=json.load(open('data/fast-llms.json')); print(len(d), 'entries ok')"
```
