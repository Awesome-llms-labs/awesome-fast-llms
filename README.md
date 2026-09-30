# Awesome Fast LLMs [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated directory of **fast LLMs** — models, providers, engines, techniques, benchmarks, and hardware notable for **inference speed**: low latency (TTFT) and high throughput (tokens/sec).

Speed is not price. This list tracks how fast models *run* — who tops the tok/s leaderboards, which chips and providers serve them fastest, the engines and optimizations behind them, and the silicon they're built on. Every speed figure is stamped ✅ **verified** (read on the source page, with the date and whether it's vendor-reported or independent) or ⚠️ **unverified** (third-party, stale, or unconfirmed). **Speed figures are never guessed.** Machine-readable records live in [`data/fast-llms.json`](data/fast-llms.json) with a `speed_verified` boolean, `claim_type`, and `measured_date` per entry.

## How this differs from [awesome-flash-llms](https://github.com/dakotac1994/awesome-flash-llms)

The sibling list covers **cost-performance**: cheap, efficient "Flash-class" models and their $/1M-token pricing. This list covers **inference speed**: latency and throughput. A model can appear in both lists only if it is genuinely notable on *both* axes — e.g. DeepSeek V4.1 Flash (cheap *and* fast). If you want the cheapest model per task, go there; if you want the fastest model per second, you're here.

## 2026 Highlights

- **Diffusion LLMs hit the top of the speed charts**: Celeris-1 leads the Artificial Analysis leaderboard at ~1,500–1,600 tok/s; Inception's Mercury 2.5 does 1,107 tok/s on commodity NVIDIA GPUs (Sept 2026) — but pays a TTFT penalty.
- **Custom silicon tiering is now measurable on the same model** (GPT-OSS-120B, AA): Cerebras ~1,697 tok/s > Groq ~480 tok/s > GPU clouds ~120–270 tok/s.
- **Groq 3 LPX** (NVIDIA's SRAM decode chip): 3,431 tok/s on Gemma 4 31B at 100K context — the fastest recorded figure for that model (Aug 2026, NVIDIA-run test).
- **The "fast tier" industry pattern**: same weights, inference-only optimization — GLM-5.3-FlashX (5×), Kimi HighSpeed (6×), MiniMax HighSpeed (1.7×), OpenAI Ultrafast (14×), Anthropic Fast mode (2.5×).
- **NVIDIA NIM 2.0.12**: 1,997 tok/s on Nemotron 3 Ultra (4× B200) — 2.5× via speculative decoding + disaggregation on the same GPUs (Sept 2026).
- **New silicon**: Etched's Sohu (transformer-only ASIC, 500K tok/s claimed, racks shipped summer 2026), Positron's Asimov ($875M raised Sept 2026), Furiosa's RNGD (3,200+ tok/s per chip).

## Speed confidence

- ✅ **verified** — the figure was read on the linked source page, with the date seen and the claim type:
  - *vendor-reported* — the vendor's/project's own page, docs, press release, or paper.
  - *independent* — a third-party measurement (Artificial Analysis, OpenBenchmarks, MLPerf, SemiAnalysis).
- ⚠️ **unverified** — third-party, stale, or unconfirmed; included for completeness, not as a citable figure.

**Figures are never guessed.** tok/s is not comparable across models, context lengths, quantization, or batch sizes — always read the figure with its conditions. See [Understanding speed metrics](docs/understanding-speed-metrics.md).

## Contents

- [Fast API models](#fast-api-models)
  - [Custom-silicon speed records](#custom-silicon-speed-records)
  - [Diffusion models](#diffusion-models)
  - [Vendor fast tiers](#vendor-fast-tiers)
- [Speed-focused inference providers](#speed-focused-inference-providers)
- [Fast open-weights models](#fast-open-weights-models)
  - [Small dense models](#small-dense-models)
  - [Sparse MoE (few active params)](#sparse-moe-few-active-params)
  - [Hybrid / linear-attention models](#hybrid--linear-attention-models)
- [Inference engines & serving stacks](#inference-engines--serving-stacks)
- [Optimization techniques](#optimization-techniques)
  - [Parallel & speculative decoding](#parallel--speculative-decoding)
  - [Attention & KV-cache](#attention--kv-cache)
  - [Quantization](#quantization)
  - [Serving & batching](#serving--batching)
- [Speed benchmarks & leaderboards](#speed-benchmarks--leaderboards)
- [Hardware accelerators](#hardware-accelerators)
  - [GPUs](#gpus)
  - [Custom inference silicon](#custom-inference-silicon)
  - [Emerging ASICs & fabrics](#emerging-asics--fabrics)
- [Methodology caveats](#methodology-caveats)
- [Guides](#guides)
- [Related repositories](#related-repositories)
- [Contributing](#contributing)
- [License](#license)

---

## Fast API models

### Custom-silicon speed records

Same-model, cross-provider figures come from Artificial Analysis (independent) unless noted.

- [GPT-OSS-120B on Cerebras](https://www.cerebras.ai/press-release/gimlet-labs-adds-cerebras-to-deliver-ultrafast-ai-inference-through-gimlet-cloud-deployment) — ✅ 1,697 tok/s, TTFT 0.49s (independent, 2026-08-30). Fastest publicly measured 120B-class serving; vendor docs claim ~3,000 tok/s (vendor-reported, Sept 2026).
- [Gemma 4 31B on Groq 3 LPX](https://www.techtimes.com/articles/325425/20260825/groq-3-lpx-hits-full-production-sram-decode-chip-reaches-3400-tokens-per-second.htm) — ✅ 3,431 tok/s at 100K context (independent, 2026-08-24, NVIDIA-run test). Fastest recorded figure for the model; ~4× the next result in that test.
- [Gemma 4 31B on Cerebras](https://www.cerebras.ai/press-release/gimlet-labs-adds-cerebras-to-deliver-ultrafast-ai-inference-through-gimlet-cloud-deployment) — ✅ 1,351 tok/s, TTFT 0.53s (independent, 2026-08-30). What wafer-scale silicon does to a 31B checkpoint.
- [GPT-OSS-120B on Groq](https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed) — ✅ ~473–493 tok/s (independent, Aug 2026). Deterministic LPU dataflow; Cerebras beats it 3–8× on this model.
- [GPT-OSS-20B on Groq](https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed) — ✅ 957 tok/s, TTFT 0.82s (independent, 2026-08-30). The LPU sweet spot: ~1K tok/s with sub-second TTFT.
- [Kimi K2.6 on Cerebras](https://www.cerebras.ai/press-release/gimlet-labs-adds-cerebras-to-deliver-ultrafast-ai-inference-through-gimlet-cloud-deployment) — ✅ 981 tok/s (independent, ~May 2026). 1T-param MoE (32B active) at 6.7× the next-best GPU cloud; a 10K-in/500-out coding task finished in 5.6s vs 163.7s on Moonshot's own endpoint.
- [DeepSeek-R1 671B on SambaNova](https://sambanova.ai/press/fastest-deepseek-r1-671b-with-highest-efficiency) — ✅ 198 tok/s/user on 16 SN40L RDUs (vendor-reported, 2025-02-13). A model that does 30–80 tok/s on typical GPU serving; AA independently measured >195 tok/s.
- [Nemotron 3.5 Lightning on Fireworks](https://www.marktechpost.com/2026/08/30/lowest-latency-inference-apis-for-voice-and-realtime-agents-a-time-to-first-token-ttft-first-benchmark/) — ✅ 501 tok/s, TTFT 0.46s (independent, 2026-08-30). NVIDIA's distilled 30B/3B-active Mamba-2 hybrid on GPU cloud.
- [Nemotron 3 Ultra on DeepInfra](https://www.marktechpost.com/2026/08/30/lowest-latency-inference-apis-for-voice-and-realtime-agents-a-time-to-first-token-ttft-first-benchmark/) — ✅ 371 tok/s, TTFT 0.28s (independent, 2026-08-30). Sub-300ms first token on a 550B MoE.
- [GPT-OSS-120B on Baseten](https://www.marktechpost.com/2026/08/30/lowest-latency-inference-apis-for-voice-and-realtime-agents-a-time-to-first-token-ttft-first-benchmark/) — ✅ TTFT 0.23s at 266 tok/s (independent, 2026-08-30). The TTFT champion on the AA provider board — GPU serving winning on first-token while losing on throughput.

### Diffusion models

Diffusion LLMs refine tokens in parallel instead of autoregressively: massive tok/s, but a first-chunk TTFT penalty. Rank them on TTFT for voice, tok/s for bulk.

- [Celeris-1](https://celeris.ai/) — ✅ ~1,490–1,612 tok/s, TTFT 0.62s (independent, AA, Sept 2026). AA's fastest model on the leaderboard; vendor homepage claims up to 2,038 tok/s (vendor-reported).
- [Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5) — ✅ 1,107 tok/s on commodity NVIDIA GPUs (vendor-reported, 2026-09-08); ~677 tok/s on AA (independent). Fastest production API model on standard GPUs; sub-170ms TTFT claimed for the Voice preview.
- [Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5) — ✅ 770 tok/s, TTFT 3.07s (independent, 2026-08-30). The diffusion "throughput trap" poster child: huge tok/s, slowest TTFT in the set.

### Vendor fast tiers

The industry pattern: same weights, inference-layer-only optimization sold as a speed tier.

- [OpenAI Ultrafast (GPT-5.6 Sol)](https://openai.com/index/previewing-ultrafast/) — ✅ up to 750 tok/s, 14× Standard processing speed (vendor-reported, 2026-08-13). A speed *tier* on Cerebras hardware, not a model; limited preview.
- [OpenAI Codex-Spark](https://runtimewire.com/article/openai-ultra-fast-eight-times-speed-devday-2026) — ✅ 1,000+ tok/s (vendor-reported, Feb 2026). Cerebras-served coding model where latency is the feature.
- [GPT-6 Luna](https://softreviewed.com/openai-gpt-6-sol-and-luna-review/) — ✅ ~175 tok/s, TTFT 0.71s (independent, Sept 2026). OpenAI's fast tier (1.05M context); "Fast" processing tier = 2× price for ~2.5× speed.
- [Gemini 3.5 Flash-Lite](https://github.com/yinjialu/ai-frontier-daily/blob/HEAD/data/firsthand/deepmind-blog/07-introducing-gemini-36-flash-35-flash-lite-and-35-flash-cyber.md) — ✅ 350 tok/s (vendor-reported, DeepMind blog 2026-07-21); ~382 tok/s on AA (independent). Google's fastest 3.5-class model, explicitly the speed tier.
- [Gemini 2.5 Flash-Lite](https://github.com/tonehq/tone/blob/HEAD/docs/MODEL_BENCHMARKS.md) — ✅ 393–887 tok/s (independent, ~Sept 2026). The older Flash-Lite still benchmarks among the fastest first-party throughputs.
- [GLM-5.3-FlashX](https://bbx.com/article/553435) — ✅ 200 tok/s, 5× GLM-5.3-Flash at 2.5× price (vendor-reported, 2026-09-18). Infra optimized by a GLM-5.3-driven "Infra Agent" in under two weeks.
- [Kimi K2.7-Code HighSpeed](https://www.techtimes.com/articles/318414/20260615/kimi-k27-code-adds-highspeed-mode-skips-independent-benchmark-submission.htm) — ✅ ~180–260 tok/s, ~6× standard (vendor-reported, 2026-06-15). No independent benchmark submitted — the claim rests on Moonshot's numbers.
- [Kimi K2 Turbo](https://platform.moonshot.ai/docs/guide/agent-support) — ✅ stable 60 tok/s, max 100 tok/s (vendor-reported, Moonshot docs). Persistent high-speed API track.
- [DeepSeek V4.1 Flash](https://codersera.com/blog/deepseek-v4-1-flash-complete-guide-2026/) — ✅ 219 tok/s, TTFT ~1s (independent, AA, Sept 2026). 552B MoE ~3× faster than its own V4-Pro.
- [DeepSeek V4 Flash](https://github.com/arpitbbhayani/the-daily-diff/blob/HEAD/src/content/stories/2026-08-15/12-hn-49310366-deepseek-v4-flash-llm-api-specifications-and-pricing.md) — ✅ 278 tok/s full precision, no quantization (vendor-reported, 2026-08-15). 1M context.
- [Grok 4-fast](https://github.com/lifejiggy/awesome-grok-skills/blob/HEAD/grok-models/grok-4-fast.md) — ✅ ~108–150 tok/s, TTFT P50 200ms / P99 800ms (independent third-party docs, 2026). xAI's dedicated fast tier, non-reasoning, 2M context.
- [Command A+](https://cohere.com/blog/command-a-plus) — ✅ 281 tok/s on AA at launch (independent, May 2026); 375 tok/s vendor claim with W4A4 (vendor-reported). 218B sparse MoE running on 2× H100.
- [North Mini Code](https://cohere.com/blog/north-mini-code) — ✅ 210 tok/s, TTFT 0.25s (independent, AA, ~June 2026). Single-H100 open coding model with the lowest TTFT in its class (Apache 2.0).
- [MiniMax M2.1 HighSpeed](https://aimlapi.com/blog/minimax-highspeed-models-m2-7-vs-m2-1-the-low-latency-ai-guide) — ✅ ~100–120 tok/s vs ~70 standard (independent third-party analysis, 2026). ~1.7× via faster MoE routing/batching.
- [Mistral Small 3.1](https://docs.ai.it.ufl.edu/docs/navigator_models/models/mistralai-mistral-small-3.1-instruct/) — ✅ 150 tok/s (vendor-reported, 2025-03-17). 24B dense on a single RTX 4090 or 32GB Mac.
- [Ministral 3 3B](https://www.marktechpost.com/2025/12/02/nvidia-and-mistral-ai-bring-10x-faster-inference-for-the-mistral-3-family-on-gb200-nvl72-gpu-systems/) — ✅ 385 tok/s on RTX 5090 (vendor-reported, Dec 2025). 273 tok/s at concurrency 8 on Jetson Thor.
- [Claude Haiku 4.5](https://pricepertoken.com/pricing-page/model/anthropic-claude-haiku-4-5) — ✅ 78–91 tok/s, TTFT 0.5s (independent, Sept 2026). Not a tok/s leader — included as the canonical "pay 2× for 2.5× speed" Fast-mode vendor tier.
- [Qwen3.8-Flash-Next](https://artificialanalysis.ai/models/qwen3-8-flash-next) — ✅ 58.1 tok/s, TTFT 2.5s (independent, AA, 2026-08-26). Cautionary "Flash": efficient architecture, below-median API speed on a reasoning model; vendor GB300 cluster claims (16,000+ tok/s/GPU) don't transfer to the API.

---

## Speed-focused inference providers

Hosted/serverless platforms with a genuine speed angle. Figures pair each with model + conditions + date + source type.

| Platform | Type | Speed angle | Representative speed | Confidence |
|---|---|---|---|---|
| [Cerebras](https://www.cerebras.ai/press-release/gimlet-labs-adds-cerebras-to-deliver-ultrafast-ai-inference-through-gimlet-cloud-deployment) | Wafer-scale chips | GPT-OSS 120B 1,697 tok/s (AA) / ~3,000 vendor; Kimi K2.6 981 tok/s | ✅ independent |
| [Groq](https://groq.com) | LPU chips | gpt-oss-20b 957 tok/s (AA); Groq 3 LPX 3,431 tok/s Gemma 4 31B (NVIDIA-run) | ✅ independent |
| [SambaNova](https://cloud.sambanova.ai) | RDU chips | R1 671B 198 tok/s/user (vendor); gpt-oss-120b 706 tok/s (AA) | ✅ vendor-reported |
| [Celeris](https://celeris.ai/) | Diffusion LM | 2,157.9 tok/s, TTFT 0.64s (AA) | ✅ independent |
| [Inception](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5) | Diffusion LM | Mercury 2.5 1,107 tok/s (vendor) / 677 (AA) | ✅ vendor-reported |
| [Fireworks AI](https://fireworks.ai) | GPU cloud | Nemotron 3.5 Lightning 501 tok/s, TTFT 0.46s (AA) | ✅ independent |
| [Together AI](https://www.together.ai) | GPU cloud | Turbo: 500 tok/s DeepSeek-V3.1 B200 (vendor); Kimi K2.7 245 tok/s (AA); 151.6 tok/s median, 393ms TTFA (OpenBenchmarks) | ✅ independent |
| [Baseten](https://www.baseten.co/) | Inference infra | TTFT 0.23s gpt-oss-120b (AA); 240.5 tok/s median (OpenBenchmarks) | ✅ independent |
| [Nebius Token Factory](https://tokenfactory.nebius.com) | GPU cloud | 269.3 tok/s median, 14.83% failures (OpenBenchmarks 2026-09-03) | ✅ independent |
| [DeepInfra](https://deepinfra.com) | GPU cloud | Nemotron 3 Ultra 371 tok/s, TTFT 0.28s (AA); 69.4 tok/s median (OpenBenchmarks) | ✅ independent |
| [Modal](https://modal.com/) | Serverless GPUs | 222.2 tok/s median, 470ms TTFA, 1.8s A100 cold start (OpenBenchmarks) | ✅ independent |
| [Telnyx](https://telnyx.com/resources/glm-5-3-latency-benchmarks-provider) | Telco + GPUs | 184.3 tok/s median, 573.5ms TTFA (OpenBenchmarks) | ✅ independent |
| [Novita AI](https://novita.ai) | Serverless API | 60.6 tok/s median, 1.4s TTFA (OpenBenchmarks) | ✅ independent |
| [Fireworks AI](https://fireworks.ai) | GPU cloud | Nemotron 3.5 Lightning 501 tok/s, TTFT 0.46s (AA); 58.2 tok/s median (OpenBenchmarks) | ✅ independent |
| [Z.AI](https://docs.z.ai) | Model platform | GLM-Z1-AirX Ultra-Fast 200 tok/s (vendor); 56.3 tok/s median (OpenBenchmarks) | ✅ independent |
| [Parasail](https://parasail.io/) | Heterogeneous cloud | 52.4 tok/s median (OpenBenchmarks); d-Matrix decode (10× claim, thin baseline) | ✅ independent |
| [NVIDIA NIM](https://www.nvidia.com/en-in/ai-data-science/products/nim-microservices/) | Hosted endpoints | 1,997 tok/s Nemotron 3 Ultra on 4× B200, 2.5× via NIM stack (vendor, 2026-09-10) | ✅ vendor-reported |
| [AWS Bedrock](https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-bedrock-priority-flex-inference-service-tiers/) | Managed API | Priority tier: 25% better output tok/s latency vs Standard (vendor, 2025-11) | ✅ vendor-reported |
| [Cloudflare Workers AI](https://www.cloudflare.com/products/workers-ai/) | Edge GPUs | 2–4× speedup on Llama 3.3 70B FP8 Fast (vendor, 2025-04-11) | ✅ vendor-reported |
| [TileRT-powered endpoints](https://newsletter.semianalysis.com/p/ultra-high-interactivity-on-nvidia) | Inference engine | 340–500 tok/s/user on 8× B200, ~3× conventional engines (SemiAnalysis, 2026-08-10) | ✅ independent |
| [Gimlet Labs](https://www.globenewswire.com/news-release/2026/09/28/3369931/0/en/gimlet-labs-adds-cerebras-to-deliver-ultrafast-ai-inference-through-gimlet-cloud.html) | Wafer-scale + GPU | 3,000 tok/s target at production scale — announced 2026-09-28, not yet GA | ⚠️ unverified (planned) |
| [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index) | Gateway | `:fastest` routing policy auto-selects the fastest partner provider | ⚠️ unverified (no aggregate figure) |
| [OpenRouter](https://openrouter.ai) | Gateway | Per-provider throughput/TTFT stats from production traffic; no published aggregate speed figure | ⚠️ unverified |
| [Alibaba Model Studio](https://help.aliyun.com/zh/model-studio/model-list) | Model platform | Speed-first "Flash" tier; no hosted tok/s figure verified | ⚠️ unverified |
| [Mistral La Plateforme](https://docs.mistral.ai) | Lab API | Fast-model lineup; no current first-party speed-tier claim located | ⚠️ unverified |
| [xAI API](https://docs.x.ai) | Lab API | "Fast" model variants + priority service; no published speed SLA | ⚠️ unverified |
| [Chutes](https://chutes.ai/) | Decentralized | Speed-based miner competition; needs an endpoint benchmark before strong inclusion | ⚠️ unverified |
| [Hyperbolic](https://hyperbolic.xyz/) | GPU marketplace | LLoCO 7.62× 128K-token claim (vendor, baseline unverified) | ⚠️ unverified |

All inference platforms above offer OpenAI-compatible endpoints unless noted (Bedrock uses its Converse API; Celeris compatibility unverified).

---

## Fast open-weights models

Small, sparse, or architecturally efficient models — fast *by construction*. Most carry no portable tok/s figure; their speed rationale is active-params-per-token, and the figure (when it exists) depends on the serving stack.

### Small dense models

- [Llama 3.2 1B / 3B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) — ⚠️ unverified. Tiny dense models; strong edge/on-device ecosystem (llama.cpp, Ollama, MLX). Llama 3.2 Community License.
- [Qwen3 0.6B / 1.7B / 4B](https://huggingface.co/Qwen/Qwen3-0.6B) — ⚠️ unverified. Small dense with hybrid thinking/non-thinking modes. Apache-2.0.
- [Gemma 3n E2B / E4B](https://huggingface.co/google/gemma-3n-E2B-it) — ⚠️ unverified. MatFormer nested submodels for on-device speed/memory trade-offs. Gemma Terms of Use.
- [Phi-4 Mini Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) — ⚠️ unverified. 3.8B dense; vendor reports 260 tok/s via ONNX INT4 vs 44 tok/s PyTorch FP16 on RTX 4090 (undated). MIT.
- [Ministral 3 3B / 8B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) — ⚠️ unverified. Edge dense models (385 tok/s on RTX 5090 per vendor — see API models). **MRL-0.1: research-only; commercial use needs a separate license.**
- [DeepSeek-R1-Distill-Qwen 1.5B / 7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) — ⚠️ unverified. Small dense reasoning-distilled models for fast local reasoning. MIT.
- [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) — ⚠️ unverified. 21B total / 3.6B active MoE; MXFP4 fits 16GB VRAM; 957 tok/s on Groq LPU (see API models). Apache-2.0.
- [Granite 4.0 Nano 350M / 1B](https://huggingface.co/ibm-granite/granite-4.0-h-1b) — ⚠️ unverified. Sub-billion dense models for extreme edge/local speed. Apache-2.0.
- [SmolLM2 135M / 360M / 1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) — ⚠️ unverified. Ultra-small dense; the floor of the speed/quality trade-off. Apache-2.0.

### Sparse MoE (few active params)

- [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) — ⚠️ unverified. 30B total / ~3B active per token. Apache-2.0.
- [Mistral Small 4 119B](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) — ⚠️ unverified. 119B total / 6.5B active (128 experts / 4 active); NVFP4 checkpoint; 256K context. Apache-2.0.
- [Command A+ (open weights)](https://huggingface.co/CohereLabs) — ✅ 110% throughput increase, 30% latency decrease vs Command A Reasoning (vendor-reported, 2026-05-20). 218B total / 25B active; W4A4 → 1× B200 or 2× H100. Apache-2.0.
- [Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) — ⚠️ unverified. ~198B total / ~11B active; hybrid sliding/global attention; MTP-3; vendor claims up to 400 tok/s (hardware/config unspecified — not portable). Apache-2.0.
- [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) — ⚠️ unverified. 309B total / 15B active (256 routed / 8 active); 1M context; 5-layer MTP speculative decoder. MIT. Independent test (2026-09-23, DGX Spark GB10): DFlash speculative decoding gave 1.58× (coding) / 1.70× (reasoning) / up to 2.08× at 250K context.

### Hybrid / linear-attention models

- [Kimi Linear 48B-A3B](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) — ✅ 2.8× decode throughput improvement (vendor-reported, paper [arXiv 2510.23556](https://arxiv.org/abs/2510.23556), Oct 2025; workload/hardware context incomplete — directional). 48B total / 3B active; Kimi Delta Attention + MLA. MIT.
- [Granite 4.0 H Tiny](https://huggingface.co/ibm-granite/granite-4.0-h-tiny) — ⚠️ unverified. Hybrid Mamba-2 + Transformer MoE: 7B total / 1B active; 128K context. Apache-2.0.
- [Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) — ⚠️ unverified. 30B total / 3B active Mamba-2 hybrid; built-in MTP; vendor claims up to 4× faster output. OpenMDW-1.1 (not Apache). Independent: ~61–75 tok/s decode, Q6_K llama.cpp on RTX 3090.
- [Nemotron 3 Nano Omni 30B-A3B](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) — ⚠️ unverified. Omni text+image+video+audio; 262K context; NVIDIA Nemotron Open Model License. Independent: 287 tok/s FP8 + thinking on vLLM / RTX PRO 6000 (~Aug 2026).

---

## Inference engines & serving stacks

The software that makes LLM serving fast. Speed mechanisms noted per engine; project-reported figures are labeled as such.

- [vLLM](https://github.com/vllm-project/vllm) — ✅ up to 24× throughput vs HF Transformers on tested workloads (SOSP 2023 paper, academic). PagedAttention + continuous batching; prefix caching, speculative decoding, GPTQ/AWQ/FP8/Marlin. The de-facto baseline. Apache-2.0.
- [SGLang](https://github.com/sgl-project/sglang) — ✅ up to 5× throughput vs Guidance/vLLM on agent/reasoning/chat workloads (project-reported, Jan 2024). RadixAttention prefix caching; disaggregated prefill (3.8× prefill / 4.8× decode gains); EAGLE speculative decoding. Apache-2.0.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — ✅ 8× higher performance on H100 vs A100 (vendor-reported, Oct 2023). Compiled NVIDIA kernels, in-flight batching, paged KV cache, FP8/INT4/FP4, CUDA graphs; independently measured up to 70% faster than llama.cpp on RTX 4090/3090 (~Sept 2026). Apache-2.0.
- [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) — ⚠️ **Archived upstream** (maintenance mode since Dec 2025). Still a proven baseline (continuous batching, tensor parallelism, Flash Attention); prefer vLLM/SGLang for new deployments. Apache-2.0.
- [Ollama](https://github.com/ollama/ollama) — ⚠️ unverified. Dead-simple local runtime: `ollama run` pulls, quantizes, and serves hundreds of pre-quantized models behind a local OpenAI-compatible API. MIT.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — ⚠️ unverified (it is the reference baseline others measure against). Pure C/C++ inference + the GGUF standard; K-quants 2–8-bit; CUDA/Metal/ROCm/Vulkan. Best speed story on non-NVIDIA hardware. MIT. (Repo moved from `ggerganov/llama.cpp` to `ggml-org/llama.cpp` in 2025.)
- [LMDeploy](https://github.com/InternLM/lmdeploy) — ⚠️ unverified (project claims up to 1.8× vLLM request throughput; benchmark date/hardware not recovered — not portable). Persistent/continuous batching, blocked KV cache, W8A8/W4A16 + KV-cache quantization. Apache-2.0.
- [Aphrodite Engine](https://github.com/aphrodite-engine/aphrodite-engine) — ⚠️ unverified. vLLM-derived serving; broad quantization (EXL2/GGUF/GPTQ/AWQ/FP8/Marlin), paged KV cache, advanced samplers. **AGPL-3.0.**
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) — ⚠️ unverified. TVM/TensorIR compilation: graph/kernel fusion, memory planning, target-specific CUDA/Metal/ROCm/Vulkan/OpenCL/WebGPU execution. Apache-2.0.
- [ExLlamaV2](https://github.com/turboderp-org/exllamav2) — ⚠️ unverified. EXL2 mixed-bit 2–8 bpw quantization, custom CUDA kernels, paged attention, dynamic batching. MIT.
- [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) — ✅ 13.2 tok/s quantized on RTX 4090; 8.00×/11.69× vs llama.cpp (SOSP 2024 paper, academic). Activation-locality-aware hot/cold neuron placement (GPU hot, CPU cold). MIT.
- [DeepSpeed-FastGen / MII](https://github.com/deepspeedai/DeepSpeed) — ✅ up to 2.3× higher effective throughput vs vLLM (project-reported, Jan 2024). Dynamic SplitFuse, continuous batching, non-contiguous KV caches. Apache-2.0. ([DeepSpeed-MII](https://github.com/deepspeedai/DeepSpeed-MII))
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) — ✅ 3× TTFT and 2× avg-latency improvement via KV-aware routing (project-reported, design docs; measured on 100K R1 queries, R1-Distill-Llama-70B FP8, 2× H100). Prefill/decode disaggregation, NIXL KV transfers, tiered KV cache. Apache-2.0.
- [llm-d](https://github.com/llm-d/llm-d) — ✅ 2.2k output tok/s/GPU on H200, 2.9k on B200 for DeepSeek expert-parallel serving (project-reported, v0.3, Oct 2025); v0.4 cut per-output-token latency 40% for DeepSeek V3.1 on H200. Kubernetes distributed inference, intelligent routing, prefix-cache offload. Apache-2.0.
- [vLLM Production Stack](https://github.com/vllm-project/production-stack) — ⚠️ unverified. Kubernetes service discovery, routing, observability, fault tolerance, model aliases for vLLM. Apache-2.0.
- [Triton Inference Server](https://github.com/triton-inference-server/server) — ⚠️ unverified (do not attribute TensorRT-LLM figures to Triton — gains come from its backends). Dynamic batching, model ensembles; LLM serving via vLLM/TensorRT-LLM backends. BSD-3-Clause.
- [AIBrix](https://github.com/vllm-project/aibrix) — ⚠️ unverified. LLM-aware routing (KV occupancy/queue depth/prefix hits), inference-shaped autoscaling, distributed KV reuse, P/D disaggregation. Apache-2.0.
- [KTransformers](https://github.com/kvcache-ai/ktransformers) — ✅ 4.62–19.74× prefill, 1.25–4.09× decode (SOSP 2025 paper, academic; hardware/workload details surfaced secondarily — confirm from paper before porting). CPU/GPU heterogeneous MoE execution with CPU expert offload. Apache-2.0.
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) — ✅ up to 2.1× speedup with DFlash speculative decoding on Qwen3-4B INT4 (project-reported, 2026 PR; hardware not in surfaced excerpt — caveat). For Intel CPU/GPU/NPU. Apache-2.0.
- [Ray Serve LLM](https://github.com/ray-project/ray) — ⚠️ unverified. vLLM-backed serving with actor autoscaling, tensor/pipeline parallel deployment, multi-model orchestration. Apache-2.0.

---

## Optimization techniques

The methods behind fast inference. Paper-reported speedups are labeled academic (independent) or vendor; they are workload-specific — directional, not portable.

### Parallel & speculative decoding

- [Speculative decoding](https://arxiv.org/abs/2211.17192) — ✅ ~2–3× on T5-XXL (academic, 2022). Small draft model proposes tokens; the target verifies in parallel. Output distribution identical to autoregressive decoding — zero quality change.
- [EAGLE / EAGLE-2 / EAGLE-3](https://github.com/SafeAILab/EAGLE) — ✅ 1.27–3.07× / 3.05–4.26× / up to 6.5× (academic, 2024–2025). Feature-level autoregressive drafters; current state of the art, supported in SGLang and vLLM. ([papers](https://arxiv.org/abs/2503.01840))
- [Medusa / Hydra / Hydra++](https://github.com/FasterDecoding/Medusa) — ✅ Medusa-2 up to ~3.6×; Hydra++ 2.70× vs autoregressive (academic, 2024). Extra prediction heads generate a candidate token tree verified in parallel.
- [REST](https://github.com/FasterDecoding/REST) — ✅ 1.62–2.36× on single-batch 7B/13B (academic, NAACL 2024). Retrieval-based speculation: no neural draft model, retrieves n-grams from a datastore.
- [Lookahead Decoding](https://github.com/hao-ai-lab/LookaheadDecoding) — ✅ ~1.6–2.1× on Vicuna-7B/13B (academic, 2024). Drafter-free: Jacobi iteration over future tokens plus n-gram pool.
- [Multi-token prediction (MTP)](https://arxiv.org/abs/2412.19437v2) — ✅ 85–90% second-token acceptance → 1.8× tokens/s (vendor — DeepSeek, 2024). Auxiliary modules jointly trained with the target; doubles as an integrated speculative drafter (DeepSeek-V3, Nemotron 3.5 Lightning, MiMo-V2.6).

### Attention & KV-cache

- [PagedAttention](https://arxiv.org/abs/2309.06180) — ✅ 2–4× throughput over FasterTransformer/Orca at comparable latency (academic, SOSP 2023). Virtual-memory-style KV-cache management; the enabler of efficient continuous batching.
- [RadixAttention / automatic prefix caching](https://arxiv.org/html/2312.07104v2) — ✅ 6.4× throughput, 3.7× lower latency (academic/project, 2024). Radix-tree prefix cache shared across requests; cuts TTFT for agentic loops.
- [Prompt caching (API-level)](https://claude.com/blog/prompt-caching) — ✅ up to 85% lower latency on long prompts; 100K-token book 11.5s→2.4s (vendor — Anthropic, 2024). Persist processed static prefixes across calls.
- [FlashAttention v1/v2/v3](https://github.com/Dao-AILab/flash-attention) — ✅ FA3 1.5–2.0× over FA2 on H100 FP16, ~1.2 PFLOP/s FP8 (academic, 2024). Exact tiled attention that never materializes the N×N matrix in HBM. Table stakes.
- [Grouped-Query Attention (GQA)](https://arxiv.org/abs/2305.13245) — ✅ 0.28s/sample vs 1.51s MHA at near-MHA quality on T5-XXL (academic, 2023). Standard in Llama 3, Qwen, Mistral families.
- [Multi-Head Latent Attention (MLA)](https://arxiv.org/pdf/2405.04434) — ✅ 93.3% smaller KV cache; 5.76× generation throughput *combined with MoE* — do not attribute the full figure to MLA alone (vendor — DeepSeek, 2024).
- [Sliding-window attention / rolling KV cache](https://ar5iv.labs.arxiv.org/html/2310.06825) — ✅ 2× attention speed at 16K; 8× less cache at 32K with W=4096 (vendor paper — Mistral, 2023). Bounds KV-cache growth.
- [KV-cache quantization (KIVI, KVQuant)](https://github.com/jy-yuan/KIVI) — ✅ KIVI: 2.35–3.47× throughput, 4× larger batches (academic, ICML 2024). Low-bit KV caches; KVQuant reaches 1M context on a single A100. ([KVQuant](https://github.com/SqueezeAILab/KVQuant))
- [KV-cache eviction (H2O, StreamingLLM)](https://github.com/mit-han-lab/streaming-llm) — ✅ StreamingLLM: 22.2× per-token speed over sliding-window recomputation (academic, ICLR 2024). Fixed-memory generation via heavy-hitter / attention-sink retention. (H2O's "29× throughput" is unverified from a primary source.)

### Quantization

- [GPTQ](https://github.com/IST-DASLab/gptq) — ✅ ~3.25× on A100 / ~4.5× on A6000 vs FP16 end-to-end (academic, 2022). Layer-wise one-shot PTQ; 175B models on one GPU. ([paper](https://arxiv.org/abs/2210.17323))
- [AWQ](https://github.com/mit-han-lab/llm-awq) — ✅ TinyChat runtime >3× over HF FP16 on desktop/mobile GPUs (academic, 2023). Activation-aware weight quantization, no backprop. ([paper](https://arxiv.org/abs/2306.00978))
- [Marlin](https://github.com/IST-DASLab/marlin) — ✅ near-ideal ~4× vs FP16 at batch sizes up to 16–32 (repo authors). Fused INT4×FP16 GEMM kernel; the fast INT4 path in vLLM.
- [GGUF K-quants (llama.cpp)](https://github.com/ggml-org/llama.cpp) — ⚠️ unverified. Blockwise K-quant families (Q4_K_M etc.) enabling 7B–70B inference at tens of tok/s on consumer/edge hardware; no universal per-token multiplier from primary sources.
- [SmoothQuant](https://github.com/mit-han-lab/smoothquant) — ✅ 1.56× speedup, 2× memory reduction, negligible accuracy loss (academic, 2022/2024). Migrates activation outliers into weights for INT8×INT8.
- [SqueezeLLM](https://arxiv.org/pdf/2306.07629.pdf) — ✅ 2.4× lower latency vs FP16 on A6000 (academic, 2023). Sensitivity-based non-uniform 3/4-bit quantization with sparse outlier storage.

### Serving & batching

- [Continuous / in-flight batching](https://www.anyscale.com/blog/continuous-batching-llm-inference) — ✅ up to 23× over naive static batching (vendor — Anyscale, 2023, measured with vLLM). Admits requests at decode-step granularity.
- [Disaggregated prefill/decode (Mooncake)](https://github.com/kvcache-ai/Mooncake) — ✅ 525% throughput in simulations; 75% more production requests for Kimi (vendor — Moonshot, 2024; peer-reviewed FAST 2025). Split compute-bound prefill from bandwidth-bound decode. ([paper](https://arxiv.org/pdf/2407.00079))
- [MoE sparsity (compute-side)](https://arxiv.org/pdf/2405.04434) — ⚠️ unverified as a standalone multiplier (reported figures are entangled with MLA/quantization). Fewer active params/token = higher tok/s at a given quality bar.

---

## Speed benchmarks & leaderboards

- [Artificial Analysis — Models leaderboard](https://artificialanalysis.ai/leaderboards/models) — standardized model output speed (tok/s) alongside quality and cost. Live/continuous; the de-facto independent speed reference — nearly every independent figure in this repo traces to it.
- [Artificial Analysis — API providers analysis](https://artificialanalysis.ai) — per-provider TTFT and output tok/s, including same-model comparisons across endpoints. Live/continuous.
- [OpenRouter performance statistics](https://openrouter.ai/openai/gpt-6-sol-20260922) — per-provider throughput, TTFT/latency, and uptime from production traffic. Rolling statistics.
- [OpenBenchmarks Inference](https://openbenchmarks.com/inference) — paired provider requests: E2E p50/p99, TTFT p50, p50 tok/s, 95%-floor speed, task success, failures. Snapshot cadence (2026-09-03 run: 600 requests/provider). Open methodology. ([repo](https://github.com/openbenchmarks-labs/inference))
- [vLLM benchmark suite](https://github.com/vllm-project/vllm) — `bench serve/throughput/latency`: request throughput, tok/s, TTFT, TPOT, inter-token latency p99. A user-run harness, not a leaderboard.
- [GuideLLM](https://github.com/vllm-project/guidellm) — load sweeps on OpenAI-compatible endpoints: TTFT/ITL/tok/s and SLO/capacity analysis. Active OSS project; a tool, not a leaderboard.
- [MLPerf Inference](https://mlcommons.org) — standardized closed/open datacenter & edge submissions; LLM metrics in tok/s under latency/accuracy constraints. Versioned releases (v5.0: Apr 2025, 17,457 results from 23 organizations). Independent consortium.
- [ML.ENERGY Leaderboard](https://ml.energy/leaderboard/) — throughput (tok/s), TPOT, energy/request, batch sizes. Community/academic. ([data](https://github.com/ml-energy/leaderboard))
- [HF Optimum LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard) — ⚠️ stale (appears inactive since ~Dec 2024). Latency, throughput, energy, memory across hardware/model configs.
- [LLMPerf leaderboard](https://github.com/matanyaloewenthal/llmperf-leaderboard) — ⚠️ archived/historical. Output throughput and TTFT under reproducible settings. ([harness](https://github.com/ray-project/llmperf))

---

## Hardware accelerators

### GPUs

- [H100 / H200 (Hopper)](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/) — ✅ 33,072 server tok/s on Llama 2 70B with 8× H200 (independent, MLPerf v5.0, Apr 2025). The industry inference baseline; FP8 Transformer Engine.
- [B200 / GB200 NVL72 (Blackwell)](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/) — ✅ 98,443 server tok/s on Llama 2 70B with 8× B200: 3.0× over H200 (independent, MLPerf v5.0, Apr 2025). FP4 support, NVL72 rack-scale; TensorRT-LLM claims Llama 4 at >40,000 tok/s on B200 (vendor-reported).
- [Instinct MI300X / MI325X](https://moreh.io/technical-report/21k-output-tokens-per-second-deepseek-inference-on-amd-instinct-mi300x-gpus-with-expert-parallelism-251113/) — ✅ DeepSeek-R1 decode >21,000 tok/s on 8× MI300X (vendor-adjacent — Moreh, AMD software partner, 2026). 192GB HBM3 / 256GB HBM3e, 5.3 TB/s; independent paper (2026): Kimi-K2.5 (1T-param MoE) at 7,327 tok/s on 4× MI325X.
- [Gaudi 3](https://www.techradar.com/pro/intel-piles-pressure-on-nvidia-with-launch-of-new-ai-accelerator-that-is-faster-and-cheaper-than-the-h100-but-will-it-be-enough-to-keep-up-with-the-stunningly-fast-h200) — ✅ ~2× H100 throughput on LLaMA 2 70B at large batch (vendor-reported, Intel, 2024 — take with salt). 128GB HBM2e (3.7 TB/s), FP8, vLLM backend.

### Custom inference silicon

- [Groq LPU](https://pondero.ai/news/2026-08-26-nvidia-groq-3-lpx/) — ✅ 1,660+ tok/s on Llama 3 70B with speculative decoding (vendor-reported). Deterministic SRAM-only dataflow (no HBM), compiler-scheduled. (NVIDIA licensed Groq's chip technology for ~$20B, Dec 2025 — reported, single secondary source.)
- [Groq 3 LPX](https://www.techtimes.com/articles/325425/20260825/groq-3-lpx-hits-full-production-sram-decode-chip-reaches-3400-tokens-per-second.htm) — ✅ 3,400 tok/s on Gemma 4 31B at 100K context (independent, AA, Aug 2026). Vera Rubin-platform decode accelerator: 256 LP30 LPUs/rack, ~500MB SRAM/chip; prefill on Rubin GPUs, decode on LPUs.
- [WSE-3 / CS-3](https://www.theregister.com/on-prem/2024/08/27/cerebras-gives-waferscale-chips-an-inferencing-twist/549643?td=readmore) — ✅ 2,522 tok/s Llama 4 Maverick per user (vendor-reported, vs NVIDIA DGX B200's 1,038 — differing conditions, directional); 969 tok/s Llama 3.1 405B (independent, AA). Whole-wafer die: 4T transistors, 44GB on-chip SRAM, 21 PB/s — no HBM trips.
- [SN40L RDU](https://sambanova.ai/press/fastest-deepseek-r1-671b-with-highest-efficiency) — ✅ 198 tok/s/user on full DeepSeek-R1 671B, 16 chips (vendor-reported, Feb 2025). Reconfigurable dataflow with 3-tier memory; full 16-bit precision at speed.
- [TPU v6 Trillium](https://www.nextplatform.com/ai/2025/09/17/google-shows-off-its-inference-scale-and-prowess/1642358) — ✅ ~800 tok/s on Llama 2 70B (MLPerf via The Next Platform, Sept 2025 — secondary source, directional). Inference-tuned TPU generation.
- [TPU v7 Ironwood](https://www.ai-market-watch.com/news/google-tpu-runs-moonshot-ais-kimi-57-faster-than-nvidia-gpus-using-deepseeks-inf-yx0ofd) — ✅ Kimi K3: 709 tok/s on 16 TPU v7 vs 452 on 16 GB200, same vLLM engine (Sept 2026, published by Inferact — vLLM founding-team startup; not independently reproduced). 192GB HBM/chip, 64MB software-managed VMEM per TensorCore.
- [Inferentia2 / Trainium2](https://aws.amazon.com/blogs/machine-learning/faster-llms-with-speculative-decoding-and-aws-inferentia2/) — ✅ Llama-3-70B median per-token latency 21.4ms (~47 tok/s) (vendor-reported, AWS blog). Cloud-native inference silicon via Neuron SDK with speculative decoding; Trainium2 is the actively developed line.

### Emerging ASICs & fabrics

- [Corsair (d-Matrix)](https://www.techradar.com/pro/microsoft-backed-a-tiny-hardware-startup-that-just-launched-its-first-ai-processor-that-does-inference-without-gpu-or-expensive-hbm-memory-and-a-key-nvidia-partner-is-collaborating-with-it?rand=1339) — ✅ 60,000 tok/s on Llama 3 8B per server; 30,000 tok/s on Llama 3 70B per rack (vendor-reported, 2025). Digital in-memory compute: logic in SRAM bit cells, no HBM. Deployed via Parasail for decode.
- [speedAI240 Slim (Untether AI)](https://www.eetimes.com/amd-and-untether-take-on-nvidia-in-mlperf-benchmarks/) — ✅ ~3× the queries/sec/watt of 8× NVIDIA H200 (independent, MLPerf ResNet-50, 2024 — pre-LLM-era workload). At-memory compute, 1400+ RISC-V cores, 75W PCIe card.
- [ACF SuperNIC / EMFASYS (Enfabrica)](https://blog.enfabrica.net/enfabrica-unveils-industrys-first-ethernet-based-ai-memory-fabric-system-for-efficient-8078bd89fdcb?gi=bb7dc18d4956) — ✅ up to 50% lower cost per token (vendor-reported, 2025–2026). 3.2 Tbps Ethernet fabric + CXL memory pooling offloads KV-cache/prefill memory from GPU HBM.
- [Sohu (Etched)](https://www.techtimes.com/articles/319393/20260630/transformer-chip-startup-etched-exits-stealth-800m-raised-1b-contracts.htm) — ✅ 500,000 tok/s on Llama 70B claimed; first racks shipped summer 2026 (vendor-reported, June 2026 — independent verification pending). Transformer-only ASIC, attention wired into silicon (TSMC N4P).
- [Asimov (Positron AI)](https://www.techtimes.com/articles/327400/20260912/positron-ai-raises-875m-prove-commodity-memory-can-beat-hbm-inference.htm) — ⚠️ unverified (26× tokens/$ vs GB300 NVL72 is simulation-only). Inference ASIC on commodity LPDDR5X instead of HBM; Atlas (FPGA gen) deployed at OCI (vendor-reported). $875M raised Sept 2026.
- [RNGD (FuriosaAI)](https://furiosa.ai/rngd) — ✅ 3,200–3,300 tok/s on Llama 3.1 8B FP8 per chip — 60 users @ 40 tok/s per server (vendor-reported, 2026). Tensor Contraction Processor, TSMC 5nm, 48GB HBM3, 180W; LG AI Research production perf/watt validation.
- [Napier (Tensordyne)](https://primanews.org/inside-the-inference-hardware-revolution-of-2026/) — ✅ up to 1,300 tok/s per user at <1/10th the power of comparable NVIDIA hardware (vendor-reported, 2026, via secondary report — directional). Logarithmic number system rack-scale hardware.

---

## Methodology caveats

Read before comparing any two figures in this list.

- **tok/s is not portable.** Every figure depends on hardware generation, batch size/concurrency, prompt/output lengths, quantization format, sampler settings, and backend version. A vendor's 3,000 tok/s (short prompts, high batch, best quantization) is an upper bound, not your workload.
- **Vendor tok/s ≠ your workload.** Single-stream chat latency (TTFT) and bulk throughput are different numbers; headline tok/s is usually measured at high batch sizes.
- **Speed ≠ TTFT.** Diffusion models (Mercury 2: 770 tok/s, 3.07s TTFT) and reasoning models dominate tok/s while losing on first-token latency. Voice agents should rank on TTFT (Baseten 0.23s, Cohere North Mini 0.25s), not tok/s.
- **Chunk-granularity artifacts inflate tok/s** on ultra-fast endpoints — one community benchmark showed a bogus 3,730 tok/s for Cerebras that was socket-chunking noise, not model speed.
- **tok/s on reasoning models often includes thinking tokens** — billed and time-consuming, but not answer tokens.
- **"Fast tier" claims need independent confirmation.** Kimi's HighSpeed (no independent submission) and similar vendor-only figures are labeled as such.
- **Aggregate throughput ≠ per-user speed.** Platform-wide tokens/day, per-server tok/s, and per-user tok/s are different metrics; this list labels which is which.
- **Figures rot fast.** Every entry carries a verification date; treat undated figures as stale.

---

## Guides

- [Choosing a fast model](docs/choosing-a-fast-model.md) — matching models to latency budgets: TTFT-first vs throughput-first workloads.
- [Understanding speed metrics](docs/understanding-speed-metrics.md) — TTFT vs TPOT vs throughput, and the batching caveat.
- [Optimization guide](docs/optimization-guide.md) — the techniques that move the numbers, in the order to try them.
- [Glossary](docs/glossary.md) — PagedAttention, speculative decoding, KV-cache, LPU, disaggregated prefill, and more.
- [Status changes](docs/status-changes.md) — retirements and material speed-figure changes, newest first.
- [Machine-readable catalog](data/fast-llms.json) — all 148 entries with speed-verification status, claim type, and measured date.

## Related repositories

- [awesome-flash-llms](https://github.com/dakotac1994/awesome-flash-llms) — sibling list: cost-performance Flash-class LLMs and their pricing. Speed vs price, side by side.
- [awesome-ai-sandboxes](https://github.com/dakotac1994/awesome-ai-sandboxes) — awesome-list of AI sandboxes: managed, OSS, browser, and adjacent.
- [awesome-ai-agents](https://github.com/dakotac1994/awesome-ai-agents) — awesome-list of AI agent frameworks and ecosystems.

## Contributing

Entries and corrections are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Every PR is checked by CI: lychee link check over all markdown files, and JSON-schema validation of `data/fast-llms.json` (including the `speed_verified` boolean, `claim_type`, and required `speed_url` + `measured_date` for verified speeds).

## License

[MIT](LICENSE) © 2026 dakotac1994
