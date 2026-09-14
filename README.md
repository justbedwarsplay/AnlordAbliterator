# AnlordAbliterator

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-11.8%20%2F%2012.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Heretic-1.4%2B-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-AGPL--3.0-00A86B?style=for-the-badge" />
</p>

<p align="center">
  <strong>Automated LLM abliteration with <a href="https://github.com/p-e-w/heretic">Heretic</a>. Honest before/after evaluation. Beautiful reports.</strong>
</p>

<p align="center">
  Download with progress → Baseline benchmarks → Heretic optimization → Merge LoRA → Re-evaluate → Compare → HTML / JSON / CSV
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#output">Output</a> •
  <a href="#gguf-quantization">GGUF</a> •
  <a href="#publishing">Publishing</a>
</p>

---

## Features

| | What it does |
|---|---|
| **🔧 Heretic-native** | Calls `heretic.main:main` as a subprocess. Optuna search over refusal direction & layer weights, Pareto selection (`refusals ↓` + `KL ↓`), merges LoRA into base model |
| **⬇️ Visible downloads** | `huggingface_hub.snapshot_download` with progress bars into `cache/hub` |
| **🔐 Ephemeral HF auth** | Optional `HF_TOKEN` kept only in the current process (`questionary` masked prompt). `Enter` = anonymous. `--no-hf-token-prompt` for CI |
| **📊 Native runner (default)** | Custom evaluators on `datasets` + `transformers` — no `lm-eval` required, works offline, tolerant to `trust_remote_code` deprecation |
| **🧪 Extended suite** | **29 tasks / 17 families** — 6 classic + GPQA, MMLU-Pro, Ifstruct, ParseBench, ExtractBench, ScreenSpot-Pro, MMMU-Pro. All selectable via `--tasks` or checkbox |
| **⚖️ Before → After** | Same benchmarks run on baseline and abliterated models, `comparison.json` with deltas |
| **💻 Hardware-aware** | `psutil` + `pynvml` monitoring. `HardwarePlanner` auto-selects load-time `quantization` / `device_map` / `max_memory` / `batch_size` for your GPU (see note below) |
| **📝 Reports** | Interactive `report.html` + `report.json` + `report.csv` + per-benchmark `*.json` |
| **🔄 Resume** | `--resume` (on by default) — skips completed Heretic + benchmark shards by `model_id + config` hash. Corrupted `*.tmp` files are ignored |
| **🪟 Windows-first** | `questionary` + `rich` prompts, no `curses`, handles `triton-windows` / `causal-conv1d` gracefully |

> **Note on `--quantization`:** this flag in AnlordAbliterator controls only **how the model is loaded into memory** during abliteration/evaluation (VRAM saving via `bitsandbytes` + CPU offload). It does **not** quantize the saved abliterated model — the output in `models/heretic_output/` is always full-precision `safetensors` (BF16/F16). For distribution quants (GGUF `Q4_K_M` etc.) see [GGUF Quantization](#gguf-quantization).

## Architecture

```
Hugging Face model (any causal LM)
        │
        ▼ snapshot_download → cache/hub
   HardwarePlanner ──▶ dtype / device_map / max_memory / batch_size
        │
        ├─► Baseline benchmarks (optional) ──► results/baseline/*.json
        │
        ▼ Heretic ── trials × (reset → abliterate → KL + refusal count)
        │              └─ Pareto front → best trial → merge LoRA
        │                              └─► models/heretic_output/abliterated-{ns}/
        │
        ├─► Abliterated benchmarks ──► results/abliterated/*.json
        │
        ▼ Comparison ──► results/comparison.json
        │
        ▼ Reports ──► reports/report.{html,json,csv}
```

Works with any Heretic-supported architecture (Qwen, Llama, Mistral, Gemma, etc.).

## Requirements

- Python 3.10+
- PyTorch 2.2+ with CUDA 11.8/12.x recommended (CPU also works)
- 16GB+ RAM; 8GB VRAM is enough for 0.8B–7B models (larger models auto-use load-time `bnb_4bit` + CPU offload)
- `transformers>=4.46`, `heretic-llm>=1.4`, `datasets>=2.18`, `accelerate`, `bitsandbytes`, `peft`, `optuna`

See `pyproject.toml` / `requirements.txt` for the full list.

## Installation

**Windows (PowerShell):**
```powershell
python -m venv anlord_env
anlord_env\Scripts\activate
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .
# optional: lm-eval harness for exact leaderboard reproduction
pip install -e .[lmeval]
```

**Linux / WSL / Colab:**
```bash
sudo apt update && sudo apt install -y python3.10-venv build-essential git
python3 -m venv anlord_env && source anlord_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

Verify:
```bash
python -m src.anlord --info
pytest tests/ -v
```

## Quick Start

### Interactive (recommended on Windows)

```bash
python -m src.anlord
```

You will be asked for:
1. **HF token** (masked, `Enter` to skip) — needed for gated datasets like `Idavidrein/gpqa`
2. **Model ID** (e.g. `Qwen/Qwen2.5-7B`, `meta-llama/Llama-3.2-3B`, or a local path)
3. **Output dir** (where `results/`, `reports/`, `models/`, `cache/` live)
4. **Tasks** — checkbox (`Space` to toggle, `Enter` to confirm)
5. **Mode** — `quick` (100 samples) or `full`

### CLI

```bash
# Classic 6 + 8 extended (14 tasks) — recommended
python -m src.anlord --model Qwen/Qwen2.5-7B --output ./output --tasks all
# all = extended = mmlu,gsm8k,hellaswag,arc_challenge,winogrande,truthfulqa,
#                  gpqa,gpqa_diamond,mmlu_pro,ifstruct,parsebench,extractbench,screenspot_pro,mmmu_pro

# Only the new benchmarks
python -m src.anlord --model Qwen/Qwen2.5-7B --output ./output \
  --tasks gpqa_diamond,mmlu_pro,ifstruct,parsebench,extractbench,screenspot_pro,mmmu_pro

# Everything (29 tasks including mmlu subsets)
python -m src.anlord --model Qwen/Qwen2.5-7B --output ./output --tasks full

# Quick smoke test
python -m src.anlord --model HuggingFaceTB/SmolLM2-135M --output ./tmp --mode quick --limit 20 --tasks standard

# Gated datasets need a token
# PowerShell:  $env:HF_TOKEN="hf_xxx"
# bash:        export HF_TOKEN=hf_xxx
# or paste it when prompted (https://huggingface.co/settings/tokens → Read)
```

## Benchmarks

Choose via `--tasks` or the interactive checkbox:

| Flag | Expands to |
|---|---|
| `all` / `*` | `extended` — 14 tasks (6 classic + 8 new) |
| `full` | all 29 tasks |
| `standard` | 6 classic only |
| `parsing` / `vision` / `gpqa` / `academic` … | group |
| `mmlu,gsm8k,gpqa_diamond` | explicit comma list |

### Task table (Native runner)

| ID | Display | Dataset | Split | Few-shot | Metric |
|---|---|---|---|---|---|
| `mmlu` | MMLU | `cais/mmlu` | test (14k) | 5 | acc |
| `gsm8k` | GSM8K | `openai/gsm8k` | test | 5 | exact_match |
| `hellaswag` | HellaSwag | `Rowan/hellaswag` | validation | 10 | acc_norm |
| `arc_challenge` | ARC-Challenge | `allenai/ai2_arc` | test | 0 | acc_norm |
| `winogrande` | Winogrande | `allenai/winogrande` | validation | 0 | acc |
| `truthfulqa` | TruthfulQA MC2 | `truthful_qa` | validation | 0 | acc |
| `gpqa` / `gpqa_diamond` | GPQA Diamond | `Idavidrein/gpqa:gpqa_diamond` | train (198) | 0 | acc |
| `gpqa_main` | GPQA Main | `Idavidrein/gpqa:gpqa_main` | train | 0 | acc |
| `gpqa_extended` | GPQA Extended | `Idavidrein/gpqa:gpqa_extended` | train | 0 | acc |
| `mmlu_pro` | MMLU-Pro | `TIGER-Lab/MMLU-Pro` | test (12k) | 5 | acc |
| `ifstruct` | Ifstruct v1.0 | `LiquidAI/ifstruct-v1.0` | test (2k) | 0 | acc |
| `parsebench` | ParseBench | `llamaindex/ParseBench` | test | 0 | F1 |
| `extractbench` | ExtractBench | `llamaindex/ExtractBench` | test | 0 | F1 |
| `screenspot_pro` | ScreenSpot-Pro | `likaixin/ScreenSpot-Pro` | test | 0 | acc |
| `mmmu_pro` | MMMU-Pro | `MMMU/MMMU_Pro` | test | 0 | acc |
| `mmlu_*` (×3), `arc_easy`, `math`, `squad_v2`, `race`, `humaneval`, `mbpp`, `ifeval` | subsets / extras | various | — | 0–5 | acc / F1 / pass@1 |

> The native GPQA/MMLU-Pro scorers use log-prob over answer letters (` A`/` B`/…) with length normalization. On 198-sample Diamond this differs ±2% from the `lm-eval` harness — delta `baseline → abliterated` stays fair because both sides use the same scorer.

## CLI Options

```
Model:
  --model, -m            HF ID or local path
  --model-commit         pin a commit hash
  --output, -o           output root (contains results/reports/models/cache)
  --cache-dir            HF cache dir (default ./cache)
  --prefetch-model / --no-prefetch-model

Auth:
  --hf-token-prompt / --no-hf-token-prompt

Heretic:
  --trials, -t           trials (default 100)
  --timeout              seconds (default 43200 = 12h)
  --eval-prompts         prompts for KL/refusal (default 100)

Benchmarks:
  --tasks, --benchmarks, -b   see table above
  --mode                 quick | full
  --limit, -l            samples per benchmark (quick default 100)
  --num-fewshot          override few-shot

Hardware (load-time only, not saved):
  --dtype                auto | float16 | bfloat16 | float32
  --device               cuda | cpu | mps
  --device-map           auto | cuda | cpu
  --quantization         auto | none | bnb_4bit | bnb_8bit  (how to fit model in VRAM)
  --batch-size           (default 1)
  --max-memory           e.g. '{"0":"7GB","cpu":"11GB"}'

Pipeline:
  --skip-baseline --skip-abliteration --skip-benchmarks
  --resume / --no-resume
  --baseline-evaluate    force a separate Heretic baseline pass

Other:
  --info --verbose, -v --quiet, -q --log-file
```

## Output

```
output/
├── cache/hub/                          # HF snapshot (safetensors + tokenizer)
├── models/
│   ├── heretic_output/
│   │   └── abliterated-{nanoseconds}/  # ← your merged model (ready to use)
│   │       ├── config.json
│   │       ├── tokenizer.json
│   │       ├── model.safetensors (or shards)
│   │       └── heretic_abliteration_metrics.json
│   ├── heretic_study/                  # Optuna study (for resume)
│   └── heretic_baseline/
├── results/
│   ├── baseline/*.json
│   ├── abliterated/*.json
│   ├── comparison.json
│   └── ablation.json
├── reports/
│   ├── report.html
│   ├── report.json
│   └── report.csv
├── run_config.json
└── environment.json
```

The abliterated model is already merged (LoRA weights folded in). Load it directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("output/models/heretic_output/abliterated-123456", trust_remote_code=True, device_map="auto")
tok = AutoTokenizer.from_pretrained("output/models/heretic_output/abliterated-123456", trust_remote_code=True)
```

## GGUF Quantization

AnlordAbliterator saves the abliterated model as full-precision `safetensors`. To distribute smaller files for `llama.cpp` / Ollama / LM Studio, convert to GGUF after the run.

### Default (no quant)

The default output is `model.safetensors` in BF16/F16 — no quantization, full quality. Keep it for Hugging Face.

| GGUF type | Bits | Size vs F16 | Quality | Use |
|---|---|---|---|---|
| `f16` | 16 | 1.0× | lossless | base for re-quant, or direct GGUF use |
| `bf16` | 16 | 1.0× | lossless | same as f16 |
| `q8_0` | 8 | ~0.50× | near-lossless | best quant quality, good default |
| `q6_k` | 6 | ~0.39× | very high | K-quant |
| `q5_k_m` | 5 | ~0.33× | high | K-quant medium, sweet spot |
| `q5_k_s` | 5 | ~0.32× | high- | K-quant small |
| `q5_0` | 5 | ~0.33× | medium | legacy 5-bit |
| `q4_k_m` | 4 | ~0.26× | good | **smallest recommended** — ~3.8× smaller than F16 |
| `q4_k_s` | 4 | ~0.25× | okay | smaller, more loss |
| `q4_0` | 4 | ~0.26× | okay | legacy 4-bit |
| `q3_*` / `q2_*` | 3/2 | ~0.20× | low | not recommended, beyond Q4_K_M |

## Troubleshooting

- **`Dataset is gated` (GPQA)** → set `HF_TOKEN` (`$env:HF_TOKEN="hf_..."` / `export HF_TOKEN=...`) or paste it at the prompt.
- **CUDA OOM** → `--quantization bnb_4bit` or use a smaller base model; `HardwarePlanner` does this automatically with `auto` (load-time only).
- **KL > 0.5** → model is damaged — try fewer `--trials`, different `--eval-prompts`, or pick another Pareto trial from `results/ablation.json`.
- **Resume stuck on zeros** → `python -m src.anlord --no-resume` once to clear corrupted `*.tmp` shards.

## License

AGPL-3.0-or-later. Heretic and each base model have their own licenses — respect them when redistributing abliterated weights.

## Acknowledgments

- [Heretic](https://github.com/p-e-w/heretic) — abliteration
- [Hugging Face Transformers & Datasets](https://huggingface.co)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — optional

<p align="center"><em>AnlordAbliterator — make models honest, keep reports beautiful.</em></p>
