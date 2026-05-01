# sfv-thesis

Research repository for **"To Predict Is To Believe: Integrating Content, Context, and Creator Features For Pre-Publication Short-Form Video Engagement Prediction"** — a thesis building a Filipino micro-creator-focused engagement prediction model.

## What this repo is

The goal is to predict two engagement metrics for short-form videos *before* they are published:

- **ECR** (Engagement Continuation Rate) — probability of viewer retention past the 5-second hook
- **NAWP** (Normalized Average Watch Percentage) — duration-normalized total viewer retention

The model uses three frozen Large Multimodal Model (LMM) backbones as feature extractors, whose embeddings are combined with structured creator metadata (follower count, account age, posting timestamps):

| Backbone | Role |
|---|---|
| VideoLLaMA2 | Spatiotemporal + auditory understanding |
| Qwen2.5-VL | Visual-semantic reasoning |
| InternVideo2 | Visual dynamics analysis |

Benchmarks: Sun et al. 2025 (LMM-EVQA) and Guan et al. 2025.

## Current status: baseline reproduction

The active engineering effort is reproducing **LMM-EVQA (Sun et al. 2025)** as a working baseline before layering thesis-specific creator/context features on top. See [`plan.md`](plan.md) for the full step-by-step.

Architecture decision: `pipeline/` calls the upstream LMM-EVQA scripts via `subprocess` and never imports their modules directly. VideoLLaMA2 and Qwen2.5-VL have incompatible `transformers` pins, so each lives in its own conda env; `pipeline/` stays light with only `pandas`, `scipy`, `pyyaml`, `tqdm`.

## Repository layout

```
sfv-thesis/
├── thesis/           # LaTeX source (chapters 1–4, abstract, title page)
├── sources/          # references.bib, rrl_matrix.csv, source PDFs
├── outputs/          # generated artifacts (scripts, revised sections)
├── pipeline/         # our uv-managed Python project — all local code lives here
│   ├── configs/      # local.yaml / cloud_a100.yaml (hardware swap point)
│   ├── run_videollama2.py
│   ├── run_qwen.py
│   ├── ensemble.py
│   └── eval.py
├── lmm-evqa/         # vendored upstream (read-only — never edit)
├── data/             # gitignored — SnapUGC subsets
├── checkpoints/      # gitignored — fine-tuned model weights
├── data_specs/       # dataset specifications
├── sensitive_data/   # participant data (treat as sensitive)
├── plan.md           # active baseline reproduction plan
├── GEMINI.md         # canonical terminology, architecture, constraints
└── CLAUDE.md         # AI assistant instructions
```

## Getting started (pipeline)

```bash
cd pipeline
uv sync
```

All heavy ML dependencies (torch, transformers, videollama2, qwen_vl) live in upstream's per-model conda envs, not in the uv env. Hardware configuration lives only in `pipeline/configs/*.yaml` — switching from local to cloud is a config edit, not a code change.

## Dataset

Filipino micro-creator dataset (30–50 participants) — voluntary MP4 donations + TikTok analytics exports. Dataset building: May–Aug 2026. Baseline development uses public SnapUGC subsets (SnapUGC-tiny → SnapUGC-mini).
