# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Maintenance

Keep this file up to date. Whenever you make a significant change to the repo — new directories, architectural decisions, renamed conventions, added tools, updated workflows — update the relevant section of this file before finishing the task.

## Project Overview

This is a thesis research repository for **"To Predict Is To Believe: Integrating Content, Context, and Creator Features For Pre-Publication Short-Form Video Engagement Prediction"** — a Filipino micro-creator-focused engagement prediction model using an ensemble of frozen Large Multimodal Models (LMMs).

**Core target metrics** (never substitute with raw views/likes/shares):
- **ECR** (Engagement Continuation Rate): probability of viewer retention past the 5-second hook
- **NAWP** (Normalized Average Watch Percentage): duration-normalized total viewer retention

## Repository Name

The repo lives at `~/Code/GitHub/sfv-thesis`

## Repository Structure

- `thesis/` — LaTeX source files for each chapter (`chapter_1.tex`–`chapter_4.tex`, `abstract-english.tex`, `title_page.tex`)
- `sources/` — Academic references (`references.bib`, `rrl_matrix.csv`) and source PDFs in `sources/papers/`
- `outputs/` — Generated artifacts (scripts, revised sections, model predictions)
- `data_specs/`, `sensitive_data/` — Dataset specifications and participant data (treat as sensitive)
- `lmm-evqa/` — **Vendored upstream** code from https://github.com/sunwei925/LMM-EVQA. Treated as read-only; never edit files inside this directory.
- `pipeline/` — Our `uv`-managed Python project. All local code lives here: adapters, configs, dataset loaders, ensemble, evaluation, and the `.venv`. This is the main project folder.
- `data/`, `checkpoints/` — Gitignored. Hold downloaded SnapUGC subsets and fine-tuned model weights respectively.
- `plan.md` — Active working plan for the LMM-EVQA baseline reproduction. Update this when scope changes.
- `GEMINI.md` — Primary project context document; canonical source for terminology, architecture, and constraints.

## Technical Model Architecture

Three frozen LMM backbones used as feature extractors (no fine-tuning):
- **VideoLLaMA2** — spatiotemporal + auditory understanding
- **Qwen2.5-VL** — visual-semantic reasoning
- **InternVideo2** — visual dynamics analysis

Extracted embeddings are concatenated with structured metadata: follower count, account age, posting timestamps.

**Keyframe extraction**: exactly 8 uniform keyframes from the initial video window.

**Benchmarks**: compare against Sun et al. (2025) and Guan et al. (2025).

## Working with LaTeX

Rules:
- Preserve all LaTeX commands unless explicitly asked to remove them
- Keep terminology consistent with `GEMINI.md` (ECR, NAWP, exact LMM names)
- Output revised LaTeX code only, no prose explanation

## Working with Scripts/Outputs

For presentation scripts, target ~130–150 words/minute and include `[M:SS – M:SS]` time markers. Save results to `outputs/`.

## Python Environment

The project lives in `pipeline/` and is managed with `uv`:

```bash
cd pipeline
uv sync
```

Add dependencies with `uv add <package>` from inside `pipeline/` (updates `pipeline/pyproject.toml` and `pipeline/uv.lock`).

API keys and credentials live in `pipeline/.env` — never commit that file (covered by `.gitignore`).

## Key Constraints (from GEMINI.md)

- Dataset participants: Filipino micro-creators, 30–50 participants
- Data source: voluntary MP4 donations + TikTok analytics exports
- Timeline: dataset building May–Aug 2026, model enhancement Aug–Dec 2026, evaluation Jan–Apr 2027
- **Frozen-backbone variant** is an *optional* future direction (originally framed as a hard constraint due to compute limits). The current baseline-reproduction phase uses LMM-EVQA's published fine-tuned checkpoints as-is; full retraining is deferred to cloud.

## Baseline Reproduction (LMM-EVQA)

The active engineering effort is reproducing **Sun et al. 2025 (LMM-EVQA)** on small SnapUGC subsets (SnapUGC-tiny → SnapUGC-mini) before extending it with thesis-specific creator/context features. Full plan: `plan.md`.

Working rules:
- **Never edit files inside `lmm-evqa/`** during the baseline phase. Treat it as a read-only vendored snapshot. All adapter/wrapper/ensemble/eval code lives in `pipeline/`. The invariant `git diff lmm-evqa/` should be empty until the extension phase. The single sanctioned future exception: a minimal documented patch to each upstream inference script that dumps the pre-regression hidden state to a `.pt` file (needed for the thesis's mid-fusion model). That patch must be flag-gated so the unmodified inference path still works, and noted in `lmm-evqa/UPSTREAM.md`.
- **Subprocess boundary, not in-process import.** `pipeline/` invokes upstream scripts via `subprocess.run([...])` (typically `conda run -n <env> python ...`) and exchanges data through files. `pipeline/` must not `import videollama2` / `import qwen_vl` / `import torch` — its uv env stays light (`pandas`, `scipy`, `pyyaml`, `tqdm`). All heavy ML deps live in upstream's per-model conda envs. Reason: upstream pins CUDA-11.8 `torch==2.2.0` + Python 3.9/3.10, and VideoLLaMA2 vs Qwen2.5-VL have incompatible `transformers` pins — sharing an env is impossible.
- **One conda env per upstream model.** E.g. `lmmevqa-videollama2` and `lmmevqa-qwen`. Their names live in `pipeline/configs/*.yaml`. Never merge them.
- **Hardware differences live only in `pipeline/configs/*.yaml`** (e.g., `local.yaml`, `cloud_a100.yaml`). No `if torch.cuda.is_available()` / device-detection branches scattered through `pipeline/` code. This keeps the local→cloud swap a config edit, not a code change.
- **LMM-EVQA is two independent models**, not a single fused pipeline. VideoLLaMA2 and Qwen2.5-VL are set up separately; their predictions are combined by our own `pipeline/ensemble.py` (simple averaging by default).
- **Local-first, cloud-ready**: develop and smoke-test on the local Windows PC (specs TBD; `pipeline/configs/local.yaml` is a placeholder until they land). Expect to move heavy inference and any fine-tuning to a cloud GPU (RunPod / Lambda / Vast.ai) unless the local box has A100/H100-class VRAM.
- **Evaluation metrics**: SROCC and PLCC (LMM-EVQA's challenge metrics) on the public SnapUGC subset. Thesis-target metrics (ECR, NAWP) come back into focus once the baseline is reproduced and we layer the Filipino creator features on top.
