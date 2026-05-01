# Plan: Recreate LMM-EVQA as a Reproducible Baseline

## Context

The thesis (*"To Predict Is To Believe..."*) names two prior works as benchmarks: **Sun et al. 2025 (LMM-EVQA)** and **Guan et al. 2025**, and uses **Li et al. 2024 (SnapUGC ECR/NAWP)** as the source of the target metrics. Before we can extend either with creator/context features (per the thesis architecture), we need a working, reproducible recreation of LMM-EVQA running against a small SnapUGC subset on this machine. The goal of this plan is to stand up that baseline in a way that:

1. Faithfully reproduces LMM-EVQA's two-model setup (VideoLLaMA2 + Qwen2.5-VL).
2. Runs end-to-end *locally first* on a small subset (SnapUGC-tiny → SnapUGC-mini), with a clean modular boundary so the same code can run on a cloud GPU later with a config swap.
3. Leaves room to layer the thesis's creator/context features on top later, and to (optionally, future work) explore frozen-backbone variants — without needing to fork the upstream repo.

## Reality check (read first)

- **LMM-EVQA is two independent models**, not a fused pipeline. The repo ships `VideoLLaMA2-audio_visual/` and `Qwen2.5-VL/` as separate folders, each with its own `train.sh` / inference scripts. The challenge-winning "ensemble" is simple averaging of the two models' predicted scores — *not* in the upstream repo, we add it ourselves.
- **Architecture: subprocess orchestration, not in-process import.** `pipeline/` calls upstream scripts via `subprocess.run([...])`, parses their CSV/JSON output, and never imports `videollama2` / Qwen modules in-process. Reasons: (1) upstream pins CUDA-11.8 `torch==2.2.0` + `transformers==4.42.3` for Python 3.9/3.10 — pulling those into our `uv` env (currently `requires-python = ">=3.12"`) would poison every other dep we add later for the Filipino-creator-feature work; (2) VideoLLaMA2 and Qwen2.5-VL have mutually incompatible `transformers` pins, so each needs its own conda env; (3) "never edit `lmm-evqa/`" stays clean when the boundary is a process boundary. Cost: data crosses the boundary as files (CSV for scores, `.pt` for embeddings later), not Python objects.
- **Extension strategy (mid fusion).** The thesis architecture concatenates LMM embeddings with creator metadata (per `GEMINI.md`), so we eventually need the *pre-regression hidden state*, not just the scalar score. Plan: in the baseline phase, run upstream unmodified and capture only the score. In the extension phase, write a single-file documented patch to each upstream inference script that *also* dumps the pre-regression hidden state to a `.pt` file alongside the score. This is the only sanctioned exception to the "never edit `lmm-evqa/`" rule, and it must be a minimal, isolated diff.
- **Local hardware (Windows PC)**: specs TBD — fill in `pipeline/configs/local.yaml` once known. VideoLLaMA2-7B fp16 ≈ 14 GB; Qwen2.5-VL-7B fp16 ≈ 14 GB. Both inference paths assume CUDA. If the local GPU has insufficient VRAM, fall back to 4-bit quantization (bitsandbytes works on Windows+CUDA) or CPU offload.
- **Fine-tuning either model locally is not realistic** unless the Windows PC has an A100/H100-class GPU. LMM-EVQA ships *fine-tuned* checkpoints — use those for inference and defer any retraining to cloud.
- **Strategy**: download LMM-EVQA's published fine-tuned weights (Baidu Yun extraction codes `3aqc` and `98zr`) and run **inference only** locally. Fine-tuning is a future cloud step.

## Repo layout to create

The user is vendoring LMM-EVQA themselves. The expected target layout:

```
sfv-thesis/
├── lmm-evqa/                          # vendored upstream — DO NOT EDIT files here
│   ├── VideoLLaMA2-audio_visual/
│   ├── Qwen2.5-VL/
│   └── UPSTREAM.md                    # source URL, commit hash, license
├── pipeline/                          # our adapter layer (all our edits go here)
│   ├── configs/
│   │   ├── local.yaml                 # Windows PC (specs TBD); device/dtype/quant filled in once known
│   │   └── cloud_a100.yaml            # device=cuda, fp16, batch=N (also usable locally if GPU permits)
│   ├── data_adapter.py                # Kaggle SnapUGC-{tiny,mini} → LMM-EVQA CSV schema
│   ├── run_videollama2.py             # thin wrapper around lmm-evqa/.../test_single_video.py
│   ├── run_qwen.py                    # thin wrapper around lmm-evqa/.../infer_evqa.py
│   ├── ensemble.py                    # mean/weighted-avg of the two models' scores
│   ├── eval.py                        # SROCC + PLCC against ground truth
│   └── README.md
├── data/                              # gitignored
│   ├── snapugc-tiny/
│   └── snapugc-mini/
├── checkpoints/                       # gitignored
│   ├── videollama2-evqa/
│   └── qwen25vl-evqa/
└── (existing) thesis/, sources/, main/, GEMINI.md, CLAUDE.md, plan.md, ...
```

Invariant: `git diff lmm-evqa/` stays empty. All our logic lives in `pipeline/`.

## Step-by-step

### Phase 0 — Repo housekeeping (do first)

1. **Repo directory** has been renamed to `sfv-thesis` (`~/Code/GitHub/sfv-thesis`). Phase 0 rename is complete.
2. **Search and replace old names** across tracked files: `THESIS-gemini` and `THESIS` have been replaced with `sfv-thesis`. Re-run `grep -rn "THESIS-gemini\|~/Code/GitHub/THESIS[^-]" .` after future scaffolding to catch any regressions.
3. **Write this plan to `plan.md` at the repo root** — already done. Future scope updates live here, not in `~/.claude/plans/`.
4. **Update `CLAUDE.md`** with: (a) new repo name, (b) a new section *"Baseline Reproduction"* pointing to `plan.md` and `pipeline/`, (c) the rule "never edit files inside `lmm-evqa/`; all adapter code lives in `pipeline/`", (d) the modular config rule (hardware differences live only in `pipeline/configs/*.yaml`), (e) soften the "frozen backbones" constraint — it's now an *optional future variant*, not a current requirement.

### Phase A — Scaffold (no GPU needed)

1. User vendors `lmm-evqa/` (clone, drop `.git`, commit). Add `lmm-evqa/UPSTREAM.md` with: source URL `https://github.com/sunwei925/LMM-EVQA`, commit hash, license text from upstream.
2. Add `.gitignore` entries for `data/`, `checkpoints/`, `*.mp4`, `*.pth`, `*.safetensors`.
3. Create `pipeline/` skeleton with empty stubs and `pipeline/README.md`.
4. Add `pipeline/configs/local.yaml` and `pipeline/configs/cloud_a100.yaml` with the same keys (device, dtype, quantization, batch_size, num_frames=8, model paths, data paths). Same key set on both — only values differ. This is the swap point. Leave `local.yaml` values as TBD placeholders until Windows PC specs are confirmed.
5. Manage Python deps via `uv` from inside `pipeline/`. Because `pipeline/` orchestrates upstream via subprocess (it does **not** import `torch` / `transformers` / `videollama2`), the uv env stays light: `uv add pandas scipy pyyaml tqdm`. Heavy ML deps live only in upstream's conda envs (one per model — see Phase C/D).

### Phase B — Dataset adapter

1. User downloads SnapUGC-tiny from Kaggle into `data/snapugc-tiny/`.
2. Inspect its CSV schema (columns, GT score column name) — likely differs from LMM-EVQA's expected format.
3. Implement `pipeline/data_adapter.py`: reads the Kaggle CSV, emits an LMM-EVQA-compatible CSV (columns the upstream scripts expect: video filename + GT engagement score). Writes to `data/snapugc-tiny/lmm_evqa_format.csv`.
4. Smoke test: print first 5 rows, assert all referenced video files exist on disk.

### Phase C — VideoLLaMA2 inference path

1. Use `lmm-evqa/VideoLLaMA2-audio_visual/download_model_weight.py` (or manual Baidu Yun pull, code `3aqc`) to fetch fine-tuned weights → `checkpoints/videollama2-evqa/`.
2. Create a dedicated conda env for VideoLLaMA2 (e.g. `conda create -n lmmevqa-videollama2 python=3.10`), then install upstream's `lmm-evqa/VideoLLaMA2-audio_visual/requirements.txt` into it. This env is isolated from the `pipeline/` uv env — they never share an interpreter. Record the conda env name in `pipeline/configs/local.yaml` so the wrapper knows which env to invoke.
3. Implement `pipeline/run_videollama2.py` as a **subprocess driver**: loads `configs/local.yaml`, builds the command line for upstream's `videollama2/test_single_video.py` (or `run_validation.sh`), invokes it via `subprocess.run([...])` inside the VideoLLaMA2 conda env (e.g. `conda run -n lmmevqa-videollama2 python ...`), iterates over rows of the adapted CSV, and aggregates predictions into `outputs/videollama2_predictions.csv`. The wrapper does **not** import any upstream module.
4. Single-video smoke test on the smallest video in SnapUGC-tiny. Confirm a numeric prediction comes out. Time it. Decide: continue locally for the full tiny set, or move now to cloud.

### Phase D — Qwen2.5-VL inference path

1. Fetch Qwen2.5-VL fine-tuned weights (Baidu Yun, code `98zr`) → `checkpoints/qwen25vl-evqa/`.
2. Create a separate conda env for Qwen2.5-VL (e.g. `conda create -n lmmevqa-qwen python=3.10`) and install its deps per upstream's README. Keep it distinct from the VideoLLaMA2 env — their `transformers` pins differ.
3. Implement `pipeline/run_qwen.py` as a subprocess driver wrapping upstream's `infer_evqa.py`, same I/O contract as `run_videollama2.py` (invoke via `conda run -n lmmevqa-qwen python ...`). Writes `outputs/qwen_predictions.csv`. No in-process imports of upstream code.
4. Single-video smoke test.

### Phase E — Ensemble + evaluation

1. Implement `pipeline/ensemble.py`: reads both prediction CSVs, joins on video id, outputs per-video averaged score. Start with simple mean; expose a `--weights 0.5,0.5` flag for later tuning.
2. Implement `pipeline/eval.py`: computes **SROCC** (`scipy.stats.spearmanr`) and **PLCC** (`scipy.stats.pearsonr`) of ensemble predictions vs ground truth from the adapted CSV. Print per-model and ensemble numbers.
3. Run on full SnapUGC-tiny. Compare numbers against LMM-EVQA's reported 0.707 / 0.714 — they will be lower because (a) tiny is a different distribution and (b) the public weights may be subset-trained. Document the gap.

**Out of scope for the baseline phase, in scope for the extension phase:** patch each upstream inference script to dump the pre-regression hidden state to `outputs/embeddings/<video_id>.pt` alongside the score. This is the single sanctioned exception to "never edit `lmm-evqa/`" — keep the diff minimal, document it in `lmm-evqa/UPSTREAM.md`, and gate it behind a CLI flag so the unmodified inference path still works. The thesis's mid-fusion model (LMM embeddings ⊕ creator metadata) consumes those `.pt` files later.

### Phase F — Cloud-portability check

1. Confirm the *only* code path that branches on hardware is `pipeline/configs/*.yaml`. No `if torch.cuda.is_available()` / device-detection branches scattered through `pipeline/` code.
2. Write a `pipeline/README.md` cloud-run section: `git clone`, `cd pipeline && uv sync`, download checkpoints, edit `cloud_a100.yaml`, `uv run python -m pipeline.run_videollama2 --config configs/cloud_a100.yaml`. No code changes.
3. (Optional, later) Dockerfile for a known-good CUDA env.

## Critical files

**To create**: everything under `pipeline/`, `lmm-evqa/UPSTREAM.md`, `.gitignore` updates, `pipeline/configs/local.yaml`, `pipeline/configs/cloud_a100.yaml`.

**To reuse from upstream (read-only)**:
- `lmm-evqa/VideoLLaMA2-audio_visual/videollama2/test_single_video.py`
- `lmm-evqa/VideoLLaMA2-audio_visual/run_validation.sh`
- `lmm-evqa/Qwen2.5-VL/infer_evqa.py`
- `lmm-evqa/Qwen2.5-VL/test_single_video_qwenvl.py`
- `lmm-evqa/VideoLLaMA2-audio_visual/prepare_dataset.py` and `lmm-evqa/Qwen2.5-VL/prepare_dataset_qwenvl.py` — reference these to understand the expected CSV schema before writing `data_adapter.py`.

**To leave alone**: `thesis/`, `sources/`, `GEMINI.md`, `main/.env`.

## Verification

End-to-end smoke test (definition of done for this plan). Run from inside `pipeline/` so `uv run` picks up the project venv:

```bash
cd pipeline

# 1. Adapter produces a valid CSV
uv run python -m pipeline.data_adapter --input ../data/snapugc-tiny --output ../data/snapugc-tiny/lmm_evqa_format.csv

# 2. Each model runs on at least one video and emits a numeric prediction
uv run python -m pipeline.run_videollama2 --config configs/local.yaml --limit 1
uv run python -m pipeline.run_qwen        --config configs/local.yaml --limit 1

# 3. Full tiny run + ensemble + eval prints SROCC/PLCC
uv run python -m pipeline.run_videollama2 --config configs/local.yaml
uv run python -m pipeline.run_qwen        --config configs/local.yaml
uv run python -m pipeline.ensemble --videollama ../outputs/videollama2_predictions.csv --qwen ../outputs/qwen_predictions.csv --output ../outputs/ensemble.csv
uv run python -m pipeline.eval     --predictions ../outputs/ensemble.csv --gt ../data/snapugc-tiny/lmm_evqa_format.csv
```

Pass criteria: SROCC and PLCC are real finite numbers (sign correct, magnitude > 0) on SnapUGC-tiny. Then promote to SnapUGC-mini.

## Open questions for group discussion

Items the team needs to specify before the corresponding phase can move. Ordered by urgency.

### Urgent — block phases that are otherwise ready

1. **Windows PC GPU specs.** What card / VRAM / driver / CUDA version is on the Windows box? Until known, `pipeline/configs/local.yaml` stays placeholder, and we can't decide between (a) full fp16 inference locally, (b) 4-bit quantization fallback, (c) skip straight to cloud. *Blocks Phase C step 4 onward.*
2. **SnapUGC dataset access + schema.** Confirm we actually have download access. Once downloaded, share: column list, GT score column name, train/val/test split file format, video filename convention. *Blocks Phase B entirely.*
3. **"tiny" / "mini" subset definition.** Are these (a) official SnapUGC releases, (b) self-carved from the VQualA 2025 EVQA challenge val split, or (c) self-carved from raw SnapUGC? Choice affects whether our SROCC/PLCC numbers are comparable to LMM-EVQA's reported 0.707 / 0.714. *Blocks Phase B + Phase E pass criteria.*
4. **LMM-EVQA fine-tuned checkpoint access (Baidu Yun).** Codes `3aqc` (VideoLLaMA2) and `98zr` (Qwen2.5-VL). Does anyone in the group have a working Baidu account / VPN? If blocked, fallback is contacting Sun et al. or fine-tuning ourselves on cloud (large detour). *Blocks Phase C step 1 + Phase D step 1.*

### Medium — needed before cloud move / public release

5. **Cloud GPU provider.** RunPod / Lambda / Vast.ai / university cluster? Pick one before writing `cloud_a100.yaml` properly. Affects cost estimate and whether we need a Dockerfile.
6. **LMM-EVQA license.** Confirm upstream's license terms before treating the vendored `lmm-evqa/` copy as fine to push to a public repo. Record findings in `lmm-evqa/UPSTREAM.md`.
7. **Ensemble weighting strategy.** Default is simple mean of VideoLLaMA2 and Qwen scores. Do we tune `0.5 / 0.5` vs `0.6 / 0.4` etc. on a held-out split, or stick with mean for the baseline?

### Low — can defer until extension phase begins

8. **Pre-regression hidden-state patch.** Confirm exactly which tensor / layer to dump from each upstream inference script, what file format (`.pt` vs `.npz`), and naming convention before writing the patch. Only relevant once baseline numbers are reproduced.
9. **Ethics / IRB for thesis dataset (Filipino creator donations).** Not blocking the LMM-EVQA baseline, but on the thesis timeline (May–Aug 2026 dataset building) — needs to be in flight before that window opens.
