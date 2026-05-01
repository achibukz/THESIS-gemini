# Plan: Recreate LMM-EVQA as a Reproducible Baseline

## Context

The thesis (*"To Predict Is To Believe..."*) names two prior works as benchmarks: **Sun et al. 2025 (LMM-EVQA)** and **Guan et al. 2025**, and uses **Li et al. 2024 (SnapUGC ECR/NAWP)** as the source of the target metrics. Before we can extend either with creator/context features (per the thesis architecture), we need a working, reproducible recreation of LMM-EVQA running against a small SnapUGC subset on this machine. The goal of this plan is to stand up that baseline in a way that:

1. Faithfully reproduces LMM-EVQA's two-model setup (VideoLLaMA2 + Qwen2.5-VL).
2. Runs end-to-end *locally first* on a small subset (SnapUGC-tiny → SnapUGC-mini), with a clean modular boundary so the same code can run on a cloud GPU later with a config swap.
3. Leaves room to layer the thesis's creator/context features on top later, and to (optionally, future work) explore frozen-backbone variants — without needing to fork the upstream repo.

## Reality check (read first)

- **LMM-EVQA is two independent models**, not a fused pipeline. The repo ships `VideoLLaMA2-audio_visual/` and `Qwen2.5-VL/` as separate folders, each with its own `train.sh` / inference scripts. The challenge-winning "ensemble" is simple averaging of the two models' predicted scores — *not* in the upstream repo, we add it ourselves.
- **Hardware on M4 Air**:
  - VideoLLaMA2-7B fp16 ≈ 14 GB; Qwen2.5-VL-7B fp16 ≈ 14 GB. Inference is feasible on M4 Air with 24/32 GB unified memory using MPS + 4-bit quantization (bitsandbytes won't work on Apple Silicon — we use `mlx` / GGUF or fall back to CPU offload). Expect **multiple minutes per video**.
  - **Fine-tuning either model locally is not realistic.** Needs A100/H100-class GPUs. LMM-EVQA ships *fine-tuned* checkpoints — we use those for inference and skip retraining on M4.
  - On 16 GB M4 Air: only single-video, heavily quantized inference; treat as plumbing-only.
  - SnapUGC-tiny end-to-end on M4: hours. On a single A100: minutes.
- **Strategy**: download LMM-EVQA's published fine-tuned weights (Baidu Yun extraction codes `3aqc` and `98zr`) and run **inference only** on M4. Fine-tuning is a future cloud step.

## Repo layout to create

The user is vendoring LMM-EVQA themselves. The expected target layout (after the `THESIS-gemini` → `THESIS` rename in Phase 0):

```
THESIS/
├── lmm-evqa/                          # vendored upstream — DO NOT EDIT files here
│   ├── VideoLLaMA2-audio_visual/
│   ├── Qwen2.5-VL/
│   └── UPSTREAM.md                    # source URL, commit hash, license
├── pipeline/                          # our adapter layer (all our edits go here)
│   ├── configs/
│   │   ├── local_m4.yaml              # device=mps, quantization=4bit, batch=1
│   │   └── cloud_a100.yaml            # device=cuda, fp16, batch=N
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

1. **Rename repo directory** `THESIS-gemini` → `THESIS`. Caveat: cannot be done from inside the directory itself; user runs `mv ~/Code/GitHub/THESIS-gemini ~/Code/GitHub/THESIS` from a parent shell, then reopens the editor / Claude session at the new path. After rename, also update the git remote URL if it points anywhere with `THESIS-gemini` in it (`git remote -v` to confirm).
2. **Search and replace `THESIS-gemini` → `THESIS`** across tracked files. Current `grep -rn "THESIS-gemini" .` is empty, so this is a no-op for now — but re-run after any future scaffolding to be safe.
3. **Write this plan to `plan.md` at the repo root** — already done. Future scope updates live here, not in `~/.claude/plans/`.
4. **Update `CLAUDE.md`** with: (a) new repo name, (b) a new section *"Baseline Reproduction"* pointing to `plan.md` and `pipeline/`, (c) the rule "never edit files inside `lmm-evqa/`; all adapter code lives in `pipeline/`", (d) the modular config rule (hardware differences live only in `pipeline/configs/*.yaml`), (e) soften the "frozen backbones" constraint — it's now an *optional future variant*, not a current requirement.

### Phase A — Scaffold (no GPU needed)

1. User vendors `lmm-evqa/` (clone, drop `.git`, commit). Add `lmm-evqa/UPSTREAM.md` with: source URL `https://github.com/sunwei925/LMM-EVQA`, commit hash, license text from upstream.
2. Add `.gitignore` entries for `data/`, `checkpoints/`, `*.mp4`, `*.pth`, `*.safetensors`.
3. Create `pipeline/` skeleton with empty stubs and `pipeline/README.md`.
4. Add `pipeline/configs/local_m4.yaml` and `pipeline/configs/cloud_a100.yaml` with the same keys (device, dtype, quantization, batch_size, num_frames=8, model paths, data paths). Same key set on both — only values differ. This is the swap point.
5. Update `main/requirements.txt` (or add `pipeline/requirements.txt`) with: `torch`, `torchvision`, `transformers`, `accelerate`, `decord`, `pillow`, `pandas`, `scipy` (for SROCC/PLCC), `pyyaml`, `tqdm`. Note: do **not** install upstream's `requirements.txt` globally — keep upstream deps inside its own venv if they conflict. `uv sync` from project root.

### Phase B — Dataset adapter

1. User downloads SnapUGC-tiny from Kaggle into `data/snapugc-tiny/`.
2. Inspect its CSV schema (columns, GT score column name) — likely differs from LMM-EVQA's expected format.
3. Implement `pipeline/data_adapter.py`: reads the Kaggle CSV, emits an LMM-EVQA-compatible CSV (columns the upstream scripts expect: video filename + GT engagement score). Writes to `data/snapugc-tiny/lmm_evqa_format.csv`.
4. Smoke test: print first 5 rows, assert all referenced video files exist on disk.

### Phase C — VideoLLaMA2 inference path

1. Use `lmm-evqa/VideoLLaMA2-audio_visual/download_model_weight.py` (or manual Baidu Yun pull, code `3aqc`) to fetch fine-tuned weights → `checkpoints/videollama2-evqa/`.
2. Set up upstream's conda env per their README (Python 3.9). Keep this env separate from the thesis project env — it will have CUDA-pinned deps that fight Apple Silicon. On M4: install CPU/MPS-compatible torch instead of upstream's pinned wheels and accept that some upstream training scripts won't run, only inference.
3. Implement `pipeline/run_videollama2.py`: loads `configs/local_m4.yaml`, calls upstream's `videollama2/test_single_video.py` (or `run_validation.sh` equivalent in Python) over each row of the adapted CSV, writes predictions to `outputs/videollama2_predictions.csv`.
4. Single-video smoke test on the smallest video in SnapUGC-tiny. Confirm a numeric prediction comes out. Time it. Decide: continue locally for the full tiny set, or move now to cloud.

### Phase D — Qwen2.5-VL inference path

1. Fetch Qwen2.5-VL fine-tuned weights (Baidu Yun, code `98zr`) → `checkpoints/qwen25vl-evqa/`.
2. Install Qwen2.5-VL deps per upstream's README. HuggingFace-native, generally smoother than VideoLLaMA2 on M4.
3. Implement `pipeline/run_qwen.py`: wraps upstream's `infer_evqa.py`, same I/O contract as `run_videollama2.py`. Writes `outputs/qwen_predictions.csv`.
4. Single-video smoke test.

### Phase E — Ensemble + evaluation

1. Implement `pipeline/ensemble.py`: reads both prediction CSVs, joins on video id, outputs per-video averaged score. Start with simple mean; expose a `--weights 0.5,0.5` flag for later tuning.
2. Implement `pipeline/eval.py`: computes **SROCC** (`scipy.stats.spearmanr`) and **PLCC** (`scipy.stats.pearsonr`) of ensemble predictions vs ground truth from the adapted CSV. Print per-model and ensemble numbers.
3. Run on full SnapUGC-tiny. Compare numbers against LMM-EVQA's reported 0.707 / 0.714 — they will be lower because (a) tiny is a different distribution and (b) the public weights may be subset-trained. Document the gap.

### Phase F — Cloud-portability check

1. Confirm the *only* code path that branches on hardware is `pipeline/configs/*.yaml`. No `if torch.backends.mps.is_available()` scattered through `pipeline/`.
2. Write a `pipeline/README.md` cloud-run section: `git clone`, `uv sync`, download checkpoints, edit `cloud_a100.yaml`, `python -m pipeline.run_videollama2 --config pipeline/configs/cloud_a100.yaml`. No code changes.
3. (Optional, later) Dockerfile for a known-good CUDA env.

## Critical files

**To create**: everything under `pipeline/`, `lmm-evqa/UPSTREAM.md`, `.gitignore` updates, `pipeline/configs/local_m4.yaml`, `pipeline/configs/cloud_a100.yaml`.

**To reuse from upstream (read-only)**:
- `lmm-evqa/VideoLLaMA2-audio_visual/videollama2/test_single_video.py`
- `lmm-evqa/VideoLLaMA2-audio_visual/run_validation.sh`
- `lmm-evqa/Qwen2.5-VL/infer_evqa.py`
- `lmm-evqa/Qwen2.5-VL/test_single_video_qwenvl.py`
- `lmm-evqa/VideoLLaMA2-audio_visual/prepare_dataset.py` and `lmm-evqa/Qwen2.5-VL/prepare_dataset_qwenvl.py` — reference these to understand the expected CSV schema before writing `data_adapter.py`.

**To leave alone**: `thesis/`, `sources/`, `GEMINI.md`, `main/.env`.

## Verification

End-to-end smoke test (definition of done for this plan):

```bash
# 1. Adapter produces a valid CSV
python -m pipeline.data_adapter --input data/snapugc-tiny --output data/snapugc-tiny/lmm_evqa_format.csv

# 2. Each model runs on at least one video and emits a numeric prediction
python -m pipeline.run_videollama2 --config pipeline/configs/local_m4.yaml --limit 1
python -m pipeline.run_qwen          --config pipeline/configs/local_m4.yaml --limit 1

# 3. Full tiny run + ensemble + eval prints SROCC/PLCC
python -m pipeline.run_videollama2 --config pipeline/configs/local_m4.yaml
python -m pipeline.run_qwen          --config pipeline/configs/local_m4.yaml
python -m pipeline.ensemble  --videollama outputs/videollama2_predictions.csv --qwen outputs/qwen_predictions.csv --output outputs/ensemble.csv
python -m pipeline.eval      --predictions outputs/ensemble.csv --gt data/snapugc-tiny/lmm_evqa_format.csv
```

Pass criteria: SROCC and PLCC are real finite numbers (sign correct, magnitude > 0) on SnapUGC-tiny. Then promote to SnapUGC-mini.

## Open risks / things to watch

- **Apple Silicon vs upstream's pinned CUDA deps**: upstream conda env will likely error on `torch` install. Plan: install MPS-compatible `torch` separately, accept that training-only scripts won't run on M4.
- **Baidu Yun checkpoint access**: may require a Baidu account / VPN. If blocked, contact authors or fall back to fine-tuning ourselves on cloud (large detour).
- **Kaggle CSV schema mismatch**: SnapUGC-tiny may not include the exact GT score column LMM-EVQA expects. The adapter handles this, but if GT is missing entirely we can only do prediction, not evaluation.
- **Quantization on M4 for VideoLLaMA2**: bitsandbytes is CUDA-only. May need `torch.float16` + CPU offload instead, which is slow. Acceptable for plumbing; not for full runs.
- **License**: confirm LMM-EVQA's license before committing the vendored copy publicly. Note it in `UPSTREAM.md`.
