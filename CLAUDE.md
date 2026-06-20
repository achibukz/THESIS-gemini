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

- `thesis/` — LaTeX source files for each chapter. Submission snapshots are nested under `MM/DD/YY/` (e.g. `thesis/6/1/26/` = June 1, 2026). The latest dated subfolder is the live working snapshot; older folders are frozen prior submissions. Each snapshot contains `chapter_1.tex`–`chapter_4.tex`, `abstract-english.tex`, `title_page.tex`, `myreferences.bib`, ethics checklist PDFs, and the `figures/` tree.
- `docs/` — Per-deliverable execution plans (one self-contained brief per deliverable, dated `YYYY-MM-DD-<scope>-<topic>.md`). Treat these as living source-of-truth for in-flight thesis work. Format mirrors `docs/2026-06-01-THSST1-thesis-repo-update-plan.md` (Context → Anchor → What to Produce → Sources → Out of Scope → Format Conventions).
- `sources/` — Academic references (`references.bib`, `rrl_matrix.csv`) and source PDFs in `sources/papers/`
- `outputs/` — Generated artifacts (scripts, revised sections, model predictions)
- `data_specs/`, `sensitive_data/` — Dataset specifications and participant data (treat as sensitive)
- `lmm-evqa/` — **Vendored upstream** code from https://github.com/sunwei925/LMM-EVQA. Treated as read-only; never edit files inside this directory.
- `pipeline/` — Our `uv`-managed Python project. All local code will live here: adapters, configs, dataset loaders, ensemble, evaluation, and the `.venv`. **Currently a scaffold** (hello-world `main.py` only) — the Baseline Reproduction rules below describe the planned architecture, not implemented code. `pipeline/configs/*.yaml` does not exist yet.
- `tiktok-analytics-exporter/` — The TikTok Analytics Exporter Chrome extension source (manifest, content/injected scripts, popup). This is Step 2 of the data acquisition pipeline. `chromeextension.md` at repo root is its plan/spec document.
- `data/`, `checkpoints/` — Gitignored. Hold downloaded SnapUGC subsets and fine-tuned model weights respectively.
- `plan.md` — Active working plan for the LMM-EVQA baseline reproduction. Update this when scope changes.
- `GEMINI.md` — Primary project context document; canonical source for terminology, architecture, and constraints. Keep it in sync with the Locked Decisions in this file — if they conflict, the Locked Decisions win and GEMINI.md must be updated.

There is **no task tracker in this repo** — the authoritative tracker is schoolMem's `active-tasks.md` (see "Syncing with School Context"). The former `thesis-tasks.md` was removed 2026-06-13.

## Syncing with School Context

Check schoolMem **proactively, without being asked**, whenever the user asks about active tasks, deadlines, blockers, meeting decisions, or overall thesis status (e.g. "what are our active tasks"). The authoritative task tracker is `~/Documents/Obsidian/schoolMem/wiki/AY2526-T3/THSST1-Thesis-in-Software-Technology-1/thesis/active-tasks.md` — it is the **only** task tracker (the repo-local `thesis-tasks.md` was removed 2026-06-13). `blockers.md` and `decisions.md` live in the same folder.

Additionally, when the user asks to "update yourself," "sync context," "check schoolmem," or otherwise refresh thesis context, read the latest material under the Obsidian vault before answering:

- `~/Documents/Obsidian/schoolMem/raw/AY2526-T3/THSST1/` — raw meeting minutes, session logs, and the task tracker.
- `~/Documents/Obsidian/schoolMem/wiki/AY2526-T3/THSST1-Thesis-in-Software-Technology-1/notes/` — dated session notes and meeting summaries.
- `~/Documents/Obsidian/schoolMem/wiki/AY2526-T3/THSST1-Thesis-in-Software-Technology-1/topics/` — durable topic pages (ECR, NAWP, pipeline architecture, dataset collection, etc.).
- `~/Documents/Obsidian/schoolMem/output/` — generated briefs handed off to this repo (often the source of new `docs/` plans).

Prefer the freshest meeting minutes when terminology, decisions, or deadlines conflict with what's in CLAUDE.md — then propose updating CLAUDE.md to match.

## THSST1 Term Deadlines (AY2526-T3)

Current term: AY2526-T3. THSST1 is the manuscript-prep term for the thesis.

- **2026-06-23** — revised manuscript due for early ethics review (panelist pre-review).
- **2026-07-06** — complete manuscript due (all chapters, including the prototype chapter for the Chrome extension). Missing this means non-endorsement for defense.
- **2026-07-13** — endorsement-to-defense deadline.
- **2026-07-14** — Thesis 1 submission: proposal defense documents.
- **2026-07-18 – 2026-07-19** — mock defense.
- **2026-07-20 – 2026-07-25** — defense week, face-to-face (DLSU local or Manila; hybrid Zoom only if a panel member cannot attend in person).
- **Mandatory project sprints**: 2026-06-20, 2026-06-27 – 2026-06-28. Attendance required for all group members.
- **Big-group meetings** now run as an accountability circle: each group presents a 5–10 min progress report and commits to next-week goals.

## Locked Decisions (from advising)

These were approved by the adviser and should not be relitigated without explicit instruction:

- **Data acquisition framing (approved 2026-05-26)**: the TikTok Analytics Exporter Chrome extension is a creator-side tool. It automates the creator's *right to access their own information* (TikTok Privacy Policy, "Your Rights and Choices"). The data flow is creator → creator; the study is the *recipient* of donated data, not the *acquirer*. This is the load-bearing claim for both Methodology §4.1 and the Ethics Review Form.
- **Chrome extension is Step 2 of the data acquisition pipeline** — it replaces TikTok's built-in "Download your data" export in Methodology Ch 4.1.1. Pipeline order (six steps as of the 2026-06-09 v2 revision): Consent → Extension Export → Submission (analytics CSV via Google Form) → Video Download (researchers retrieve videos from the public platform via video IDs) → Anonymization → Verification.
- **No face blurring.** The model is not biometric/face-specific. The informed consent form will explicitly state that the creator's likeness may appear in training data; secondary subjects in the background are covered by the same consent. Drop any mention of automated face-detection/blurring from Ch 4.1.3.
- **Synthetic / AI-generated datasets are rejected** — validation overhead outweighs benefit.
- **Topic is locked.** No more pivots; refinement only.

**Stale-term sweep rule:** whenever a decision is locked, reversed, or a term/component is dropped (e.g. InternVideo2, MP4 donation), grep the whole repo (and GEMINI.md) for the old term and list every hit to the user — don't fix only the file at hand. This is how MP4/InternVideo2 references survived in GEMINI.md, `thesis-tasks.md`, and the ethics-revision instructions after the decisions that removed them.

## Internal-Only Notes (do not cite in papers)

- **Matthew Ong dataset** is a suggestion for *post-SnapUGC* cross-dataset validation. It is a proposal only. **Do not cite or reference in the manuscript, RRL, or ethics documents.**

## Technical Model Architecture

Two frozen LMM backbones used as feature extractors (no fine-tuning):
- **VideoLLaMA2** — spatiotemporal + auditory understanding
- **Qwen2.5-VL** — visual-semantic reasoning

(**InternVideo2 was removed** from the architecture — decided June 2026. Methodology §4.2 still needs its branch edited out; do not reference InternVideo2 in new manuscript text.)

Extracted embeddings are concatenated with structured metadata: follower count, account age, posting timestamps.

**Keyframe extraction**: exactly 8 uniform keyframes from the initial video window.

**Benchmarks**: compare against Sun et al. (2025) and Guan et al. (2025).

## Working with LaTeX

Rules:
- Preserve all LaTeX commands unless explicitly asked to remove them
- Keep terminology consistent with `GEMINI.md` (ECR, NAWP, exact LMM names)
- Output revised LaTeX code only, no prose explanation

## Revising Prose (thesis / drafts / docs)

When the user asks to reword, rewrite, or replace any sentence or phrase in thesis prose (markdown drafts in `outputs/`, LaTeX in `thesis/`, or planning docs in `docs/`), **always present 3 distinct rewording options before editing**. Label them A, B, C with a short qualifier (e.g. "rights-first, plain" / "creator as the actor" / "shortest"). Wait for the user to pick before applying the edit. This applies even when the directive sounds like "just reword" — the user still wants choices.

Exception: a pure typo, grammar fix, or numbering correction (e.g. §3.1 → §4.1) does not need options — apply directly.

## Session Log

A rolling `log.md` at the repo root records what each working session changed. **Update `log.md` after each major milestone** — a section accepted into LaTeX, a deliverable closed, a decision settled, a new file landed. Do not log every keystroke or in-flight wording iteration; only milestone-worthy events.

Entry conventions:
- Newest entry on top. Each entry has an H2 date header (`## YYYY-MM-DD`).
- Sub-sections per entry: **Files touched** (path + one-line summary), **Decisions made** (option picks, settled questions), **Manuscript section status** (which §X.Y are drafted / accepted / pending), **Open items / next steps**.
- Reference files by repo-relative path; reference picks by their A/B/C label and a short quote where useful.

After updating `log.md`, cross-check the entry against `~/Documents/Obsidian/schoolMem/wiki/AY2526-T3/THSST1-Thesis-in-Software-Technology-1/thesis/active-tasks.md`. Add a **"Suggested checkbox flips"** list to the log entry: for each logged milestone that plausibly closes an open task in that file, quote the task row and say why. Present the list to the user and flip only the boxes they confirm (`- [ ]` → `- [x]`). Never flip silently; never skip the suggestion step just because no match is exact.

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
- Data source: creator-donated TikTok analytics exports via the Chrome extension (two CSVs: engagement + follower history); researchers retrieve the corresponding videos from the public TikTok platform. **Direct MP4 donation was dropped 2026-06-09** — do not reintroduce it in methodology or ethics text.
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
