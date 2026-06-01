# THS-ST1 — Methodology Refinement (Ch 4)

_Generated: 2026-06-01_

Self-contained brief to paste into Claude inside `~/Code/GitHub/sfv-thesis`. Focused task: refine **Chapter 4 (Methodology)** with the technical granularity the adviser flagged on 2026-05-26, and bring §4.1.1 (Data Acquisition) and §4.1.3 (Ethical Safeguards) in line with the locked, adviser-approved policy.

This plan is a sibling of:
- `docs/2026-06-01-THSST1-thesis-repo-update-plan.md` — the Data Acquisition writeup spec.
- `docs/2026-06-01-THSST1-ethics-review-form.md` — the Ethics Review Form / Consent Form spec.

Read both before starting — they hold the locked framing this plan instantiates inside the methodology text.

---

## Context for Claude

- Adviser feedback (2026-05-26): the methodology section lacks technical granularity. Tighten the data-handling description and the model pipeline so a reader can follow the operational sequence end-to-end.
- The locked decisions (see CLAUDE.md "Locked Decisions") shift two pieces of the chapter that are currently inconsistent with policy: (1) Step 2 of the data acquisition pipeline should be the Chrome extension export, not TikTok's built-in "Download your data" archive; (2) §4.1.3's face-detection-and-blurring paragraph is no longer policy and must be rewritten.
- This deliverable feeds the **2026-07-06 complete-manuscript freeze**. Get the rewrite into the latest dated snapshot under `thesis/` well before then so panel review has time to catch issues.

---

## The Anchor (read this first)

Two coupled rewrites and one expansion:

> **Rewrite A (§4.1.1 Data Acquisition):** Step 2 of the pipeline is the Chrome extension export — replacing the line that currently names TikTok's official "Download your data" archive. The extension is a creator-side automation of the right-of-access granted by TikTok's Privacy Policy ("Your Rights and Choices"). Pipeline order is unchanged: Consent → **Extension Export** → Submission → Anonymization → Verification.
>
> **Rewrite B (§4.1.3 Ethical Safeguards):** Drop the "automated face-detection and blurring algorithms" paragraph. Replace it with the new likeness-consent framing: creators consent to their likeness appearing in donated frames; the model does not perform face recognition or other biometric inference; secondary subjects are covered by the same consent because the donating creator confirms standing to donate the footage.
>
> **Expansion (§4.1 Building the Dataset overall):** Add technical granularity per the 2026-05-26 advising note — data-handling specifics (anonymization mapping table format, verification cross-check fields, storage location, encryption regime), and pipeline specifics (extension output schema, anonymization ID format, MP4 normalization steps).

---

## What Claude Should Produce

### A. §4.1.1 (Data Acquisition and Collection Procedure) — locked rewrite

Edit the existing prose in `thesis/<latest-snapshot>/chapter_4.tex` so:

- The five-step pipeline now reads: **Step 1 Informed Consent → Step 2 Chrome Extension Export → Step 3 Submission → Step 4 Anonymization → Step 5 Verification.**
- The current sentence naming TikTok's built-in archive is removed; the new Step 2 names the Chrome Extension (`tiktok-analytics-exporter`) running locally in the creator's authenticated browser session.
- Add one sentence cross-referencing the extension's privacy posture: no remote endpoint contacted, no credentials stored, no viewer-level data collected.
- Pull DT-1's TikTok policy citation (see the ethics-form plan §D) into a short footnote or inline citation.
- Keep the existing data-flow figure reference (`figures/chap3/Data Acquisition.png`) but note in this plan that the figure should be re-rendered with the updated Step 2 label. (Do not edit the figure — flag it for the diagrams pass.)

### B. §4.1.3 (Ethical Safeguards and Limitations) — locked rewrite

Edit the existing prose so:

- The paragraph beginning "Regarding facial privacy and biometric data..." is rewritten to remove all references to "automated face-detection and blurring algorithms."
- New wording: the informed consent form (Appendix A) explicitly covers the creator's likeness and the likeness of any secondary subjects incidentally appearing in donated frames. The model performs engagement prediction and does not perform facial recognition, identity matching, or other biometric inference. Likeness is processed only as raw multimodal input to the LMM ensemble.
- Strengthen the TikTok-ToS compliance paragraph using the DT-1 citation block.
- Add explicit data retention periods inline: **non-anonymized PII max 2 years, anonymized data max 3 years** from first publication or project completion. Match the wording used in the Ethics Review Form.

### C. §4.1 (Building the Dataset) — technical granularity expansion

Add new prose or extend existing subsections to cover:

1. **Extension output schema.** Document the CSV columns the extension produces (caption, video duration, view counts, watch-time metrics, post timestamp, NAWP, ECR-derivable signals, etc.). Reference the extension source as the canonical schema; if the column list isn't already documented somewhere in the repo, add a `data_specs/extension-csv-schema.md` and link it from the chapter.
2. **MP4 ↔ analytics row alignment.** Describe how the verification step (Step 5) cross-references the analytics-export timestamps with the MP4 metadata to bind each row to the correct video file. Mention the historical caption-vs-video-ID misalignment risk discovered during extension work, and how the post-timestamp + duration pairing resolves it.
3. **Anonymization mapping format.** Describe the alphanumeric ID scheme that replaces creator handles (length, character set, collision strategy) and where the mapping table is stored (encrypted, segregated from the analytics dataset, accessible only to proponents).
4. **Storage / encryption regime.** Name the storage location (encrypted cloud portal for ingestion → encrypted research workstation for processing → encrypted long-term archive for retention period). Avoid naming specific vendors in-text unless the adviser has locked one.
5. **Withdrawal workflow.** Document the operational steps a participant takes to withdraw: contact email → identity verification → mapping-table lookup → cascade deletion across ingestion portal, working dataset, and archive.

### D. §4.2 (Enhancing the Model) — light tightening

This section is mostly stable. Only changes required:

- Make the "frozen backbones" statement consistent with `CLAUDE.md`'s nuance: the **frozen-backbone variant is the current operational stance for the baseline phase**, not an immutable architectural commitment. Wording should leave room for the cloud-fine-tune path without overpromising.
- Add a one-sentence reference to the ensemble combination strategy (simple averaging by default) as implemented in `pipeline/ensemble.py`. Do not name a final weighting; that depends on baseline-reproduction results from MTR-1.

### E. §4.3 (Evaluating the Model) — no changes

Out of scope for this refinement pass.

---

## Source Material Claude Should Use

In the thesis repo:
- `thesis/<latest-snapshot>/chapter_4.tex` — primary edit target.
- `thesis/<latest-snapshot>/myreferences.bib` — add TikTok policy citation here.
- `docs/2026-06-01-THSST1-thesis-repo-update-plan.md` — for the locked data-acquisition framing.
- `docs/2026-06-01-THSST1-ethics-review-form.md` — for the locked consent / retention wording (must match across docs).
- Chrome extension source (in repo) — for the extension CSV schema, output behaviours, and the "does not do X" non-goals.
- `data_specs/` — destination for any new schema docs.

From outside:
- TikTok Privacy Policy (Rest of World): <https://www.tiktok.com/legal/page/row/privacy-policy/en>
- TikTok Terms of Service (Rest of World): <https://www.tiktok.com/legal/page/row/terms-of-service/en>

---

## Open Items to Flag (do not invent answers)

- Whether `appendix_A.tex` or `appendix_B.tex` is the slot the adviser wants for the consent form. The ethics-form plan claims A; confirm during the next 1:1.
- Whether the storage/encryption regime should name a specific vendor or stay vendor-agnostic.
- Whether the figure `figures/chap3/Data Acquisition.png` will be re-rendered in-house or whether the original designer needs to update it.
- Whether the `data_specs/extension-csv-schema.md` already exists somewhere in the repo. If yes, link rather than duplicate.

---

## Out of Scope for This Task

- **DE-2 (participant inclusion/exclusion criteria)** — adviser-deferred.
- **DE-3 (full Informed Consent Form text)** — covered by `docs/2026-06-01-THSST1-ethics-review-form.md`.
- **RRL micro-influencer expansion** (Ch 2) — separate writeup task.
- **Chapter 5 (prototype chapter)** — explicitly out of scope this cycle.
- **MTR-1 baseline reproduction** — covered by `plan.md`.

---

## Format Conventions

- LaTeX only for §4 edits; no Markdown leakage into `.tex` files.
- Preserve all existing LaTeX commands, labels, and figure references unless explicitly rewriting them.
- Bibliography: add TikTok policy entries to `myreferences.bib` as `@misc` with `urldate`.
- Additive edits to §4.1.3 should reuse existing citation keys where possible.
- Match the existing chapter's terminology — `ECR`, `NAWP`, `VideoLLaMA2`, `Qwen2.5-VL`, `InternVideo2` (exact spellings).
- Output revised LaTeX code only, no prose explanation, when producing the actual chapter edits.
