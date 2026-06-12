# Session Log

Rolling record of what changed in this repo, who decided what, and what's left open. Newest entry on top. See `CLAUDE.md` → "Session Log" for the convention.

---

## 2026-06-13

### Files touched

- `CLAUDE.md` — synced against schoolMem: proactive schoolMem check rule (task/deadline/status questions read `active-tasks.md` first); added 2026-06-23 early-ethics-review and 2026-07-14 proposal-docs deadlines; relabeled 07-18–19 as mock defense; data source updated to two-CSV extension export + researcher-side public video download (MP4 donation dropped 2026-06-09); InternVideo2 removed from architecture (two backbones remain).
- `outputs/ethics-review.md`, `outputs/ethics-revision-instructions.md`, `outputs/ethics-revision-references.md` — renamed with `2026-06-09-` prefix to match dated convention; references in this log updated.

### Decisions made

- **InternVideo2 removal is decided**, not tentative — CLAUDE.md architecture section updated; §4.2 branch edit still pending in manuscript.
- **schoolMem `active-tasks.md` is the authoritative task tracker** — repo `thesis-tasks.md` is a coarser backlog; also saved as persistent memory.

### Manuscript section status

- No manuscript text changed this session (repo/context maintenance only).

### Open items / next steps

- §4.2 — remove InternVideo2 branch in manuscript text (June task).
- 2026-06-23 revised manuscript for early ethics review is the nearest deadline.
- Active-tasks checkbox sync from 2026-06-09 sessions was never performed — "This Week" rows in schoolMem still unchecked; verify which are actually done before flipping.

---

## 2026-06-09 (session 2)

### Files touched

- `outputs/2026-06-09-ethics-review.md` — **new**. Full inconsistency audit across all four ethics forms (General Checklist, Review Form, Checklist A, Checklist G). 5 critical/major findings, 3 minor.
- `outputs/2026-06-09-ethics-revision-instructions.md` — **new**. Prompt-style revision instructions for each Google Doc: Step 2 Chrome extension update, Step 3 Google Form update, Checklist A submission fix, adviser PhD credential, Esleta comma, Checklist G audio checkbox, Q1 clarification note, Q8 storage detail, Q9 kept NO (user decision), Checklist G withdrawal procedure filled in.
- `outputs/2026-06-09-ethics-revision-references.md` — **new**. Separate prompt for the Review Form's Reference Document column: all 3.1.x section numbers → 4.1.x, "Table 3.1" → "Figure 4.1", "Table 3.2" → "Table 4.1", page numbers flagged for verification.
- `docs/2026-06-01-THSST1-ethics-review-form.md` — **deleted** (outdated execution plan).
- `outputs/revised_ethics.txt` — **deleted** (outdated draft).

### Decisions made

- **Q9 (future research reuse) stays NO** — user explicitly kept this; Checklist A's 3-year retention language is not treated as "making data available for future research" in the form's sense.
- **Q6 (automated system) stays NO** — Chrome extension runs on the creator's own device/session; not classified as researcher-deployed automation.
- **Q1 (publicly available) stays YES** — add clarification note that data is voluntarily donated from private creator dashboards, not scraped from public pages.
- All changes applied to Google Docs by user in this session.

### Manuscript section status

- Ethics forms: revisions applied in Google Docs. PDFs in `thesis/6/1/26/` are the original submitted versions — do not overwrite unless re-exporting from Docs.
- Page numbers in the Reference Document column are placeholders `[verify]` — need to be filled from the compiled manuscript before re-submission.

### Open items / next steps

- Fill in `[verify]` page numbers in the Reference Document column once the manuscript is compiled.
- Re-export updated ethics form PDFs from Google Docs when ready and replace files in `thesis/` snapshot.
- Informed Consent Form still open (not yet drafted or reviewed).
- Duplicate lowercase PDFs (`checklist-A.pdf`, `general_checklist.pdf`) in `thesis/6/1/26/` — confirm whether these are identical to the named versions and delete if so.

---

## 2026-06-09

### Files touched

- `outputs/2026-06-01-methodology-3.1-building-the-dataset.md` — renumbered §3.1 → §4.1 (Methodology = Ch 4); reworded inline cross-references (no more `§X.Y.Z` callouts in prose); applied Option B for TikTok "Your Rights and Choices" framing; Step 3 changed to Google Form; Step 3 confirmation reworded (Option B); changes-summary and reviewer-checklist headers updated to match new numbering.
- `outputs/2026-06-09-methodology-4.1-latex-and-bib.md` — **new**. LaTeX paste-ready block for §4.1 (intro + §4.1.1 / §4.1.2 / §4.1.3) plus two BibTeX entries (`tiktok_privacy_2026`, `tiktok_tos_2026`). Mirrors the markdown draft and tracks the same option picks.
- `CLAUDE.md` — added "Revising Prose" section (always present 3 labeled options before editing thesis prose); added "Session Log" section (rule that produced this file).
- `~/.claude/projects/-Users-achibukz-Code-GitHub-sfv-thesis/memory/feedback_rewording_options.md` + `MEMORY.md` — saved the 3-options rule as a persistent feedback memory.

### Decisions made

- **TikTok rights framing** → Option B ("Each participant exercises the access right granted to users under TikTok's Privacy Policy ‘Your Rights and Choices' clause…"). Replaces the previous "framing relies on…" phrasing.
- **Step 3 submission channel** → Google Form (replaces "secure, encrypted ingestion portal").
- **Step 3 closing sentence** → Option B ("Before the upload completes, the creator must affirm a final consent statement on the form acknowledging the transfer.").
- **Step 5 verification sentence** → Option A (clearest cause/effect: "…to prevent a row from being matched to the wrong video, since creators sometimes re-upload videos with the same caption and produce duplicate filenames.").
- **Rewording protocol** → always offer A/B/C options first; codified in `CLAUDE.md` and in auto-memory.

### Manuscript section status

- §4.1 (intro) — drafted; **not yet pasted into LaTeX**.
- §4.1.1 Data Acquisition — drafted, **accepted into LaTeX by user**.
- §4.1.2 Dataset Variables and Metrics — drafted, **pending — not yet pasted into LaTeX**.
- §4.1.3 Ethical Safeguards and Limitations — drafted, **accepted into LaTeX by user**.
- BibTeX: `tiktok_privacy_2026` and `tiktok_tos_2026` **added to `myreferences.bib` by user**. `li2024delving` / `li2025vquala` already present, no action needed.

### Open items / next steps

- Decide whether §4.1 intro + §4.1.2 also go into LaTeX, or stay markdown-only until further revision.
- Re-render `figures/chap3/Data Acquisition.png` so Step 2 reads "Chrome Extension Export" (figure label `fig:data_acqui` stays the same).
- Confirm `data_specs/extension-csv-schema.md` exists, or generate it from the extension source before manuscript freeze.
- Confirm `appendix_A.tex` is the locked slot for the informed consent form (vs `appendix_B.tex`).
- Confirm the placeholder `year = {2024}` and `urldate = {2026-06-09}` values in the new BibTeX entries match the live TikTok policy revision dates.
- `active-tasks.md` still has a separate item for "§4.1.3 — refine participant criteria specifically." The current §4.1.3 rewrite covers identity/likeness, retention, withdrawal, and platform compliance, but does not refine the participant inclusion criteria (those sit in §4.1.1's "active TikTok content creators … 30–50 Filipino micro-creators"). Open question for next session.
- Storage / encryption regime for the dataset still deferred pending adviser decision (carried over from prior session).
