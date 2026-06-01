# Thesis Vocabulary

Source of truth for project-specific terminology, locked framings, banned terms, and style choices used by the `thesis-humanizer` skill. Edit this file by hand as the project evolves; the skill will pick up changes on the next invocation.

---

# Canonical terms

- ECR — Engagement Continuation Rate (expand on first use per chapter)
- NAWP — Normalized Average Watch Percentage (expand on first use per chapter)
- SFV — short-form video (lowercase expansion; "SFV" is the bare acronym in second uses)
- SROCC — Spearman Rank-Order Correlation Coefficient (never expand inline; assumed known)
- PLCC — Pearson Linear Correlation Coefficient (never expand inline; assumed known)
- RMSE — Root Mean Squared Error (never expand inline; assumed known)
- AWT — Average Watch Time (expand on first use per chapter)
- LMM — Large Multimodal Model (expand on first use per chapter)
- LMM-EVQA — exact spelling, hyphenated, uppercase E V Q A
- VideoLLaMA2 — exact spelling, no hyphen, capital V L L M
- Qwen2.5-VL — exact spelling, dot and hyphen preserved
- InternVideo2 — exact spelling, no hyphen
- SnapUGC — exact spelling, capital S and U G C
- SnapUGC-tiny, SnapUGC-mini — subset names, exact spelling and hyphenation
- TikTok Studio — exact spelling, two capitalized words
- TikTok Analytics Exporter — exact spelling, project's Chrome extension
- Filipino micro-creator(s) — preferred phrasing for participants
- MP4 — uppercase
- API — uppercase

# Locked framings

- Data acquisition is creator-side data portability, not scraping. The Chrome extension automates the creator's right to access their own information under TikTok's Privacy Policy ("Your Rights and Choices"). Approved by adviser on 2026-05-26.
- Pipeline Step 2 is the Chrome Extension Export, replacing TikTok's built-in "Download your data" archive. The five-step pipeline is: Consent → Extension Export → Submission → Anonymization → Verification.
- No face blurring or automated face-detection algorithms. The informed consent form explicitly covers the use of the creator's likeness as part of training data. The model performs engagement prediction and does not perform biometric inference. Secondary subjects in donated footage are covered by the same consent because the donating creator confirms standing to donate.
- Data retention: non-anonymized PII max 2 years, anonymized data max 3 years, from first publication or project completion.
- Synthetic and AI-generated datasets are rejected for this study. Validation overhead is judged to outweigh the benefit.
- The frozen-backbone variant is the current baseline-reproduction stance. The manuscript should describe it as the operational starting point, not an immutable architectural commitment; a cloud fine-tune path remains open.
- LMM-EVQA is two independent models (VideoLLaMA2 and Qwen2.5-VL) combined by a project-built ensemble (simple averaging by default), not a single fused pipeline.
- Target metrics for the thesis are ECR and NAWP. Raw views, likes, and shares are not substitutes.

# Banned terms

- Matthew Ong dataset — internal-only suggestion for post-SnapUGC cross-dataset validation. Do not cite or name in the manuscript, the RRL, the ethics review form, the consent form, the appendices, slides, or any artefact intended for the adviser, panel, or public reader.
- "automated face-detection and blurring algorithms" — superseded; do not reintroduce.
- Promotional / press-release adjectives: "groundbreaking", "transformative", "cutting-edge", "revolutionary", "state-of-the-art" (use specific comparisons instead, with citations).

# Style notes

- Use LaTeX backtick double-quotes for scare quotes and short verbatim references: `` ``cold start'' ``, `` ``Download your data'' ``. Never use curly Unicode quotes.
- Every empirical claim, quantitative figure, and "researchers have shown" assertion ends in `\cite{key}` to a real key in `myreferences.bib`. If unsure whether a key exists, flag the gap rather than inventing one.
- Third person throughout chapter prose. The study, the proposed model, this work — not I or we, unless the surrounding paragraph already uses them.
- Use "Furthermore" rather than "Moreover" for additive transitions (matches existing chapter voice).
- Use `\textbf{...}` sparingly: typically once per chapter on the research question or a load-bearing definition. Do not bold technical terms inline.
- Numbers: percentages with `\%` (e.g. `75\%`), currency with `\$` (e.g. `\$254.4 billion`), no rounding without a reason.
- Acronyms: spell out on first use within a chapter with the acronym in parentheses, then use the bare acronym for the rest of the chapter.
- Lists: `\begin{itemize}` only for genuine enumerations (model branches, feature categories, step procedures). Do not break running prose into bullets for visual variety.
