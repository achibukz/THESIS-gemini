# THS-ST1 — Ethics Review Form + General Research Ethics Checklist

_Generated: 2026-06-01_

Self-contained brief to paste into Claude inside `~/Code/GitHub/sfv-thesis`. Focused task: produce a complete DLSU **Ethics Review Form** + **General Research Ethics Checklist** package for THSST1, plus the **Informed Consent Form** that ships as an appendix.

This plan is a sibling of `docs/2026-06-01-THSST1-thesis-repo-update-plan.md` (the Data Acquisition writeup). Read that plan first — its locked framing is the source of truth for the ethics narrative.

---

## Context for Claude

- Topic locked. Adviser-approved framing (2026-05-26) is that the Chrome extension is a creator-side tool that automates the creator's right to access their own information under TikTok's Privacy Policy.
- This deliverable is required before defense endorsement (2026-07-13). The forms must be ready well before then so any IRB-style revision is feasible inside the July 6 manuscript freeze.
- The forms live in `thesis/<MM/DD/YY>/` alongside the rest of the manuscript:
  - `RESEARCH ETHICS REVIEW FORM.pdf`
  - `General_Research_Ethics_Checklist.pdf`
  - `Checklist A - Human Participants.pdf`
  - `Checklist G - Internet.pdf`
  - `general_checklist.pdf`, `checklist-A.pdf`, `clearance.pdf`
- The Informed Consent Form will be appended via the existing `appendix_A.tex` / `appendix_B.tex` LaTeX scaffolds.

---

## The Anchor (read this first)

The ethical justification across all three artefacts is one claim:

> The study **does not collect data from TikTok**. The study **receives data donated by creators**, who exercise their right (granted by TikTok's own Privacy Policy, "Your Rights and Choices") to access their own information. The Chrome extension is the technical tool that automates this access on the creator's local machine. The model trained downstream does not perform face recognition or other biometric inference — likeness appears only as an incidental part of the donated video frames.

Every section of the forms should resolve back to this anchor when challenged.

---

## What Claude Should Produce

### A. Ethics Review Form (fill-out spec)

For each section of `RESEARCH ETHICS REVIEW FORM.pdf`, produce a Markdown answer sheet at `outputs/ethics-review-form-answers.md` containing the exact text to type into the PDF. Cover:

1. **Study identification** — title, proponents, adviser, term (AY2526-T3), institution.
2. **Research objectives** — short summary lifted from Ch 1 §1.2, no scope drift.
3. **Methods overview** — point to Ch 4.1 (Building the Dataset) and Ch 4.2 (Enhancing the Model), with the locked five-step pipeline (Consent → Extension Export → Submission → Anonymization → Verification).
4. **Participants** — Filipino micro-creators, 1k–100k followers, voluntary donation, recruited via distributed Google Form link. Note: participant *criteria* refinement is deferred to a separate task (DE-2); use the existing Ch 4.1.1 wording for now.
5. **Data collected** — analytics CSV (creator's own video-level metrics) + raw MP4 files (creator's own posts). No viewer-level PII, no follower identities, no third-party content.
6. **Data handling** — local-only extension processing → encrypted submission portal → anonymized storage. Use the data flow diagram from the data-acquisition plan.
7. **Risks** — minimal-risk study. Document the absence of intervention, the absence of facial/biometric inference, and the absence of viewer-level data collection.
8. **Data retention** — explicit:
   - **Non-anonymized PII: max 2 years** from first publication or project completion.
   - **Anonymized data: max 3 years** from first publication or project completion.
9. **Withdrawal** — participants may withdraw and request deletion of donated data any time before final model aggregation.
10. **Consent process** — reference the Informed Consent Form (Appendix A).

### B. General Research Ethics Checklist + Checklist A + Checklist G

For each item, produce a one-line answer + a one-sentence justification, written to `outputs/ethics-checklists-answers.md` with headings matching the PDF section titles. Specifically resolve:

- **Checklist A (Human Participants)** — required because creators are recruited individuals donating data. Justify "minimal risk" classification.
- **Checklist G (Internet / Online Data)** — required because the data originates from an online platform (TikTok). The "online" angle is the *creator accessing their own analytics*, not the study scraping the platform. Cite the same TikTok Privacy Policy clause used in the Data Acquisition writeup.
- **General Research Ethics Checklist** — cover compliance with DLSU's general research-ethics principles, anonymization, retention, consent, withdrawal.

### C. Informed Consent Form (LaTeX appendix)

Produce `appendix_A.tex` (or whichever appendix slot the adviser prefers) with the full consent text. Required clauses:

1. **Study identification + contact info** — proponents, adviser, institution, contact email.
2. **Purpose of the study** — short, non-technical summary.
3. **What participation involves** — install the Chrome extension, authenticate with the participant's own TikTok account, trigger the analytics export, upload the resulting CSV + their donated MP4 files via the encrypted portal.
4. **Data collected** — itemized list (analytics CSV fields + raw MP4 files). Stress that **no viewer-level data, no follower identities, and no third-party data** are collected.
5. **Likeness clause (new, replaces the old face-blurring policy)** — explicit statement that the creator's likeness may appear in donated video frames and may be used as part of model training data. The model does not perform face recognition or other biometric inference, and likeness is not separately analysed. Secondary subjects (people incidentally captured in donated videos) are covered by the same clause — the creator confirms they have the standing to donate the footage.
6. **Storage and security** — encrypted storage, anonymized IDs, segregated network location.
7. **Retention** — 2 years for non-anonymized PII, 3 years for anonymized data, dated from first publication or project completion.
8. **Right to withdraw** — participant may withdraw and request deletion of donated data any time before final model aggregation.
9. **Compensation / risk statement** — minimal-risk, no compensation, voluntary.
10. **Signatures + date**.

### D. TikTok ToS / Privacy Policy citation block

Fold in **DT-1 (Review TikTok Studio Terms of Service)**. Produce a short paragraph that:

- Names the live URLs for the TikTok Terms of Service and Privacy Policy.
- Quotes the "Your Rights and Choices" clause verbatim (subject to verification — flag the access date).
- States that the extension does not bypass any access control, does not store credentials, and does not contact any remote endpoint.
- States the conclusion: the extension is an **automation of an existing user right**, not a circumvention or scraping mechanism.

This block should be reusable in both the Ethics Review Form §3/§4 and the Methodology Ch 4.1.1 rewrite.

---

## Source Material Claude Should Use

In the thesis repo:
- `docs/2026-06-01-THSST1-thesis-repo-update-plan.md` — locked data-acquisition framing (parent document).
- `thesis/<latest-snapshot>/chapter_4.tex` — current Methodology text; cross-reference §4.1.1 and §4.1.3.
- `thesis/<latest-snapshot>/appendix_A.tex`, `appendix_B.tex` — existing appendix scaffolds.
- `thesis/<latest-snapshot>/RESEARCH ETHICS REVIEW FORM.pdf`, `General_Research_Ethics_Checklist.pdf`, `Checklist A - Human Participants.pdf`, `Checklist G - Internet.pdf` — source PDFs to mirror.
- `outputs/` — destination for fill-out answer sheets.
- Chrome extension source — grounds the "does not do X" claims in real code.

From outside:
- TikTok Privacy Policy (Rest of World): <https://www.tiktok.com/legal/page/row/privacy-policy/en>
- TikTok Terms of Service (Rest of World): <https://www.tiktok.com/legal/page/row/terms-of-service/en>

---

## Out of Scope for This Task

- **DE-2 (formal participant inclusion/exclusion criteria)** — adviser-deferred. Use the existing Ch 4.1.1 wording in the forms.
- **RRL micro-influencer expansion** — separate writeup task.
- **Chapter 5 (prototype chapter)** — not in scope for this term yet (do not write or plan).
- **Baseline reproduction (MTR-1)** — covered by `plan.md`.

---

## Format Conventions

- LaTeX for the consent form appendix; match the existing `appendix_A.tex` / `appendix_B.tex` style.
- Markdown for the answer sheets in `outputs/`.
- Bibliography: append a `@misc` entry per cited TikTok URL to `myreferences.bib` inside the latest dated snapshot under `thesis/`, with an `urldate` field for the access date.
- Additive edits only — do not restructure existing methodology or appendices without confirmation.
- Flag every place a TikTok policy clause is paraphrased rather than quoted verbatim — those need live-URL verification before submission.
