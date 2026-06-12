# Ethics Forms Review
Reviewed: 2026-06-09
Files: General_Research_Ethics_Checklist.pdf, RESEARCH ETHICS REVIEW FORM.pdf, Checklist A - Human Participants.pdf, Checklist G - Internet.pdf

---

## Critical Issues

### 1. Checklist G Q1 — "Is data publicly available?" answered YES
**Wrong.** TikTok analytics are private; only accessible to the account owner. The correct answer is NO.  
The "If NO" requirement is a letter of support from the website/server owner — but our situation is that the creators themselves are the data custodians and their signed consent forms serve this function. That argument needs to be made explicitly in the form or an attachment.

### 2. Checklist G Q6 — "Will data be collected using an automated system?" answered NO
**Wrong.** The Chrome extension automates analytics extraction from the creator's TikTok dashboard. This should be YES, with a description of the extension as the automation tool.  
This contradicts the locked decision (approved 2026-05-26) that the Chrome extension is Step 2 of the pipeline.

### 3. Review Form Brief Procedure Step 2 — still references "Download your data" archive
The form says creators "generate an official 'Download your data' archive from their TikTok settings." The approved pipeline uses the Chrome extension instead. This needs to be updated to reflect the actual data acquisition method.

### 4. Review Form + Checklist A — submission method still says "secure, encrypted cloud portal"
Both documents say creators upload files to an "encrypted cloud portal." This was updated to a Google Form in the methodology chapter (§4.1). The forms are now inconsistent with the manuscript.

### 5. Checklist G Q9 — "Will data be made available for future research?" answered NO
Contradicts Checklist A's data retention policy, which explicitly states anonymized feature sets and model weights will be kept for **three years** "for reproducibility and future comparative research." This is functionally making data available for future research. Q9 should be YES with appropriate caveats about anonymization.

---

## Minor Issues

### 6. Checklist A — adviser name missing "PhD"
General Checklist lists "Briane Paul V. Samson, PhD"; Checklist A lists "Briane Paul V. Samson" (no credential). Should be consistent.

### 7. Checklist A — Esleta name missing comma
General Checklist: "Esleta, Joshua James B."  
Checklist A: "Esleta Joshua James B."  
All four names should follow the same Last, First Middle I. format.

### 8. Checklist G — Audio not checked under data type
Video/Film and Text are checked, but Audio is unchecked. MP4 files contain audio tracks, and VideoLLaMA2 explicitly performs auditory analysis. Audio should be checked.

---

## Duplicate Files (housekeeping)
There are lowercase copies of two PDFs in the same directory:
- `checklist-A.pdf` (alongside `Checklist A - Human Participants.pdf`)
- `general_checklist.pdf` (alongside `General_Research_Ethics_Checklist.pdf`)

Confirm whether these are identical copies or different versions; delete whichever is stale.

---

## Summary Table

| # | Location | Severity | Issue |
|---|----------|----------|-------|
| 1 | Checklist G Q1 | Critical | Data marked "publicly available" — incorrect for private TikTok analytics |
| 2 | Checklist G Q6 | Critical | No automation system declared — Chrome extension not disclosed |
| 3 | Review Form Step 2 | Critical | "Download your data" procedure — should reference Chrome extension |
| 4 | Review Form + Checklist A | Major | Submission method says "encrypted cloud portal" — manuscript says Google Form |
| 5 | Checklist G Q9 | Major | Future research reuse marked NO — contradicts 3-year retention policy in Checklist A |
| 6 | Checklist A | Minor | Adviser missing PhD credential |
| 7 | Checklist A | Minor | Esleta name missing comma |
| 8 | Checklist G | Minor | Audio data type not checked despite auditory analysis in pipeline |
