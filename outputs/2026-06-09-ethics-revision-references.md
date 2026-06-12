# Ethics Forms — Reference Document Column Corrections
Date: 2026-06-09  
Scope: Research / Project Ethics Review Form only.  
The "Reference Document" column uses the old Chapter 3 section numbering (3.1.x). The methodology chapter is now Chapter 4. Update every row in that column as follows.

---

## Correction table

| Row | Current (wrong) | Correct |
|-----|-----------------|---------|
| Target participants | Section 3.1.1: Data Acquisition and Collection Procedure, pg 14 | Section 4.1.1: Data Acquisition and Collection Procedure, pg [verify] |
| Data to be used / collected | Section 3.1.2: Dataset Variables and Metrics (Table 3.2), pg 15 | Section 4.1.2: Dataset Variables and Metrics (Table 4.1), pg [verify] |
| Brief procedure | Section 3.1.1: Table 3.1 (Step-by-Step Data Acquisition), pg 14 | Section 4.1.1: Figure 4.1 (Step-by-Step Data Acquisition), pg [verify] |
| Potential risks | Section 3.1.3: Ethical Safeguards and Limitations, pg 16 | Section 4.1.3: Ethical Safeguards and Limitations, pg [verify] |
| Applicable Terms of Use | Section 3.1.3: Ethical Safeguards and Limitations, pg 16 | Section 4.1.3: Ethical Safeguards and Limitations, pg [verify] |
| Steps to safeguard | Section 3.1.3: De-identification and Storage Protocols, pg 16 | Section 4.1.3: Ethical Safeguards and Limitations, pg [verify] |

---

## Notes

- **"Brief procedure" row:** the step-by-step acquisition flow is now a figure in the LaTeX (`\ref{fig:data_acqui}`), not a table. Change "Table 3.1" → "Figure 4.1" (verify the figure number in the compiled PDF).
- **"Data to be used / collected" row:** the feature list is `\ref{tab:feature_list}` in §4.1.2. Change "Table 3.2" → "Table 4.1" (verify against compiled PDF).
- **"Steps to safeguard" row:** the subsection heading in the current draft is "Ethical Safeguards and Limitations" — the old sub-label "De-identification and Storage Protocols" no longer appears as a separate heading. Update the label to match the current heading.
- **All page numbers** ("pg 14", "pg 15", "pg 16") are from the old draft. Replace every `[verify]` placeholder with the correct page number from the current compiled manuscript PDF before submitting.
