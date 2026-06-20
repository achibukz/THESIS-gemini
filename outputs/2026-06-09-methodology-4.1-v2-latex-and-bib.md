# §4.1 Building the Dataset — LaTeX Paste-Ready (v2)

_Date: 2026-06-09. Updated from v1 (`outputs/2026-06-09-methodology-4.1-latex-and-bib.md`): creators no longer donate raw MP4 files; researchers retrieve videos from TikTok's public platform using video IDs in the analytics CSV. Creators are required by the informed consent form to set their videos to public with downloads enabled. Pipeline expanded from five to six steps._

---

## 1. LaTeX section block

```latex
\section{Building the Dataset}
\label{sec:dataset}

The construction of the dataset addresses two gaps in existing short-form video (SFV) benchmarks. Content-only datasets such as SnapUGC lack the creator-specific and contextual metadata required for cold-start prediction \cite{li2024delving}. Automated scraping, the alternative most often used to enlarge a benchmark, produces inconsistent labels and cannot reach the backend retention metrics that platforms expose only to verified account holders \cite{li2025vquala}. \hl{This study works around both limits through a data donation approach in which Filipino micro-creators voluntarily submit their first-party analytics.} The donation route gives the study verified ground-truth labels and avoids any access pattern that would conflict with platform terms.

\subsection{Data Acquisition and Collection Procedure}
\label{subsec:data_acq}

Target participants are active TikTok content creators who maintain a professional or business account. Recruitment proceeds through a distributed Google Form link, with an ideal minimum of 30 to 50 Filipino micro-creators. \ref{fig:data_acqui}.}

\textbf{Step 1: Informed Consent.} The creator reviews the research objectives, the data handling policy, and the likeness clause covering donated footage before providing a digital signature. The consent form is attached as Appendix A and is signed before any donation step proceeds.

\textbf{Step 2: Chrome Extension Export.} The participant installs the TikTok Analytics Exporter Chrome extension and authenticates through their own TikTok account. The extension reads the analytics that TikTok Studio already exposes to the logged-in creator and writes them to a CSV on the participant's local disk. It runs entirely inside the creator's authenticated browser session: no remote endpoint is contacted by the extension, no credentials are stored, and no viewer-level data is collected. Each participant exercises the access right granted to users under TikTok's Privacy Policy ``Your Rights and Choices'' clause \cite{tiktok_privacy_2026}; the extension only automates the export the creator is already entitled to perform manually.

\textbf{Step 3: Submission.} \hl{The creator uploads the analytics CSV through a Google Form operated by the study.} Before the upload completes, the creator must affirm a final consent statement on the form acknowledging the transfer.

\hl{\textbf{Step 4: Video Download.} The research team downloads each video from TikTok's public platform using the video IDs in the submitted CSV. This step is contingent on the creator having fulfilled the informed consent requirement to set their videos to public with downloads enabled before submission.}

\hl{\textbf{Step 5: Anonymization.}} Researchers strip identifying fields from the analytics CSV (account handle, profile name, contact email) and replace the creator handle with a randomized alphanumeric ID. The mapping between the original handle and the anonymized ID is held separately from the working dataset, under the safeguards stated later in this section.

\hl{\textbf{Step 6: Verification.} The post timestamps in the analytics export are cross-referenced with each retrieved MP4 to confirm that every row binds to the correct video file. The verification step pairs post date, post time, and reported duration to confirm the match.}

\subsection{Dataset Variables and Metrics}
\label{subsec:dataset_vars}

The dataset integrates three feature layers that together frame the prediction problem. The Multimodal Content Layer holds the raw MP4 video, caption text, and hashtags. The Creator-Related Layer holds follower count and account age. The Contextual Layer holds the precise posting timestamp, separated into hour-of-day and day-of-week. Visual and acoustic signals are extracted automatically from each MP4 by the Large Multimodal Model (LMM) ensemble introduced in the modeling section that follows; no manual encoding is required at this stage.

The analytics CSV produced by the Chrome extension export step is the canonical source for every non-content feature. Each row corresponds to one of the participating creator's videos and carries the post timestamp, video duration, view count, like count, share count, comment count, total watch time, average watch time (AWT), and the precomputed Normalized Average Watch Percentage (NAWP) value. The two project-target metrics are derived from this row directly: NAWP is read from the export, and Engagement Continuation Rate (ECR) is recovered from the retention curve TikTok exposes alongside the watch-time fields. Table \ref{tab:feature_list} lists the initial feature set; the full column schema is documented in \texttt{data\_specs/extension-csv-schema.md} and reflects the live column order written by the extension.

While these are the primary variables for the initial model, the feature set is not closed. \hl{Some analytics exports expose additional fields, including audience-region breakdowns and traffic-source splits, that may support later experiments.} The methodology allows the inclusion of such signals after the baseline is reproduced and only when the field is available consistently across donors.

\subsection{Ethical Safeguards and Limitations}
\label{subsec:ethics_limits}

Ethical integrity is maintained through the procedures stated in the informed consent form (Appendix A) and the Ethics Review Form.

\hl{\textbf{Public availability and downloads.} The informed consent form (Appendix A) includes a requirement that participants enable public visibility and TikTok's download permission on their posted videos before proceeding to the submission step. These settings must remain active through the completion of the video retrieval step.}

\textbf{Identity and likeness.} Creator handles are replaced with randomized identifiers at the anonymization step, and the original-to-anonymized mapping is held separately from the working dataset. \hl{The informed consent form covers the use of the creator's likeness as part of the retrieved frames; the proposed model performs engagement prediction and does not perform face recognition or other biometric inference. Secondary subjects appearing incidentally in the background of retrieved footage are covered by the same clause, since the creator confirms standing to authorize retrieval of the footage at the time of consent.}

\textbf{Data retention.} Non-anonymized donor information is held for a maximum of two years from the first publication of any result derived from the dataset or from project completion, whichever is later. Anonymized data is held for a maximum of three years on the same basis. After each retention window, the corresponding files are deleted, and the anonymization mapping table is deleted at the close of the two-year window so that no path remains from anonymized rows back to the original creator.

\textbf{Withdrawal.} A participant may withdraw their donation at any time before the final model aggregation by contacting the proponent email listed in the consent form. After identity is verified through the original donation record, the participant's data is deleted across the ingestion portal, the working dataset, and the long-term archive.

\hl{\textbf{TikTok platform compliance.} The analytics export step is initiated by the creator, performed inside the creator's own authenticated browser session, and limited to data the creator is already entitled to access \cite{tiktok_privacy_2026, tiktok_tos_2026}. The extension does not bypass any access control, does not store credentials, and does not contact any remote endpoint. Video retrieval in Step 4 operates on publicly accessible content that the creator has explicitly made downloadable, in compliance with TikTok's Terms of Service \cite{tiktok_tos_2026}. The study does not scrape, access non-public data, or bypass platform access controls at any stage.}
```

---

## 2. New BibTeX entries

Append the two `tiktok_*` entries to `myreferences.bib`. The two `li...` keys cited above (`li2024delving`, `li2025vquala`) are already present in the bib file — no action needed for those.

```bibtex
@misc{tiktok_privacy_2026,
  author    = {{TikTok}},
  title     = {Privacy Policy},
  year      = {2024},
  url       = {https://www.tiktok.com/legal/page/row/privacy-policy/en},
  urldate   = {2026-06-09},
  note      = {Section: ``Your Rights and Choices''}
}

@misc{tiktok_tos_2026,
  author    = {{TikTok}},
  title     = {Terms of Service},
  year      = {2024},
  url       = {https://www.tiktok.com/legal/page/row/terms-of-service/en},
  urldate   = {2026-06-09}
}
```

**Before submission, confirm:**

- The `year` field reflects the last-updated date stamped at the top of the live TikTok policy page (placeholder `2024` above — replace with the actual revision year shown on the policy).
- `urldate` is the date the policy was last consulted for the manuscript (set to today; update if you re-verify closer to defense).
- Citation style: if `myreferences.bib` elsewhere uses `howpublished = {\url{...}}` instead of the `url` + `urldate` fields, switch to that form to keep the file uniform.

---

## 3. What still needs LaTeX-side wiring (not auto-generated here)

1. **Figure update.** `figures/chap3/Data Acquisition.png` is referenced by `\ref{fig:data_acqui}`. The pipeline is now six steps (was five): add Step 4 ``Video Download'' between Submission and Anonymization, and renumber the old Steps 4–5 to 5–6. Re-render and re-save in place. The `\begin{figure} ... \end{figure}` block in `chapter_4.tex` does not need to change as long as the label stays `fig:data_acqui`.
2. **Table.** `\ref{tab:feature_list}` keeps the existing four-row summary (Raw MP4 File / Follower Count / Creator Age / Posting Time). No table edit required here.
3. **Schema file.** The prose names `data_specs/extension-csv-schema.md`. If that file does not yet exist in the repo, generate it from the extension source before manuscript freeze so the reference resolves.
4. **Appendix slot.** Prose refers to ``Appendix A'' for the consent form — confirm `appendix_A.tex` is the locked slot (not `appendix_B.tex`).
5. **Informed consent form.** Add a clause requiring participants to enable public visibility and TikTok's download permission on their posted videos, and to maintain these settings through the video retrieval step (Step 4).
