# §4.1 Building the Dataset — Draft for Review

_Draft date: 2026-06-01. Generated via the `thesis-humanizer` skill in generate mode against `docs/2026-06-01-THSST1-methodology-refinement.md`, with locked framings from `docs/2026-06-01-THSST1-thesis-repo-update-plan.md` and `docs/2026-06-01-THSST1-ethics-review-form.md`. Markdown only — convert to LaTeX after revisions are settled._

Citations are written as `\cite{...}` for direct LaTeX paste-through. Keys flagged in the **Open Items** section at the bottom are not yet present in `myreferences.bib`.

---

## 4.1 Building the Dataset

The construction of the dataset addresses two gaps in existing short-form video (SFV) benchmarks. Content-only datasets such as SnapUGC lack the creator-specific and contextual metadata required for cold-start prediction \cite{li2024delving}. Automated scraping, the alternative most often used to enlarge a benchmark, produces inconsistent labels and cannot reach the backend retention metrics that platforms expose only to verified account holders \cite{li2025vquala}. This study works around both limits through a data donation approach in which Filipino micro-creators voluntarily submit their first-party analytics together with the raw MP4 files of their posts. The donation route gives the study verified ground-truth labels and avoids any access pattern that would conflict with platform terms.

> Figure reference (unchanged): `figures/chap3/Data Acquisition.png` — re-render needed so Step 2 reads ``Chrome Extension Export'' instead of TikTok's built-in archive. Flagged for the diagrams pass; not edited here.

### 4.1.1 Data Acquisition and Collection Procedure

Target participants are active TikTok content creators who maintain a professional or business account. Recruitment proceeds through a distributed Google Form link, with an ideal minimum of 30 to 50 Filipino micro-creators. After providing informed consent, each participant donates their entire available video history (raw MP4 files) together with the analytics export described below. This per-participant density allows the model to analyze long-term creator patterns alongside specific video performance. The collection procedure runs in five stages summarized in Figure \ref{fig:data_acqui}.

**Step 1: Informed Consent.** The creator reviews the research objectives, the data handling policy, and the likeness clause covering donated footage before providing a digital signature. The consent form is attached as Appendix A and is signed before any donation step proceeds.

**Step 2: Chrome Extension Export.** The participant installs the TikTok Analytics Exporter Chrome extension and authenticates through their own TikTok account. The extension reads the analytics that TikTok Studio already exposes to the logged-in creator and writes them to a CSV on the participant's local disk. It runs entirely inside the creator's authenticated browser session: no remote endpoint is contacted by the extension, no credentials are stored, and no viewer-level data is collected. Each participant exercises the access right granted to users under TikTok's Privacy Policy ``Your Rights and Choices'' clause \cite{tiktok_privacy_2026}; the extension only automates the export the creator is already entitled to perform manually.

**Step 3: Submission.** The creator uploads the analytics CSV and the donated MP4 files through a Google Form operated by the study. Before the upload completes, the creator must affirm a final consent statement on the form acknowledging the transfer.

**Step 4: Anonymization.** Researchers strip identifying fields from the analytics CSV (account handle, profile name, contact email) and replace the creator handle with a randomized alphanumeric ID. The mapping between the original handle and the anonymized ID is held separately from the working dataset, under the safeguards stated later in this section.

**Step 5: Verification.** The post timestamps in the analytics export are cross-referenced with each donated MP4 to confirm that every row binds to the correct video file. The verification step pairs post date, post time, and reported duration to resolve the caption-versus-video-ID misalignment risk encountered during extension development, in which filename collisions across re-uploaded captions could otherwise have produced an incorrect binding.

### 4.1.2 Dataset Variables and Metrics

The dataset integrates three feature layers that together frame the prediction problem. The Multimodal Content Layer holds the raw MP4 video, caption text, and hashtags. The Creator-Related Layer holds follower count and account age. The Contextual Layer holds the precise posting timestamp, separated into hour-of-day and day-of-week. Visual and acoustic signals are extracted automatically from each MP4 by the Large Multimodal Model (LMM) ensemble introduced in the modeling section that follows; no manual encoding is required at this stage.

The analytics CSV produced by the Chrome extension export step is the canonical source for every non-content feature. Each row corresponds to one of the participating creator's videos and carries the post timestamp, video duration, view count, like count, share count, comment count, total watch time, average watch time (AWT), and the precomputed Normalized Average Watch Percentage (NAWP) value. The two project-target metrics are derived from this row directly: NAWP is read from the export, and Engagement Continuation Rate (ECR) is recovered from the retention curve TikTok exposes alongside the watch-time fields. Table \ref{tab:feature_list} lists the initial feature set; the full column schema is documented in `data_specs/extension-csv-schema.md` and reflects the live column order written by the extension.

While these are the primary variables for the initial model, the feature set is not closed. Donated archives sometimes expose additional fields, including audience-region breakdowns and traffic-source splits, that may support later experiments. The methodology allows the inclusion of such signals after the baseline is reproduced and only when the field is available consistently across donors.

> Table reference (unchanged): `\label{tab:feature_list}` — the existing four-row table (Raw MP4 File / Follower Count / Creator Age / Posting Time) can stay as the high-level summary. The detailed schema lives in `data_specs/extension-csv-schema.md` rather than in the chapter table.

### 4.1.3 Ethical Safeguards and Limitations

Ethical integrity is maintained through the procedures stated in the informed consent form (Appendix A) and the Ethics Review Form.

**Identity and likeness.** Creator handles are replaced with randomized identifiers at the anonymization step, and the original-to-anonymized mapping is held separately from the working dataset. The informed consent form covers the use of the creator's likeness as part of the donated frames; the proposed model performs engagement prediction and does not perform face recognition or other biometric inference. Secondary subjects appearing incidentally in the background of donated footage are covered by the same clause, since the donating creator confirms standing to donate the footage at the time of consent.

**Data retention.** Non-anonymized donor information is held for a maximum of two years from the first publication of any result derived from the dataset or from project completion, whichever is later. Anonymized data is held for a maximum of three years on the same basis. After each retention window, the corresponding files are deleted, and the anonymization mapping table is deleted at the close of the two-year window so that no path remains from anonymized rows back to the original creator.

**Withdrawal.** A participant may withdraw their donation at any time before the final model aggregation by contacting the proponent email listed in the consent form. After identity is verified through the original donation record, the participant's data is deleted across the ingestion portal, the working dataset, and the long-term archive.

**TikTok platform compliance.** The data collection is initiated by the creator, performed inside the creator's own authenticated browser session, and limited to data the creator is already entitled to access \cite{tiktok_privacy_2026, tiktok_tos_2026}. The extension does not bypass any access control, does not store credentials, and does not contact any remote endpoint. The study is the recipient of donated data, not the acquirer of TikTok data.

---

## Changes summary (vs. the current `chapter_4.tex` §4.1)

1. **§4.1 intro** — tightened from three to one paragraph; replaced ``elegant variation'' framing (multiple synonyms for the same dataset gap) with a single direct claim.
2. **§4.1.1** — Step 2 rewritten to name the Chrome extension; added the privacy posture sentence; cited the TikTok ``Your Rights and Choices'' clause; expanded Step 5 to name the caption-vs-video-ID misalignment risk explicitly.
3. **§4.1.2** — added a paragraph describing the CSV-row format and naming the schema file; pulled NAWP / AWT acronym introductions into this section so the modeling section does not have to re-introduce them; left the high-level feature table in place but moved the detailed schema to `data_specs/extension-csv-schema.md`.
4. **§4.1.3** — full rewrite. Dropped the ``automated face-detection and blurring algorithms'' paragraph (banned framing). Four short paragraphs: identity and likeness (combined), retention windows with explicit 2-year / 3-year language, the withdrawal procedure, and TikTok platform compliance. Sample-size and voluntary-response limitations are moved to the Scope and Limitations section as requested. Storage/encryption regime is left for later — not yet locked.

## Open Items (do not invent answers)

1. **Citations not yet in `myreferences.bib`** — both used by the draft. Add as `@misc` entries with `urldate`:
   - `tiktok_privacy_2026` → <https://www.tiktok.com/legal/page/row/privacy-policy/en>
   - `tiktok_tos_2026` → <https://www.tiktok.com/legal/page/row/terms-of-service/en>
2. **Verbatim ``Your Rights and Choices'' clause.** The draft paraphrases. Before submission, pull the exact wording from the live policy URL and decide whether to quote verbatim in the chapter, in the consent form, or both.
3. **Figure re-render.** `figures/chap3/Data Acquisition.png` still shows Step 2 as TikTok's built-in export. The chapter prose has been updated; the figure has not. Flag for the diagrams pass.
4. **`data_specs/extension-csv-schema.md`.** The draft cites this file as the canonical column-list source. Confirm whether it already exists in the repo — if not, generating it from the extension source is a follow-up before manuscript freeze.
5. **Appendix slot.** The draft says ``Appendix A'' for the consent form, matching the ethics-form plan. Confirm with the adviser whether `appendix_A.tex` or `appendix_B.tex` is the locked slot.
6. **Storage and encryption regime.** Deliberately not described in this draft — the specifics have not been settled. Decide before manuscript freeze whether this stays out, lives in the Ethics Review Form only, or earns a short paragraph back in §3.1.3.

---

## Reviewer checklist before LaTeX conversion

- [ ] Pipeline numbering reads Consent → Extension Export → Submission → Anonymization → Verification across the chapter, the consent form, the Ethics Review Form, and the figure.
- [ ] Every `\cite{key}` resolves to a real `myreferences.bib` entry (add the two TikTok entries first).
- [ ] No reference to ``automated face-detection and blurring algorithms'' remains anywhere in §4.1.
- [ ] Retention windows match the Ethics Review Form (2 years non-anonymized, 3 years anonymized).
- [ ] All quotes use LaTeX backtick double-quotes once the file is converted to `.tex`. The markdown above uses `` `` ... '' `` so the conversion is mechanical.
