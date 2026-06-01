# THS-ST1 — Data Acquisition Ethics Writeup (Chrome Extension)

_Generated: 2026-06-01_

Self-contained brief to paste into Claude inside `~/Code/GitHub/sfv-thesis`. Single focused task: write up the **Data Acquisition** section so that the ethics of the TikTok Analytics Exporter Chrome extension are anchored on the user's **right to access their own information** as established in TikTok's own Privacy Policy.

---

## Context for Claude

I'm Aki, working on the thesis **"To Predict Is To Believe: Integrating Content, Context, and Creator Features For Pre-Publication Short-Form Video Engagement Prediction"** (THS-ST1, DLSU AY2526-T3).

- **Topic is locked.** No more pivoting — only refinement.
- **Dataset:** Analytics CSV + MP4 donated by 30–50 Filipino micro-creators (1k–100k followers), collected via the TikTok Analytics Exporter Chrome extension (already built).
- **The Chrome extension** is the data acquisition tool. It runs locally in the creator's browser, authenticates with the creator's own TikTok account, and exports the analytics data that TikTok Studio already shows the creator. No back-end. No remote storage. No bypass of any access control.
- **What this writeup is for:** the Data Acquisition section of the Methodology chapter, plus the ethics justification carried into the Ethics Review form.
- **Adviser already approved this framing on 2026-05-26.** This task is execution, not direction-setting.

---

## The Anchor (read this first)

**Primary source to cite:** TikTok Privacy Policy (Rest of World) — <https://www.tiktok.com/legal/page/row/privacy-policy/en>

**Specific section to lean on:** *"Your Rights and Choices"* — particularly the clause that gives users the right to **access** their own information.

The argument we're building:

> The Chrome extension does not collect data *about* TikTok users. It facilitates the participating creator's exercise of a right that TikTok itself documents and grants: the right to access their own information. The data flow is creator → creator. The extension is a tool that automates a manual export the creator could perform themselves through the TikTok Studio UI.

This is the load-bearing claim. Everything else in the section supports it.

---

## What Claude Should Produce

A **Data Acquisition** subsection inside the Methodology chapter (and a parallel passage in the ethics documentation / Ethics Review form). It should have these parts, in this order:

### 1. Data source and ownership
- Data is sourced from TikTok Studio analytics belonging to the participating creator.
- The creator is the **data subject and the data controller** of their own analytics for the purpose of this study.
- No third-party data is collected. No follower-level PII is scraped. Only the creator's own video-level analytics + the creator's own MP4 files.

### 2. Why creators can lawfully share this data with the study
Cite TikTok's Privacy Policy "Your Rights and Choices" section and pull the **right to access** clause verbatim (short quote, with section heading + URL). Then explain:
- TikTok grants users the right to access their own information.
- A creator exercising that right and choosing to donate the accessed data to a research study is a data-portability act, not a scraping act.
- The study is the *recipient* of donated data, not the *acquirer* of TikTok's data.

### 3. How the Chrome extension fits this framing
The five-pillar framework, **rewritten as direct support for the access-right anchor** rather than as standalone claims:

- **Automation of an existing right.** The extension only automates the export the creator can already perform inside TikTok Studio. It does not unlock data, does not bypass any access gate, and does not request anything TikTok would not show the logged-in creator.
- **User agency at every step.** Install → authenticate with the creator's own TikTok account → manually trigger the export. Each step is an affirmative action by the creator. Multi-step gating constitutes explicit consent.
- **Right to data portability.** Frame the export as a technical realisation of the creator's portability right (TikTok Privacy Policy "Your Rights and Choices"; aligned in principle with GDPR Art. 20 and the PH Data Privacy Act §16(f)).
- **Privacy by design — local-only processing.** All extension code runs in the creator's browser. No remote endpoint is contacted by the extension. The exported CSV is written to the creator's local disk. Transfer to the study only happens via a separate, explicit consent-gated step.
- **No additional inference about non-consenting parties.** The exported data covers the creator's own videos and the aggregate analytics TikTok exposes to them. No viewer-level PII, no follower identities, no third-party content.

### 4. Data flow diagram (textual; ask if Claude should draw it)
Creator's TikTok Studio (TikTok servers) → creator's authenticated browser session → Chrome extension running locally → CSV on creator's disk → (separate consent step) → study's secure storage.

Each arrow is a step the creator must affirmatively take.

**Pipeline numbering (locks in the Ch 4.1.1 rewrite):** Step 1 Consent → **Step 2 Extension Export (replaces TikTok's built-in "Download your data" export)** → Step 3 Submission → Step 4 Anonymization → Step 5 Verification. The existing Ch 4.1.1 text that names TikTok's official analytics archive as Step 2 must be rewritten so the Chrome extension is the canonical acquisition path. This is the locked, adviser-approved (2026-05-26) framing.

### 5. What the extension does **not** do
Explicit non-goals — important for the Ethics Review form:
- Does not scrape other creators' content or analytics.
- Does not collect viewer-level data.
- Does not store credentials.
- Does not upload anywhere on its own — no telemetry, no analytics, no remote logging.
- Does not run without the creator manually triggering it.

### 6. Compliance mapping
A short table mapping each pillar to the legal/policy basis:

| Concern                       | Basis                                              |
| ----------------------------- | -------------------------------------------------- |
| Right to access own data      | TikTok Privacy Policy, "Your Rights and Choices"   |
| Right to data portability     | GDPR Art. 20; PH Data Privacy Act §16(f)           |
| Consent for donation to study | DLSU-IRB informed consent form (separate document) |
| Data minimisation             | Only creator-owned analytics + creator-owned MP4   |
| Local-only processing         | Privacy-by-design (Art. 25 GDPR analogue)          |

### 7. Open items to flag (do not invent answers)
Have Claude flag these for me rather than guess:
- Exact section heading + clause wording from the TikTok Privacy Policy (need to pull verbatim from the live URL).
- Whether the policy version we cite should be pinned to a specific access date.
- Whether the methodology section should reproduce the data flow as a figure or keep it textual.

---

## Source Material Claude Should Use

In the thesis repo:
- Existing `thesis/chapter_2.tex` (RRL — for cross-reference style and citation conventions)
- Existing `outputs/revised_ethics.txt` if present (prior ethics framing to merge with)
- The Chrome extension source — to ground the "what it does / does not do" claims in actual code, not vibes

From outside:
- <https://www.tiktok.com/legal/page/row/privacy-policy/en> — primary anchor. Pull the "Your Rights and Choices" section verbatim for the access-right quote.

---

## Out of Scope for This Task

Not in this session — handle separately:
- Participant criteria (DE-2)
- Informed consent form draft (DE-3)
- RRL micro-influencer category expansion
- Baseline reproduction (MTR-1)
- Prototype chapter (Chapter 5) — separate writeup task

---

## Format Conventions

- LaTeX/Markdown per existing chapter format — match what's already there.
- Bibliography entries: append to existing `.bib` with consistent citation keys. TikTok Privacy Policy as a `@misc` entry with an access date.
- Additive edits only — don't restructure existing methodology unless I confirm.
- Flag anywhere the wording leans on a specific TikTok policy clause that needs verbatim verification.
