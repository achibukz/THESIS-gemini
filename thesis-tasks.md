# Thesis Task Tracker

Thesis: *To Predict Is To Believe: Integrating Content, Context, and Creator Features For Pre-Publication Short-Form Video Engagement Prediction*

---

## Dataset — Technical

### DT-1: Review TikTok Studio Terms of Service
Check whether the TikTok Studio usage agreement permits automated data collection via API interception, even for personal/research use. Identify any clauses that affect how collected data can be stored, shared, or published in a thesis. Document the relevant sections for the ethics chapter.

### DT-2: Frame the Chrome Extension as a Creator-Side Tool
Ensure the extension is positioned as a personal analytics aid for participating creators — not a research scraper. The data flows from the creator's own TikTok Studio to their own machine. Update README, popup copy, and any thesis references to reflect this framing. Stress that no data leaves the device without the creator's explicit action.

### DT-3: Privacy Handling — Blurring and Anonymisation
Define how sensitive creator data (follower counts, video performance, profile identifiers) will be handled in the thesis. Options include: blurring screenshots used in the document, anonymising creator IDs in the dataset, and aggregating statistics before publication. Decide on the approach and document it in the data handling section of Chapter 2.

---

## Dataset — Ethics

### DE-1: Frame Micro-Influencer Participation Correctly
Define "micro-creator" for this study (e.g., 1k–100k followers, primarily Filipino audience). Frame participation as voluntary data donation — creators share their own analytics and MP4 files, not scraped data. This framing affects both the ethics application and the recruitment pitch.

### DE-2: Define Participant Criteria
Write formal inclusion/exclusion criteria for the 30–50 Filipino micro-creator participants:
- Follower range
- Minimum number of videos to donate
- Content language (Filipino/English mix acceptable?)
- Account age or activity requirements
- Platform exclusivity or multi-platform creators allowed?

### DE-3: Draft Informed Consent Form
Create an informed consent document covering:
- What data is collected (analytics CSV + MP4 files)
- How data is stored and secured
- How it will be used (model training, thesis publication)
- Anonymisation / pseudonymisation commitments
- Right to withdraw and data deletion process
- Contact info for queries

Must be DLSU-IRB compliant. Coordinate with adviser before finalising.

---

## Model — Testing

### MT-1: Find Method to Incorporate Additional Features
Research how to extend LMM-EVQA's architecture to include creator and context features alongside the LMM embeddings. Candidate approaches:
- Late fusion: concatenate structured features with LMM embeddings before the regression head
- A separate MLP branch for structured features, merged before final prediction
- Feature importance analysis to decide which features are worth adding

This is a design task — don't implement until the baseline is reproduced and validated.

### MT-2: Research Minimum Specs for Model Training (do after MT-3)
Once the baseline is running on SnapUGC, benchmark GPU memory usage, training time per epoch, and disk I/O. Use these numbers to spec the cloud VM (RunPod / Lambda / Vast.ai). Document the minimum VRAM, CPU, and storage needed for a full thesis-scale training run.

---

## Model — Training (Ordered Steps)

### MTR-1: Reproduce Baseline on SnapUGC
Run LMM-EVQA's published pipeline on the SnapUGC-tiny subset. Verify that SROCC and PLCC scores match (or are within margin of) the numbers reported by Sun et al. 2025. This is the correctness gate — nothing downstream is valid until this passes.

### MTR-2: Smoke-Test on Non-SnapUGC Sample Data
Once the baseline is verified, run inference on a small sample of Filipino creator videos (from the donated MP4 set, even just 5–10 clips). The goal is not accuracy — it's confirming the pipeline handles different video characteristics (aspect ratio, duration, resolution, language) without crashing or producing degenerate outputs.

### MTR-3: Determine Minimum VM Specs for Full Training
Profile memory and compute from MTR-1 and MTR-2 results. Write a VM spec sheet covering: VRAM floor, recommended VRAM, CPU cores, RAM, storage (NVMe vs HDD), and estimated cost per training run on RunPod/Lambda/Vast.ai. This informs the cloud migration decision.
