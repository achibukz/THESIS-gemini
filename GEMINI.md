### GEMINI.md

## Project Overview
**Thesis Title**: To Predict Is To Believe: Integrating Content, Context, and Creator Features For Pre-Publication Short-Form Video Engagement Prediction.  
**Core Objective**: Develop a creator-oriented engagement prediction model that integrates multimodal content signals with creator characteristics and contextual factors.

**Primary Metric Focus (Target Variables)**:
* **ECR (Engagement Continuation Rate)**: Probability of retaining a viewer past the initial five-second "hook".
* **NAWP (Normalized Average Watch Percentage)**: A duration-normalized measure of total viewer retention.

## Technical Architecture (Ensemble LMM)
The framework utilizes an ensemble of three Large Multimodal Models (LMMs):
* **VideoLLaMA2**: For joint spatiotemporal and auditory understanding.
* **Qwen2.5-VL**: For general visual-semantic reasoning.
* **InternVideo2**: For refined analysis of visual dynamics.
* **Feature Fusion**: High-dimensional content embeddings are concatenated with structured metadata including follower count, account age, and posting timestamps.

## Dataset Specifications
* **Target Participants**: Filipino Micro-creators (30 to 50 participants).
* **Data Source**: Voluntary data donation of raw MP4 files and official TikTok analytics exports.
* **Multimodal Content Layer**: Raw video files, captions, and hashtags.
* **Creator-Related Layer**: Follower counts and account age.
* **Contextual Layer**: Specific posting timestamps.

## Contextual Guidelines for Gemini
1. **Metric Priority**: Focus exclusively on ECR and NAWP; exclude traditional social metrics like raw view counts, likes, or shares.
2. **Implementation Constraints**: Use frozen LMM backbones as feature extractors due to computational resource limits.
3. **Benchmarking**: Compare performance against Sun et al. (2025) and Guan et al. (2025) models.
4. **Preprocessing**: Extract exactly eight keyframes using uniform sampling to analyze the initial video window.

## Project Timeline (2026-2027)
* **Building the Dataset**: May 2026 – August 2026.
* **Enhancing the Model**: August 2026 – December 2026.
* **Evaluating the Model**: January 2027 – April 2027.