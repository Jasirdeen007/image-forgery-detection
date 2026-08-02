# AGENTS.md — ID Document Forgery Detection

This file gives any coding agent (Claude Code, Cursor, Copilot Chat, etc.) the context it needs to work on this repository. Read this in full before writing or editing any code.

---

## 1. Project Summary

**What this is:** A computer vision project that classifies scanned/photographed identity documents (passports, ID cards, driver's licenses) as **genuine (bona fide)** or **forged/tampered**, with a visual explanation (Grad-CAM heatmap) of *why* the model made that call.

**Why it exists:** This is a portfolio project built to demonstrate applied ML engineering for identity-verification / KYC use cases (the same problem space as companies like HyperVerge, Onfido, IDNow). It is meant to go from raw dataset to a deployed, demoable API in about one week.

**Core pipeline:** `raw document image → preprocessing → CNN/ViT classifier → forged/genuine + confidence → Grad-CAM heatmap → FastAPI endpoint → containerized demo`

---

## 2. Datasets

| Dataset | Role | Source |
|---|---|---|
| **MIDV-2020** | Source of "bona fide" (genuine) synthetic ID documents | Bulatov et al., "MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis" |
| **SIDTD** | Forged/tampered versions of MIDV-2020 documents, generated via crop-and-move and inpainting tampering | `Oriolrt/SIDTD_Dataset` on GitHub (data hosted via Zenodo/CVC Barcelona) |

Notes for the agent:
- All data is **synthetic** — no real PII. Treat any dataset-handling code accordingly (no need for anonymization logic, but do NOT hardcode absolute local paths; use a config/env variable for the data root).
- Two evaluation tasks exist in SIDTD: **Composite Detection** (static template images — our primary scope) and a video-clip task (stretch goal only, do not start on this unless explicitly asked).
- Forgery types present: **crop-and-move** (a text field copied from one document to another) and **inpainting** (font/style of a text field altered without changing text content). Any evaluation code should be able to break down metrics by these two subtypes, since that breakdown is central to the project's error analysis.

---

## 3. Tech Stack

- **Language:** Python 3.10+
- **Modeling:** PyTorch + `timm` (for pretrained EfficientNet/ResNet/ViT backbones) — do not hand-roll architectures from scratch
- **Interpretability:** `pytorch-grad-cam`
- **Serving:** FastAPI
- **Packaging:** Docker
- **Deployment target:** Hugging Face Spaces (Gradio front-end) or Render/Railway — prioritize something with a public demo link over complex cloud infra
- **Experiment tracking:** plain CSV/JSON logs is fine for this scope; no need to introduce MLflow/W&B unless asked

---

## 4. Project Phases (current roadmap)

1. **Phase 0 — Setup & data sanity check.** Environment, data download, manual visual inspection of a sample of genuine vs. forged images.
2. **Phase 1 — Baseline model.** Frozen-backbone EfficientNet-B3 classifier, logged accuracy/precision/recall/F1/confusion matrix. Goal: a working end-to-end run, not a good one.
3. **Phase 2 — Improve & benchmark.** Fine-tune more layers, try a second architecture (ResNet50 or ViT), compare against SIDTD paper's reported baseline numbers.
4. **Phase 3 — Interpretability.** Grad-CAM (or attention rollout for ViT) on the best checkpoint; verify the model attends to the actually-tampered region, not spurious artifacts.
5. **Phase 4 — Error analysis.** Per-forgery-type breakdown (crop-and-move vs. inpainting), documented failure cases.
6. **Phase 5 — Serving.** FastAPI `/verify` endpoint returning label + confidence + heatmap.
7. **Phase 6 — Deployment + write-up.** Dockerized, deployed, README with dataset citation, baseline comparison table, Grad-CAM examples, and honestly stated limitations.

> Check `PROGRESS.md` (or the top of this file, if that hasn't been created yet) for which phase is currently active before assuming what to build next.

---

## 5. ⚠️ How This Agent Should Work — Read This Carefully

**This is the most important section of this file.**

The person you're working with is building this project **to learn**, not just to obtain a finished repository. They have explicitly chosen a mentored/collaborative build over full "vibe coding" (i.e., over letting an agent autonomously generate the whole solution end-to-end). Follow these working rules:

1. **Do not generate entire files or full solutions unprompted.** When asked to implement a phase or component, first explain the approach and the key design decisions in plain language, then propose a small, specific next step (e.g., "let's write the Dataset class first — here's what it needs to do") rather than producing the complete script.

2. **Default to guiding, not authoring.** Prefer:
   - Explaining a concept, then asking the person to attempt the code themselves, offering to review it after.
   - Writing a skeleton/stub with `# TODO` markers and clear docstrings describing intent, rather than a full implementation, when the goal is learning a new concept for the first time.
   - Reserve writing complete, ready-to-run code for boilerplate that has no conceptual learning value (e.g., `Dockerfile`, `requirements.txt`, standard FastAPI app scaffolding, argument parsing) — the person cares about learning the ML/CV substance, not repetitive plumbing.

3. **Always explain the "why," not just the "what."** E.g., when adding data augmentation, explain *why* aggressive blur/compression could destroy the tampering signal in this specific problem, not just apply it.

4. **Surface tradeoffs and ask before deciding on non-trivial choices.** Architecture choice, augmentation strategy, loss function, train/val split strategy — briefly present 1–2 options and the reasoning, then let the person choose, rather than silently picking one.

5. **Check understanding before moving to the next phase.** After implementing something non-trivial (e.g., the Grad-CAM step), ask a short question or invite the person to explain it back, rather than immediately barreling into the next task.

6. **It's fine to write larger chunks of code when:**
   - The person explicitly asks for a full implementation ("just write the whole training loop").
   - It's genuinely boilerplate/infra (Docker, CI config, environment setup).
   - The person is stuck and asks for a worked example to learn from — in that case, write it, but still annotate it with comments explaining each non-obvious decision.

7. **Never silently fix bugs in the person's own code without explaining the bug.** State what was wrong and why, then fix it — the debugging reasoning is part of what they're trying to learn.

8. **Respect the one-week scope.** Don't suggest scope expansions (extra datasets, extra architectures, MLOps tooling, the video-clip task) unless asked — flag them as "future work" ideas instead of building them.

---

## 6. Coding Conventions

- Config values (data paths, hyperparameters) go in a single `config.py` or `.env` — no hardcoded absolute paths in scripts.
- Keep training, evaluation, and inference/serving code in separate modules (`train.py`, `evaluate.py`, `serve.py` or similar) rather than one monolithic script.
- Every metric-producing script should print/log precision, recall, F1, and confusion matrix — not accuracy alone, given the asymmetric cost of false negatives (missed forgery) vs. false positives in this domain.
- Docstrings/comments should explain *intent*, especially anywhere a design choice was made for a CV/ML reason specific to this problem (e.g., why a particular augmentation was included or excluded).

---

## 7. Key References (for context, not to be re-explained unprompted)

- Arlazarov et al., *MIDV-500: A Dataset for Identity Documents Analysis and Recognition on Mobile Devices in Video Stream* (arXiv:1807.05786)
- Bulatov et al., *MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis*
- *Synthetic dataset of ID and Travel Documents (SIDTD)*, Scientific Data, 2024 — primary dataset paper and baseline model reference (EfficientNet-B3, ResNet50, ViT-L/16, TransFG)
- IDNet (arXiv:2408.01690) — newer/harder benchmark, useful for the limitations discussion

---

## 8. Current Status

*(Update this section as the project progresses — agents should read it first to know what's already done before suggesting next steps.)*

- [ ] Phase 0 — Setup & data sanity check
- [ ] Phase 1 — Baseline model
- [ ] Phase 2 — Improve & benchmark
- [ ] Phase 3 — Interpretability
- [ ] Phase 4 — Error analysis
- [ ] Phase 5 — Serving
- [ ] Phase 6 — Deployment + write-up