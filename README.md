<h1 align="center">Peptide Diffusion</h1>
<p align="center"><em>De Novo Peptide Sequencing via Self-Conditioned Multinomial Diffusion — CSE 676, UB 2026</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-58A6FF?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Course-CSE%20676%20Deep%20Learning-7C3AED?style=flat-square" alt="Course">
  <img src="https://img.shields.io/github/languages/code-size/Gustav-Proxi/peptide-diffusion?style=flat-square&color=58A6FF" alt="Repo size">
</p>

---

## Overview

Database-free peptide identification from tandem mass spectra (MS/MS). Given a spectrum of fragment-ion peaks, the model generates the amino-acid sequence directly — no reference proteome required. This enables discovery in novel organisms and mixed microbial communities where canonical database search fails.

The V2 model surpasses the published 2025 SOTA (InstaNovo) by **+3.1 pp AA recall** and **+26.9 pp peptide accuracy** on the E. coli EV benchmark at 5% FDR.

---

## Architecture

<p align="center">
  <img src="figures/figure0_architecture.png" alt="V2 Architecture: PeakEncoder + Self-Conditioned Absorbing Diffusion + CFID+SGIR" width="90%">
</p>

The pipeline has four stages:

1. **PeakEncoder** — maps the top-200 m/z peaks into a dense representation with sinusoidal positional encoding and a learnable B/Y-ion pair bias scalar `s` (initialized to 0, AlphaFold3-inspired). Removing PeakEncoder drops AA recall from 74.8% to 41.9%.

2. **Absorbing Multinomial Diffusion** — a Transformer denoiser operating over discrete amino-acid tokens. Forward diffusion absorbs tokens to a `[MASK]` state; the reverse process iteratively unmasks over T=20 steps with a cosine noise schedule.

3. **Self-Conditioning** — the previous denoised estimate is fed back as an additional input at each reverse step. Single largest gain: **+12.5 pp peptide accuracy** over V1.

4. **CFID + SGIR Reranking** — beam search (K=8) generates candidates; Cosine Fragment Ion Distribution scoring and Spectral-Guided Ion Rescoring select the best sequence. Best config: CFID+SGIR.

---

## Results

<p align="center">
  <img src="figures/figure5_progression.png" alt="Full model development progression vs InstaNovo SOTA" width="90%">
</p>

| Model | AA Recall | Peptide Accuracy |
|---|---|---|
| LSTM Baseline | 31.51% | 2.7% |
| GRU Baseline | 44.30% | 6.7% |
| Absorbing Diffusion (no PeakEnc) | 41.88% | 5.4% |
| V1 — PeakEnc + BY + CFID | 75.19% | 44.99% |
| **V2 — CFID + SGIR (ours)** | **76.00%** | **59.96%** |
| InstaNovo (published 2025 SOTA) | 72.90% | 33.10% |

3-seed mean, E. coli EV benchmark, 5% FDR. V2 beats InstaNovo by **+3.1 pp AA recall** and **+26.9 pp peptide accuracy**.

### Ablation Summary (3-seed mean)

| Config | AA Recall | Pep Accuracy |
|---|---|---|
| V2 argmax | 76.21% | 57.84% |
| V2 CFID | 75.93% | 59.60% |
| **V2 CFID + SGIR** | **75.99%** | **59.96%** |
| V2 SGIR only | 76.17% | 57.61% |
| V2 CFID + mass-correct | 75.21% | 50.44% |
| V2 rerank-spectral | 73.73% | 53.67% |

Mass correction is a negative result — post-hoc 50 ppm filtering is too aggressive and hurts peptide accuracy by ~9 pp, because the model was not trained with a mass-consistency constraint.

---

## Datasets

| Dataset | Description |
|---|---|
| **E. coli EV proteomics** | ~3,600 MS/MS spectra from E. coli extracellular vesicles; mzML + database-search xlsx ground truth. 70/15/15 train/val/test split. |
| **Wastewater proteomics** | Mixed-community sample (no reference proteome). 6 peptides from Sample 2 survived 5% FDR filtering; ESM-2 PPL scored. |

### Wastewater Findings

6 peptides from Sample 2 survived 5% FDR filtering without a reference proteome:

| Sequence | ESM-2 PPL | Anomalous Frac |
|---|---|---|
| FNDVIPMGEQAINTNEGAYR | 41.13 | 0.85 |
| NNGNAIGVDLAAIPFVAGDR | 31.51 | 0.85 |
| GSNYNEVVTLADVTIVQGIR | 36.63 | 0.90 |
| DLDVEFTALDGASVQVIAYR | 38.35 | 0.95 |
| ALDNAIDGGQYSFLEVAINR | 37.87 | 0.90 |
| QLDNNCVYLGATAGVPIAK  | 37.42 | 0.84 |

High anomalous fraction indicates structural novelty relative to the E. coli EV reference distribution — consistent with mixed-community microbial origin. ESM-2 PPL z-scores < −1.0 vs. E. coli GT; BLAST validation pending.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** `torch`, `pandas`, `numpy`, `pyteomics`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`.

---

## Usage

### Train

```bash
python src/diffusion.py
```

### Evaluate inference strategies (ablation)

```bash
python eval_novels.py
```

### Score sequences with ESM-2

```bash
python src/esm_scoring.py \
    --diffusion_csv results/diffusion_predictions.csv \
    --out_csv results/esm2_scores.csv
```

### Retrain all baselines + diffusion

```bash
python retrain_all.py
```

### Notebooks (in order)

| Notebook | Description |
|---|---|
| `notebooks/01_eda.ipynb` | Exploratory data analysis of E. coli EV spectra |
| `notebooks/02_preprocessing.ipynb` | mzML → tensor pipeline |
| `notebooks/03_baseline.ipynb` | LSTM / GRU baselines |
| `notebooks/04_diffusion.ipynb` | Full V2 training run (Colab-ready, T4 GPU) |

---

## Project Structure

```
peptide-diffusion/
├── src/
│   ├── diffusion.py          # V2 training loop — absorbing diffusion + self-conditioning
│   ├── baseline.py           # LSTM / GRU baselines + shared Encoder
│   ├── preprocessing.py      # mzML → tensor pipeline
│   ├── ensemble.py           # Ensemble grid search
│   ├── esm_scoring.py        # ESM-2 per-residue pseudo-perplexity scorer
│   ├── data_loader.py        # Dataset / DataLoader utilities
│   └── wastewater_pipeline.py # Wastewater inference pipeline
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_diffusion.ipynb
├── figures/                  # Result plots and architecture diagrams
├── results/                  # CSV outputs (predictions, ablation, wastewater)
├── checkpoints/              # Training metrics CSVs
├── report/                   # NeurIPS-style LaTeX report + PDF
├── eval_novels.py            # Ablation across 7 inference strategies
├── eval_iterative.py         # Iterative inference evaluation
├── retrain_all.py            # Re-runs all baselines + diffusion from scratch
└── requirements.txt
```

---

## Key Design Decisions

| Decision | Outcome |
|---|---|
| Absorbing (discrete) diffusion vs. Gaussian | Semantically sound for token sequences; avoids argmax quantization bias |
| Self-conditioning | +12.5 pp peptide accuracy — single largest gain |
| PeakEncoder + B/Y-ion bias | +32.9 pp AA recall vs. no encoder; bias scalar learns physics injection |
| Cosine noise schedule | Empirically outperforms linear; preserves signal at low-noise timesteps |
| Attention mask fix | Prevents cross-spectrum contamination in batch training |
| Mass correction (post-hoc) | Negative result: −9 pp peptide accuracy; fix requires mass-conditioned training loss |

---

## Team

Vaishak Girish Kumar · Akshay Mohan Revankar · Sanika Vilas Nanjan  
University at Buffalo — CSE 676 Deep Learning, Spring 2026
