# Transformer Gradient Geometry Analysis

This repository presents a systematic empirical study of optimization dynamics in Transformer-based neural architectures under **Full Fine-Tuning** and **Low-Rank Adaptation (LoRA)**. The project focuses on analyzing gradient structure, stability, and depth-wise learning behavior to understand how parameter-efficient adaptation constrains and approximates unconstrained optimization.

The study emphasizes **mechanistic analysis** over task-level performance and provides quantitative evidence explaining the behavior of low-rank adaptation during training.

---

## Project Scope

The project investigates the following aspects of Transformer optimization:

- Geometric properties of gradient updates
- Sensitivity to learning rate variations
- Directional alignment between optimization trajectories
- Distribution of learning across model depth
- Stability and variance of parameter updates

**Goal:** Characterize how LoRA modifies learning dynamics and identify the trade-offs between efficiency and representational flexibility.

---

## Experimental Pipeline

The repository consists of five sequential experiments implemented as Jupyter notebooks.

### 1. Baseline Gradient Geometry
**`01_baseline_gradient_geometry.ipynb`**

Establishes reference statistics for gradient norms and random projections under standard fine-tuning. This notebook provides baseline geometric distributions used for comparative analysis.

### 2. Learning Rate Sensitivity
**`02_lr_sensitivity_gradient_geometry.ipynb`**

Analyzes the effect of different learning rates on gradient variance, convergence behavior, and optimization stability. Identifies stable and unstable learning regimes.

### 3. Full Fine-Tuning vs LoRA
**`03_lora_vs_fullft_gradient_geometry.ipynb`**

Compares gradient magnitude, dispersion, and distribution between Full Fine-Tuning and LoRA. Quantifies the effect of low-rank constraints on optimization dynamics.

### 4. Alignment Analysis
**`04_alignment_check.ipynb`**

Computes cosine similarity between Full Fine-Tuning updates and effective LoRA updates across training steps. Evaluates how closely LoRA follows the dominant optimization trajectory.

### 5. Layer-wise Learning Activity
**`05_layer_activity.ipynb`**

Aggregates gradient norms by Transformer layer and analyzes mean and variance across depth. Reveals depth-selective learning patterns and stability properties.

---

## Methodology

Experiments are conducted using the **DistilGPT-2** architecture within the Hugging Face Transformers framework. Two training strategies are evaluated:

1. **Full Fine-Tuning**
2. **LoRA-based adaptation**

During training, per-parameter gradient statistics are logged and stored as serialized files for offline analysis. Collected metrics include:

- Gradient norms
- Random projections
- Cosine similarity
- Layer-wise variance estimates

---

## Quantitative Findings

### Learning Rate Sensitivity
Lower learning rates produce more stable gradients, while higher learning rates increase variance.

### Full Fine-Tuning vs LoRA
- **LoRA** significantly reduces gradient dispersion
- **Full Fine-Tuning** exhibits higher variance and larger update magnitudes
- **LoRA** produces smaller and more consistent updates

### Alignment Analysis
Partial but stable trajectory alignment observed, with cosine similarity converging to approximately **0.17–0.19** during training.

### Layer-wise Analysis
- **Full Fine-Tuning:** Distributes learning across all layers
- **LoRA:** Concentrates adaptation in shallow and intermediate layers with reduced variance

---

## Repository Organization

```
.
├── notebooks/              # Experimental and analysis notebooks
├── logs/                   # Serialized training statistics (gitignored)
├── figures/                # Generated visualizations (gitignored)
├── README.md               # Project documentation
├── requirements.txt        # Dependency specification
└── .gitignore             # Artifact exclusion rules
```

> **Note:** Only source notebooks and configuration files are tracked in version control.

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Recommended Environment

- Python 3.9 or later
- PyTorch 2.0 or later
- Transformers 4.35 or later
- PEFT
- NumPy
- Matplotlib

> **GPU acceleration is recommended** for training experiments.

---

## Usage

1. Execute notebooks in **sequential order** from `01` to `05`
2. Training notebooks generate serialized log files
3. Archive and reuse log files for analysis notebooks
4. Analysis notebooks operate exclusively on stored logs and **do not require retraining**

### Reproducibility

For reproducible results, maintain:
- Consistent random seeds
- Identical hyperparameters
- Standardized dataset preprocessing

---

## Limitations

- Experiments conducted on **small to medium-scale models** (may not generalize to very large architectures)
- **Limited hyperparameter exploration**
- Only **LoRA** is analyzed in depth

---

## Future Directions

Future extensions may include:

- Rank-scaling studies
- Comparisons with **DoRA** and **GaLore**
- Experiments on larger models
- Task-specific evaluations
- Theoretical modeling of optimization geometry
