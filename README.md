# Transformer Gradient Geometry Analysis

## Overview

This repository presents an empirical study of optimization dynamics in Transformer-based language models, with a focus on understanding the geometric and statistical properties of gradients under Full Fine-Tuning and Low-Rank Adaptation (LoRA). The project investigates how parameter-efficient fine-tuning alters learning trajectories, stability, and depth-wise adaptation.

The work provides quantitative evidence explaining the effectiveness and limitations of LoRA through controlled experiments and systematic analysis.

---

## Project Description

Recent parameter-efficient fine-tuning methods such as LoRA achieve competitive performance with significantly fewer trainable parameters. However, the underlying optimization mechanisms are not yet fully understood. This project addresses this gap by analyzing gradient geometry, learning rate sensitivity, trajectory alignment, and layer-wise learning activity in Transformer models.

The experiments aim to provide a mechanistic interpretation of LoRA as a constrained and stable approximation of full fine-tuning.

---

## Research Objectives

The primary objectives of this project are:

- **Characterize gradient behavior** during Transformer fine-tuning
- **Analyze the effect of learning rate** on optimization stability
- **Quantify alignment** between Full Fine-Tuning and LoRA updates
- **Study depth-wise learning distributions** across model layers
- **Assess the stability and variance** of parameter updates

---

## Experimental Design

The study is organized into five sequential experiments:

### 1. Baseline Gradient Geometry
Establishes reference statistics for gradient norms and projections.

### 2. Learning Rate Sensitivity
Evaluates the effect of different learning rates on gradient stability.

### 3. Full Fine-Tuning vs LoRA
Compares gradient magnitude and dispersion between methods.

### 4. Alignment Analysis
Measures cosine similarity between Full Fine-Tuning and LoRA updates.

### 5. Layer-wise Learning Activity
Analyzes how learning is distributed across Transformer layers.

---

## Methodology

Experiments are conducted using the **DistilGPT-2** model within the Hugging Face Transformers framework. Two fine-tuning strategies are evaluated:

- **Full Fine-Tuning**
- **LoRA-based adaptation**

Gradient statistics are logged during training and analyzed offline. Metrics include:

- Gradient norms
- Random projections
- Cosine similarity
- Layer-wise variance estimates

---

## Quantitative Results

### Learning Rate Sensitivity
- Lower learning rates exhibit greater stability
- Higher rates increase gradient variance

### Full Fine-Tuning vs LoRA
- **LoRA** significantly reduces gradient dispersion
- **Full Fine-Tuning**: Mean projection ≈ 0.23 with high variance
- **LoRA**: Mean projection ≈ 0.09 with substantially lower variance

### Alignment Analysis
- Partial trajectory alignment observed
- Cosine similarity stabilizes in the range of **0.17 to 0.19** after early training

### Layer-wise Analysis
- **Full Fine-Tuning**: Distributes learning across all layers
- **LoRA**: Concentrates learning in shallow and intermediate layers with lower variance

---

## Discussion

The results indicate that LoRA constrains optimization to a low-dimensional subspace, producing implicit regularization and improved stability. Key findings:

- **Reduced variance** leads to consistent convergence
- **Depth-selective learning** explains partial misalignment with full fine-tuning trajectories
- LoRA functions as a **constrained approximation** of full fine-tuning that preserves dominant optimization directions while limiting parameter updates

---

## Limitations

- Experiments conducted on **small to medium-scale models**, which may limit generalization to large foundation models
- **Limited hyperparameter exploration**
- Only **LoRA** is analyzed in depth

---

## Future Work

Future research directions include:

- **Rank-scaling studies** to understand the impact of LoRA rank parameter
- **Comparisons with DoRA and GaLore** for broader method evaluation
- **Experiments on larger models** to validate findings at scale
- **Task-specific evaluations** across diverse downstream tasks
- **Theoretical modeling** of optimization geometry

---



## Contact

For questions or collaboration inquiries, please open an issue or contact [your email].
