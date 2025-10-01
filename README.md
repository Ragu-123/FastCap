# Enhanced FastCap: An Ultra-Efficient Image Captioning Model

Enhanced FastCap is a state-of-the-art, open-source image captioning model designed for ultra-efficient inference on standard CPUs while maintaining competitive quality. This project integrates nine novel techniques—from a Spatial-Mamba Vision Backbone to Adaptive Quantization-Aware Training—to create a model that is projected to be **5.1x faster** and **8.0x smaller** than current state-of-the-art solutions.

This repository contains the complete PyTorch implementation, training and evaluation scripts, and configuration files to reproduce and extend the Enhanced FastCap architecture.

![architecture](https://github.com/user-attachments/assets/9416e93e-c22a-4f9f-ba30-b240ecb1918a)

---

## Key Innovations

This model is built upon a synergistic combination of nine core innovations, each with a strong theoretical foundation:

* **Spatial-Mamba Vision Backbone:** A novel vision encoder with linear complexity O(n) for efficient and powerful spatial feature extraction.
* **Mixture of Expert (MoE) Decoder:** The first MoE-based decoder for image captioning, enabling adaptive, conditional computation for greater efficiency.
* **Iterative Conditional Masked Refinement (ICMR):** A non-autoregressive generation strategy with formal convergence guarantees for fast and high-quality captioning.
* **Dynamic Length-Aware Generation (DLAG):** Dynamically predicts caption length to reduce computational waste and improve quality on variable-length text.
* **Progressive Multi-Teacher Distillation (PMTD):** A curriculum-based distillation strategy that leverages a hierarchy of teacher models for superior knowledge transfer.
* **Cross-Modal Feature Alignment (CMFA):** A contrastive learning module that aligns vision and language spaces to improve cross-modal understanding.
* **Semantic Consistency Regularization (SCR):** A training-time regularizer that enforces consistency between parallel and sequential generation paths to improve semantic coherence.
* **Rank-Augmented Linear Attention (RALA):** An efficient attention mechanism based on rank restoration theory to preserve quality with linear complexity.
* **Adaptive Quantization-Aware Training (AQAT):** An advanced quantization technique that adaptively assigns bit-widths to layers based on their sensitivity, enabling significant compression with minimal accuracy loss.

---

## Quick Start

### 1. Installation

First, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/enhanced-fastcap.git
cd enhanced-fastcap

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

This project uses the `yerevann/coco-karpathy` dataset from Hugging Face. Run the provided script to download and cache the dataset automatically.

```bash
bash data/download.sh
```

### 3. Inference with a Pre-Trained Model

Generate a caption for any image using a trained checkpoint.

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_checkpoint.pth.tar \
    --image /path/to/your/image.jpg
```

### 4. Training

To start a new training run from scratch, use the `train.py` script with the base configuration file.

```bash
python scripts/train.py --config configs/model/fastcap_base.yaml
```

The training script will handle model initialization, data loading, training, validation, and checkpointing.

---


## Acknowledgements

Thanks to the open-source community and the maintainers of COCO, Hugging Face datasets, and PyTorch for foundational resources that make research and reproducibility possible.

---

