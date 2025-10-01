# FastCap: Training Guide

This guide provides detailed instructions for training the FastCap model. It covers everything from standard training to advanced techniques like knowledge distillation and quantization-aware training.

All training processes are managed through the main `scripts/train.py` script and controlled by YAML configuration files located in the `configs/` directory.

## 1. Standard Training from Scratch

This is the most common starting point. You will train the model on a dataset like COCO from a random initialization.

### Step 1: Configure Your Dataset

Before starting, open `configs/model/fastcap_base.yaml` and update the dataset paths to point to your local machine.

```yaml
# In configs/model/fastcap_base.yaml
training:
  # ... other parameters
  
  # Update these paths
  train_annotations: '/path/to/your/coco/annotations/captions_train2014.json'
  val_annotations: '/path/to/your/coco/annotations/captions_val2014.json'
  image_dir: '/path/to/your/coco/images/'
```

### Step 2: Run the Training Script

Execute the `train.py` script, pointing it to the base configuration file.

```bash
python scripts/train.py --config configs/model/fastcap_base.yaml
```

The script will:

* Load the model architecture defined in the config.
* Load the COCO Karpathy dataset using the paths you specified.
* Begin training, showing progress with a tqdm bar.
* Periodically run validation and compute CIDEr, BLEU, and other metrics.
* Save checkpoints to the `checkpoints/` directory. The model with the best CIDEr score will be saved as `best_checkpoint.pth.tar`.

## 2. Advanced Training Scenarios

The codebase is designed to support advanced research workflows through different configuration files.

### Pre-training

For learning robust, general-purpose features, you can perform a longer pre-training run on a large dataset.

**Configuration:** `configs/training/pretrain.yaml`

**Command:**

```bash
python scripts/train.py --config configs/training/pretrain.yaml
```

**Key Differences:** This config typically specifies a larger `num_epochs` and may use a different, larger dataset.

### Fine-tuning

After pre-training, you can fine-tune the model on a specific, high-quality dataset like COCO to achieve the best possible performance.

**Configuration:** `configs/training/finetune.yaml`

**Command:**

```bash
python scripts/train.py --config configs/training/finetune.yaml
```

**Key Differences:** This config is defined with a lower learning rate and, most importantly, specifies a `resume_checkpoint` path to load the weights from your pre-trained model.

### Progressive Multi-Teacher Distillation (PMTD)

To create a smaller, faster model by transferring knowledge from larger teacher models, use the distillation training mode.

**Configuration:** `configs/training/distillation.yaml`

**Command:**

```bash
# Note: This requires a modified train script that can handle PMTD
python scripts/train.py --config configs/training/distillation.yaml
```

**Key Differences:** This config sets the `training.mode` to `pmtd` and includes paths to the pre-trained teacher models. The `train.py` script will automatically load the PMTD module and apply the curriculum-based loss.

## 3. Quantization-Aware Training (QAT)

After achieving a high-quality, full-precision model, you can perform QAT to create a highly compressed and efficient INT8/INT4 version.

### Step 1: Sensitivity Analysis

First, run the `quantize.py` script. This script loads your best full-precision model and runs a sensitivity analysis to determine which layers are most critical.

**Command:**

```bash
python scripts/quantize.py \
    --config configs/model/fastcap_base.yaml \
    --checkpoint checkpoints/best_checkpoint.pth.tar \
    --output-path checkpoints/quantized_model.pth
```

**Output:** This script will print the computed layer sensitivities and the adaptive bit allocation. It saves a post-training quantized model, which is a good baseline.

### Step 2: Fine-Tuning with QAT

For the best performance, you should fine-tune the model with quantization enabled.

**Configuration:** `configs/model/fastcap_quantized.yaml`

**Command:**

```bash
# Note: This requires a modified train script that applies quantization during training
python scripts/train.py --config configs/model/fastcap_quantized.yaml
```

**Key Differences:** This config loads the best full-precision checkpoint (`resume_checkpoint`), uses a very low learning rate, and enables quantization-specific logic in the training loop.

## 4. Resuming from a Checkpoint

If your training run is interrupted, you can easily resume from the last saved checkpoint.

Open your configuration file (e.g., `configs/model/fastcap_base.yaml`).

Set the `resume_checkpoint` parameter to the path of the checkpoint file.

```yaml
# In your config file
training:
  # ...
  resume_checkpoint: 'checkpoints/checkpoint.pth.tar' # Or the path to your last checkpoint
```


