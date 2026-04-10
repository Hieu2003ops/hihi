# 🚀 MobileIE: Mobile Image Enhancement with HybridMixUNet and QAT

This repository provides a comprehensive pipeline for **Mobile Image Enhancement (MobileIE)**. It features `HybridMixUNet`, a lightweight architecture designed for mobile deployment, supported by Quantization-Aware Training (QAT) and an end-to-end export pipeline to TFLite (FP32 & INT8). 

The project is heavily optimized for evaluating performance (PSNR/SSIM) vs. efficiency (Latency/Model Size) on mobile devices using the DPED dataset.

## 📑 Table of Contents
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Training](#training)
- [Export & Quantization](#export--quantization)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Inference & Visualization](#inference--visualization)
- [Ablation Studies](#ablation-studies)

---

## 🛠 Requirements
- Python `< 3.12`
- Install dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Optional)* For WandB logging, create a `.env` file in the root directory and add your `WANDB_API_KEY`.

---

## 📂 Repository Structure

The codebase is modularized for academic research and deployment:

### 1. 🏗️ Model & Data
* `model_builder.py`: Defines the `HybridMixUNet` architecture, ablation variants, and handles checkpoint loading.
* `loss.py`: Implements robust loss functions (PSNR, MS-SSIM, Content Loss, OutlierAwareLoss, and combinations).
* `dped_dataset.py`: PyTorch Lightning DataModule for DPED dataset management (supports full-HD and patch modes).
* `data_aug.py`: GPU-accelerated data augmentations using Kornia.

### 2. 🏃‍♂️ Training
* `train_qat.py`: Main PyTorch Lightning training script. Supports standard FP32/BF16 training and FX Graph Mode Quantization-Aware Training (QAT).
* `train_utils_builder.py`: Registries and factories for Loss functions and LR Schedulers.

### 3. ⚙️ Export & Quantization
* `to_tflite.py`: End-to-end pipeline to convert PyTorch models to ONNX -> TF SavedModel -> TFLite (supports both FP32 and INT8 PTQ).
* `quantize.py`: Converts Lightning checkpoints to FX INT8 models (handles QAT conversion and PTQ observer warmup).

### 4. 📊 Evaluation & Benchmarking
* `eval_pytorch.py` / `eval_tflite.py`: Scripts to compute PSNR/SSIM on the DPED test split for PyTorch and TFLite models.
* `benchmark_ckpts.py` / `benchmark_quantized.py`: Automated benchmarking tools to measure model size (MiB), parameter count, PSNR/SSIM metrics, and synthetic forward latency.
* `eval_original_images.py`: Per-phone TFLite conversion and evaluation on original, full-resolution images.

### 5. 👁️ Inference
* `infer_tflite.py`: Runs single-image inference using TFLite. Supports dynamic resize and tiled/patched inference for high-res images.
* `infer_visualize.py`: Generates visual comparison panels (Input vs. Enhanced vs. Ground Truth) and ranks images by metrics.

---

## 🏃‍♂️ Training

You can train the model from scratch or use Quantization-Aware Training (QAT). The training script is fully integrated with PyTorch Lightning and Weights & Biases (WandB).

**Example: QAT with BF16 precision and Cosine Warmup**
```bash
python train_qat.py \
  --model_name model_c32_loss02_quantize_norestart \
  --channels 32 \
  --loss_version 2 \
  --precision bf16 \
  --qat True \
  --scheduler_type cosine_warmup \
  --warmup_epochs 5 \
  --warmup_start_factor 0.1 \
  --eta_min 5e-6 \
  --num_epochs 50 \
  --limit_train_batches 0.01 \
  --use_wandb False
```

---

## ⚙️ Export & Quantization

To deploy the model on mobile devices, export the trained PyTorch checkpoint to TFLite (FP32 or INT8).

**Convert to TFLite (FP32)**
```bash
python to_tflite.py \
  --ckpt_path /path/to/model.ckpt \
  --channels 24 \
  --output_dir /path/to/results \
  --model_name modelv7_c24
```

**Convert to INT8 (PyTorch FX Graph)**
```bash
python quantize.py \
  --ckpt_path /path/to/model.ckpt \
  --save_path /path/to/save/int8 \
  --data_dir /path/to/dped 
```

---

## 📊 Evaluation & Benchmarking

### 1. Evaluate Metrics (PSNR / SSIM)
**Evaluate a TFLite file:**
```bash
python eval_tflite.py \
  --tflite_file /path/to/results/model.tflite \
  --data_dir ./dataset/dped/dped \
  --output_csv /path/to/results/eval_one.csv
```

**Evaluate all TFLite files in a directory:**
```bash
python eval_tflite.py \
  --tflite_dir /path/to/results \
  --data_dir ./dataset/dped/dped \
  --output_csv /path/to/results/eval_all.csv
```

### 2. Run Benchmarks (Latency & Memory)
To measure the real-world efficiency of your checkpoints:
```bash
# Benchmark PyTorch Checkpoints
python benchmark_ckpts.py --ckpt_dir ./ckpts --data_dir ./dataset/dped/dped --output_csv ./ckpts/benchmark_results.csv

# Benchmark Quantized (INT8/TFLite) Models
python benchmark_quantized.py --model_dir ./results/int8 --data_dir ./dataset/dped/dped
```

---

## 👁️ Inference & Visualization

Generate comparison images (Input | Output | Ground Truth) and rank them based on MSSIM or PSNR:
```bash
python infer_visualize.py \
  --metric combined \
  --top_k 10 
```

Run inference on a single high-resolution image using TFLite (supports dynamic padding and tile-based overlapping):
```bash
python infer_tflite.py \
  --tflite_file /path/to/model.tflite \
  --input /path/to/input.jpg \
  --strategy auto
```

---

## 🔬 Ablation Studies

We provide automated bash scripts to run various ablation studies (Channel width, Loss versions, QAT, and Internal block components).

```bash
export DATA_DIR="/home/tinhanh/MobileAI/mixedmodel/dataset/dped/dped"
export RESULTS_BASE="/home/tinhanh/MobileAI/submission/Source-Codes/scripts/ablation/results"
export EPOCHS=50
export CHANNELS=32
export LOSS_VERSION=2

# 1. Channel Ablation (Sweeps channels: 16, 24, 32, 64)
bash ./scripts/ablation/channel_ablation.sh "$DATA_DIR" "$RESULTS_BASE/channel_ablation_loss${LOSS_VERSION}" "$LOSS_VERSION" "$EPOCHS"

# 2. Loss Ablation (Sweeps loss_version: 1, 2, 3)
bash ./scripts/ablation/loss_ablation.sh "$DATA_DIR" "$RESULTS_BASE/loss_ablation_c${CHANNELS}"

# 3. QAT Ablation (Trains with and without QAT, then quantizes NO-QAT)
bash ./scripts/ablation/qat_ablation.sh "$DATA_DIR" "$RESULTS_BASE/qat_ablation_c${CHANNELS}_loss${LOSS_VERSION}" "$CHANNELS" "$LOSS_VERSION" "$EPOCHS"

# 4. Component Ablation (Tests hybrid internal variants like ablate_gate, ablate_residual, etc.)
bash ./scripts/ablation/block_ablation.sh "$DATA_DIR" "$RESULTS_BASE/component_ablation_c${CHANNELS}_loss${LOSS_VERSION}" "$CHANNELS" "$LOSS_VERSION" "$EPOCHS"
```