# A Unified Framework for Image-to-3D Part Generation via Variable Granularity

This repository contains the implementation of a unified framework for image-to-3D part generation with variable granularity support.

Our method introduces a unified approach for generating 3D parts from single images under flexible control conditions.



## Installation

We use:

- torch 2.5.1 + CUDA 12.4  
- Python 3.11  

### Create environment

```bash
conda create -n flexpart python=3.11
conda activate flexpart
```

### Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Install dependencies

```bash
bash settings/setup.sh
```

If no root permission is available:

```bash
conda install -c conda-forge libegl libglu pyopengl
```

---

## Interactive Demo

We provide an interactive Gradio-based demo.  
Users can draw points or bounding boxes on the input image to guide part-level 3D generation.

### Run demo

```bash
python app.py
```

---

## Inference

Generate 3D parts from a single image:

```bash
python scripts/inference_flexpart.py \
  --image_path example.png \
  --model_path ./weight \
  --mask_path ./mask.npy \
  --box_path ./box.npy \
  --point_path ./point.npy \
  --tag experiment \
  --render
```

### Arguments

- image_path: input image path  
- render: enable mesh rendering  

---

## Data Preprocessing

A preprocessing pipeline is provided for preparing training data.

Please follow dataset preparation guidelines included in the codebase.

---

## Training

We adopt a progressive training strategy.

---

### Stage 1: Base training

```bash
bash scripts/train.sh \
  --config configs/mp8_nt512.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir output \
  --tag stage1
```

---

### Stage 2: Fine-tuning

```bash
bash scripts/train.sh \
  --config configs/mp16_nt1024.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir output \
  --load_pretrained_model stage1 \
  --load_pretrained_model_ckpt [CHECKPOINT_ID] \
  --tag stage2
```

