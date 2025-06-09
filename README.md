# Stable Diffusion Training with LAION-400M

This repository contains code for fine-tuning Stable Diffusion models using the LAION-400M dataset.

## Features

- Fine-tune Stable Diffusion models on the LAION-400M dataset
- Robust dataset handling with error recovery
- Mixed precision training support
- Automatic memory-efficient attention for GPUs with 16GB VRAM
- Inference script for generating images
- Web UI for interactive image generation

## Setup

### Prerequisites

- Python 3.8+
- Intel Core i7-14700 CPU
- NVIDIA 5060 Ti GPU with 16GB VRAM
- 32GB system RAM
- 500GB SSD
- Hugging Face account with API token

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stable-diffusion-training.git
cd stable-diffusion-training
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Authentication

Before using the LAION-400M dataset, you need to authenticate with Hugging Face:

```bash
python run_auth_test.py --token YOUR_HUGGING_FACE_TOKEN
```

### Testing Dataset Loading

Test that the dataset can be loaded correctly:

```bash
python run_test.py --token YOUR_HUGGING_FACE_TOKEN --num_samples 2
```

### Training

Start training with a small dataset to verify everything works:

```bash
python run_training.py --token YOUR_HUGGING_FACE_TOKEN --max_train_samples 1000 --max_val_samples 100 --train_batch_size 2 --val_batch_size 2 --num_train_epochs 1 --gradient_accumulation_steps 4 --lr_warmup_steps 100 --mixed_precision no
```

For full training:

```bash
python run_training.py --token YOUR_HUGGING_FACE_TOKEN --max_train_samples 5000 --max_val_samples 500 --train_batch_size 4 --val_batch_size 4 --num_train_epochs 10 --gradient_accumulation_steps 4 --lr_warmup_steps 500 --mixed_precision no
```

### Generating Images

After training, generate images using:

```bash
PYTHONPATH=. python inference.py --model_path models/stable-diffusion-finetuned/checkpoint-final --prompt "a beautiful landscape" --num_images 4
```

### Web UI

Launch the web UI for interactive image generation:

```bash
PYTHONPATH=. python ui.py --model_path models/stable-diffusion-finetuned/checkpoint-final
```

## Project Structure

- `src/`: Source code
  - `data/`: Dataset handling code
  - `models/`: Model definitions
  - `training/`: Training logic
- `scripts/`: Utility scripts
- `configs/`: Configuration files
- `run_training.py`: Main training script
- `inference.py`: Image generation script
- `ui.py`: Web UI for image generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Stability AI](https://stability.ai/) for the Stable Diffusion model
- [LAION](https://laion.ai/) for the LAION-400M dataset
- [Hugging Face](https://huggingface.co/) for model hosting and libraries
