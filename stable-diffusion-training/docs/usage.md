# Stable Diffusion Training with LAION-400M

This document provides instructions for using the Stable Diffusion training pipeline with the LAION-400M dataset, a large-scale dataset containing 400 million image-text pairs.

## Setup

### Prerequisites

- Python 3.8 or higher
- Intel Core i7-14700 CPU
- NVIDIA 5060 Ti GPU with 16GB of VRAM
- 32GB of system RAM
- 500GB SSD (or equivalent free disk space)

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

## Data Preparation

### Using the LAION-400M Dataset

The project is configured to use the LAION-400M dataset from Hugging Face Hub. The dataset will be automatically downloaded during training, so no separate download step is required.

Since LAION-400M is a gated dataset, you'll need to provide a Hugging Face token with the `--hf_token` argument. You can control the number of samples used for training and validation with the `--max_train_samples` and `--max_val_samples` arguments.

### Using Your Own Dataset

If you want to use your own dataset, you need to organize it in the following structure:

```
data/
  custom_dataset/
    images/
      image1.jpg
      image2.jpg
      ...
    metadata.json
```

The `metadata.json` file should have the following format:

```json
[
  {
    "image_id": "00000000",
    "image_path": "data/custom_dataset/images/image1.jpg",
    "caption": "A description of the image"
  },
  ...
]
```

## Training

### Configuration

The training configuration is specified in YAML files in the `configs` directory. You can modify the default configuration or create your own.

### Training Command

To train the model with the default configuration:

```bash
python train.py --config configs/default_config.yaml
```

You can override configuration values with command-line arguments:

```bash
python train.py --config configs/default_config.yaml --train_batch_size 8 --num_train_epochs 50
```

### Resuming Training

To resume training from a checkpoint:

```bash
python train.py --config configs/default_config.yaml --resume_from_checkpoint models/stable-diffusion-finetuned/checkpoint-1000
```

## Inference

### Generating Images

To generate images with a fine-tuned model:

```bash
python inference.py --model_path models/stable-diffusion-finetuned/checkpoint-final --prompt "a photo of a cat" --num_images 4
```

### Using the UI

To launch the Gradio UI for interactive image generation:

```bash
python ui.py --model_path models/stable-diffusion-finetuned/checkpoint-final
```

This will start a web server at `http://localhost:7860` where you can interact with the model.

## Tips for Better Results

1. **Data Quality**: Use high-quality images with good captions for training.
2. **Training Duration**: Train for at least 10,000 steps for good results.
3. **Batch Size**: Use the largest batch size that fits in your GPU memory.
4. **Learning Rate**: Start with a learning rate of 1e-5 and adjust as needed.
5. **Text Encoder**: Fine-tuning the text encoder can improve results but requires more VRAM.

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce the batch size
2. Use gradient accumulation (increase `gradient_accumulation_steps`)
3. Reduce the image resolution
4. Use mixed precision training (`--mixed_precision fp16`)
5. Memory efficient attention is enabled automatically for 16GB GPUs.

### Poor Image Quality

If the generated images have poor quality:

1. Train for more steps
2. Use a larger dataset
3. Adjust the learning rate
4. Try different guidance scales during inference
