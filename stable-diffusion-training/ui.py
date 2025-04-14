#!/usr/bin/env python
"""
Gradio UI for generating images with a fine-tuned Stable Diffusion model.
"""
import os
import argparse
import logging
import sys
from pathlib import Path

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class StableDiffusionUI:
    """
    Gradio UI for Stable Diffusion image generation.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the UI.

        Args:
            model_path: Path to the fine-tuned model
            device: Device to run inference on
            torch_dtype: Data type for model parameters
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        self.pipeline = self.pipeline.to(device)

        # Enable memory optimization
        self.pipeline.enable_attention_slicing()

        logger.info("Model loaded successfully")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
    ):
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt
            negative_prompt: Negative text prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            width: Image width
            height: Image height
            seed: Random seed

        Returns:
            Generated image
        """
        # Set random seed
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Generate image
        with torch.autocast(self.device):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )

        return result.images[0]

    def create_ui(self):
        """
        Create the Gradio UI.

        Returns:
            Gradio interface
        """
        with gr.Blocks(title="Stable Diffusion Text-to-Image") as interface:
            gr.Markdown("# Stable Diffusion Text-to-Image Generator")
            gr.Markdown(f"Using model: {self.model_path}")

            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3,
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Enter negative prompt here...",
                        lines=2,
                    )

                    with gr.Row():
                        with gr.Column():
                            num_inference_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=10,
                                maximum=150,
                                value=50,
                                step=1,
                            )
                            guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.1,
                            )

                        with gr.Column():
                            width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=1024,
                                value=512,
                                step=64,
                            )
                            height = gr.Slider(
                                label="Height",
                                minimum=256,
                                maximum=1024,
                                value=512,
                                step=64,
                            )

                    seed = gr.Slider(
                        label="Seed (-1 for random)",
                        minimum=-1,
                        maximum=2147483647,
                        value=-1,
                        step=1,
                    )

                    generate_button = gr.Button("Generate Image")

                with gr.Column():
                    output_image = gr.Image(label="Generated Image")

            generate_button.click(
                fn=self.generate_image,
                inputs=[
                    prompt,
                    negative_prompt,
                    num_inference_steps,
                    guidance_scale,
                    width,
                    height,
                    seed,
                ],
                outputs=output_image,
            )

        return interface

def parse_args():
    parser = argparse.ArgumentParser(description="Gradio UI for Stable Diffusion")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to create a publicly shareable link",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server name",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Create UI
    ui = StableDiffusionUI(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    # Launch UI
    interface = ui.create_ui()
    interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )

if __name__ == "__main__":
    main()
