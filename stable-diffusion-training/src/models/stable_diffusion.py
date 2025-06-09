import os
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

class StableDiffusionModel:
    """Wrapper around Diffusers components with memory optimizations."""

    def __init__(self, pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4", revision: str | None = None, torch_dtype: torch.dtype = torch.float16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        # Freeze VAE and text encoder by default
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Enable memory efficient attention for 16GB GPUs
        self.enable_memory_efficient_attention()

    def enable_memory_efficient_attention(self) -> None:
        """Enable attention slicing or xFormers when available."""
        try:
            self.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            self.unet.enable_attention_slicing()

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.torch_dtype)
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = 0.18215 * latents
        return latents

    def prepare_for_training(
        self,
        learning_rate: float = 1e-5,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-8,
        train_text_encoder: bool = False,
    ) -> torch.optim.Optimizer:
        params = list(self.unet.parameters())
        if train_text_encoder:
            self.text_encoder.train()
            for p in self.text_encoder.parameters():
                p.requires_grad_(True)
            params += list(self.text_encoder.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )
        return optimizer

    def create_pipeline(self) -> StableDiffusionPipeline:
        pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe = pipe.to(self.device)
        pipe.enable_attention_slicing()
        return pipe

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        self.noise_scheduler.save_pretrained(os.path.join(save_directory, "scheduler"))

