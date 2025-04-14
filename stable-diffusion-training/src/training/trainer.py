"""
Trainer for fine-tuning Stable Diffusion models.
"""
import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb
from diffusers import UNet2DConditionModel

from ..models.stable_diffusion import StableDiffusionModel

logger = logging.getLogger(__name__)

class StableDiffusionTrainer:
    """
    Trainer for fine-tuning Stable Diffusion models.
    """

    def __init__(
        self,
        model: StableDiffusionModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_train_epochs: int = 100,
        max_train_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: Optional[str] = "fp16",
        output_dir: str = "models/stable-diffusion-finetuned",
        logging_dir: str = "logs",
        logging_steps: int = 10,
        save_steps: int = 500,
        resume_from_checkpoint: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "stable-diffusion-finetuning",
        wandb_entity: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Stable Diffusion model
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            lr_scheduler: Learning rate scheduler
            num_train_epochs: Number of training epochs
            max_train_steps: Maximum number of training steps
            gradient_accumulation_steps: Number of gradient accumulation steps
            mixed_precision: Mixed precision training ("fp16", "bf16", or None)
            output_dir: Directory to save model checkpoints
            logging_dir: Directory to save logs
            logging_steps: Number of steps between logging
            save_steps: Number of steps between saving checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity name
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        # Create output and logging directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

        # Initialize optimizer
        self.optimizer = model.prepare_for_training()

        # Initialize accelerator
        if mixed_precision is not None:
            from accelerate import Accelerator
            self.accelerator = Accelerator(
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with="wandb" if use_wandb else None,
            )

            # Prepare model, optimizer, and dataloaders
            self.model.unet, self.optimizer, self.train_dataloader = self.accelerator.prepare(
                self.model.unet, self.optimizer, self.train_dataloader
            )

            if self.val_dataloader is not None:
                self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        else:
            self.accelerator = None

        # Initialize Weights & Biases
        if use_wandb:
            if self.accelerator is not None:
                self.accelerator.init_trackers(
                    project_name=wandb_project,
                    init_kwargs={"wandb": {"entity": wandb_entity}},
                )
            else:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                )

        # Calculate total training steps
        if max_train_steps is None:
            self.max_train_steps = num_train_epochs * len(train_dataloader)
        else:
            self.max_train_steps = max_train_steps

        # Initialize learning rate scheduler
        if lr_scheduler is None and self.accelerator is not None:
            from transformers import get_scheduler
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=int(0.1 * self.max_train_steps),
                num_training_steps=self.max_train_steps,
            )
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

    def train(self) -> None:
        """
        Train the model.
        """
        logger.info("Starting training")

        # Resume from checkpoint if specified
        if self.resume_from_checkpoint is not None:
            self._resume_from_checkpoint()

        # Initialize progress bar
        progress_bar = tqdm(range(self.max_train_steps), desc="Training")
        global_step = 0

        # Training loop
        for epoch in range(self.num_train_epochs):
            self.model.unet.train()

            for step, batch in enumerate(self.train_dataloader):
                # Use autocast for mixed precision training
                autocast_context = torch.cuda.amp.autocast(enabled=self.mixed_precision == "fp16")
                with autocast_context:
                    # Log data types for debugging
                    if step == 0 and epoch == 0:
                        logger.info(f"UNet dtype: {next(self.model.unet.parameters()).dtype}")
                        if 'pixel_values' in batch:
                            logger.info(f"Input dtype: {batch['pixel_values'].dtype}")

                    loss = self._training_step(batch)

                # Update progress bar
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                # Update weights
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    progress_bar.update(1)
                    global_step += 1

                # Log metrics
                if global_step % self.logging_steps == 0:
                    self._log_metrics({"train/loss": loss.detach().item()}, global_step)

                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self._save_checkpoint(global_step)

                # Validation
                if self.val_dataloader is not None and global_step % self.save_steps == 0:
                    self._validation_step(global_step)

                # Check if we've reached the maximum number of steps
                if global_step >= self.max_train_steps:
                    break

            # Save checkpoint at the end of each epoch
            self._save_checkpoint(global_step, f"epoch_{epoch}")

        # Save final model
        self._save_checkpoint(global_step, "final")

        logger.info("Training completed")

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch: Batch of data

        Returns:
            Loss value
        """
        # Get images and text embeddings
        pixel_values = batch["pixel_values"].to(self.model.device)
        input_ids = batch["input_ids"].to(self.model.device)

        # Ensure correct data types
        if self.mixed_precision == "fp16":
            # When using mixed precision, ensure inputs are float16
            pixel_values = pixel_values.to(dtype=torch.float16)
        else:
            # Otherwise use float32
            pixel_values = pixel_values.to(dtype=torch.float32)

        # Encode images to latent space
        latents = self.model.encode_images(pixel_values)

        # Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.model.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = self.model.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings
        encoder_hidden_states = self.model.text_encoder(input_ids)[0]

        # Predict noise
        noise_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Calculate loss
        if self.model.noise_scheduler.config.prediction_type == "epsilon":
            loss = F.mse_loss(noise_pred, noise)
        elif self.model.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.model.noise_scheduler.get_velocity(latents, noise, timesteps)
            loss = F.mse_loss(noise_pred, target)
        else:
            raise ValueError(f"Unknown prediction type: {self.model.noise_scheduler.config.prediction_type}")

        return loss

    def _validation_step(self, global_step: int) -> None:
        """
        Perform validation.

        Args:
            global_step: Current global step
        """
        logger.info("Running validation")

        self.model.unet.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                val_loss += self._training_step(batch).detach().item()

        val_loss /= len(self.val_dataloader)

        # Log validation loss
        self._log_metrics({"validation/loss": val_loss}, global_step)

        # Generate sample images
        self._generate_samples(global_step)

        self.model.unet.train()

    def _generate_samples(self, global_step: int, num_samples: int = 4) -> None:
        """
        Generate sample images for visualization.

        Args:
            global_step: Current global step
            num_samples: Number of samples to generate
        """
        if not self.use_wandb:
            return

        # Create pipeline
        pipeline = self.model.create_pipeline()
        pipeline.to(self.model.device)

        # Generate images
        prompts = [
            "a photo of a cat",
            "a photo of a dog",
            "a beautiful landscape with mountains",
            "a portrait of a person",
        ][:num_samples]

        images = []
        for prompt in prompts:
            with torch.autocast("cuda", enabled=self.mixed_precision == "fp16"):
                image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            images.append(wandb.Image(image, caption=prompt))

        # Log images
        wandb.log({"samples": images}, step=global_step)

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Current step
        """
        # Add learning rate
        if self.lr_scheduler is not None:
            metrics["train/learning_rate"] = self.lr_scheduler.get_last_lr()[0]

        # Log to console
        log_str = f"Step {step}: "
        log_str += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)

        # Log to Weights & Biases
        if self.use_wandb:
            if self.accelerator is not None:
                self.accelerator.log(metrics, step=step)
            else:
                wandb.log(metrics, step=step)

    def _save_checkpoint(self, global_step: int, suffix: Optional[str] = None) -> None:
        """
        Save a checkpoint.

        Args:
            global_step: Current global step
            suffix: Optional suffix for the checkpoint directory
        """
        if suffix is not None:
            save_dir = os.path.join(self.output_dir, f"checkpoint-{suffix}")
        else:
            save_dir = os.path.join(self.output_dir, f"checkpoint-{global_step}")

        logger.info(f"Saving checkpoint to {save_dir}")

        # Save model
        if self.accelerator is not None:
            unwrapped_unet = self.accelerator.unwrap_model(self.model.unet)
            self.model.unet = unwrapped_unet

        self.model.save_pretrained(save_dir)

        # Save optimizer and scheduler
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

        # Save training state
        torch.save(
            {
                "global_step": global_step,
                "epoch": global_step // len(self.train_dataloader),
            },
            os.path.join(save_dir, "training_state.pt"),
        )

    def _resume_from_checkpoint(self) -> None:
        """
        Resume training from a checkpoint.
        """
        logger.info(f"Resuming from checkpoint: {self.resume_from_checkpoint}")

        # Load model
        self.model.unet = UNet2DConditionModel.from_pretrained(
            os.path.join(self.resume_from_checkpoint, "unet"),
            torch_dtype=self.model.torch_dtype,
        ).to(self.model.device)

        # Load optimizer
        optimizer_path = os.path.join(self.resume_from_checkpoint, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))

        # Load scheduler
        scheduler_path = os.path.join(self.resume_from_checkpoint, "scheduler.pt")
        if os.path.exists(scheduler_path) and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path))

        # Load training state
        training_state_path = os.path.join(self.resume_from_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            global_step = training_state["global_step"]

            # Skip already processed steps
            self.max_train_steps += global_step
