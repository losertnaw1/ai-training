"""
Configuration utilities.
"""
import os
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def update_config_from_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with command-line arguments.
    
    Args:
        config: Configuration dictionary
        args: Command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Update data configuration
    if hasattr(args, "resolution"):
        config["data"]["resolution"] = args.resolution
    if hasattr(args, "center_crop"):
        config["data"]["center_crop"] = args.center_crop
    if hasattr(args, "random_flip"):
        config["data"]["random_flip"] = args.random_flip
    
    # Update model configuration
    if hasattr(args, "pretrained_model_name_or_path"):
        config["model"]["pretrained_model_name_or_path"] = args.pretrained_model_name_or_path
    if hasattr(args, "revision"):
        config["model"]["revision"] = args.revision
    if hasattr(args, "train_text_encoder"):
        config["model"]["train_text_encoder"] = args.train_text_encoder
    
    # Update training configuration
    if hasattr(args, "output_dir"):
        config["training"]["output_dir"] = args.output_dir
    if hasattr(args, "logging_dir"):
        config["training"]["logging_dir"] = args.logging_dir
    if hasattr(args, "seed"):
        config["training"]["seed"] = args.seed
    if hasattr(args, "mixed_precision"):
        config["training"]["mixed_precision"] = args.mixed_precision
    if hasattr(args, "train_batch_size"):
        config["training"]["train_batch_size"] = args.train_batch_size
    if hasattr(args, "val_batch_size"):
        config["training"]["val_batch_size"] = args.val_batch_size
    if hasattr(args, "num_train_epochs"):
        config["training"]["num_train_epochs"] = args.num_train_epochs
    if hasattr(args, "max_train_steps"):
        config["training"]["max_train_steps"] = args.max_train_steps
    if hasattr(args, "gradient_accumulation_steps"):
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if hasattr(args, "learning_rate"):
        config["training"]["learning_rate"] = args.learning_rate
    if hasattr(args, "lr_scheduler"):
        config["training"]["lr_scheduler"] = args.lr_scheduler
    if hasattr(args, "lr_warmup_steps"):
        config["training"]["lr_warmup_steps"] = args.lr_warmup_steps
    if hasattr(args, "adam_beta1"):
        config["training"]["adam_beta1"] = args.adam_beta1
    if hasattr(args, "adam_beta2"):
        config["training"]["adam_beta2"] = args.adam_beta2
    if hasattr(args, "adam_weight_decay"):
        config["training"]["adam_weight_decay"] = args.adam_weight_decay
    if hasattr(args, "adam_epsilon"):
        config["training"]["adam_epsilon"] = args.adam_epsilon
    if hasattr(args, "max_grad_norm"):
        config["training"]["max_grad_norm"] = args.max_grad_norm
    if hasattr(args, "logging_steps"):
        config["training"]["logging_steps"] = args.logging_steps
    if hasattr(args, "save_steps"):
        config["training"]["save_steps"] = args.save_steps
    if hasattr(args, "resume_from_checkpoint"):
        config["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    
    # Update Weights & Biases configuration
    if hasattr(args, "use_wandb"):
        config["wandb"]["use_wandb"] = args.use_wandb
    if hasattr(args, "wandb_project"):
        config["wandb"]["project"] = args.wandb_project
    if hasattr(args, "wandb_entity"):
        config["wandb"]["entity"] = args.wandb_entity
    
    return config
