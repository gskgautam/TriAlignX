"""
Utility functions for TriAlignX framework
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device(device: str = "auto") -> torch.device:
    """Setup device for training"""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    return torch.device(device)

def setup_huggingface_token(config):
    """Setup Hugging Face token for authentication"""
    if 'huggingface' in config and 'token' in config['huggingface']:
        token = config['huggingface']['token']
        if token != "hf_your_token_here":
            import os
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            print("Hugging Face token set from config")
            return True
        else:
            print("Warning: Please replace 'hf_your_token_here' with your actual Hugging Face token in config.yaml")
            return False
    return False

def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("trialignx")
    logger.setLevel(getattr(logging, level))
    
    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, "trialignx.log"))
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_checkpoint(model, optimizer, epoch, loss, path: str):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path: str):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def count_parameters(model) -> int:
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_directories(paths: list):
    """Create directories if they don't exist"""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience 