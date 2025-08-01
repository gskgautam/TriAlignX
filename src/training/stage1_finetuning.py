"""
Stage 1: Axis-specific fine-tuning for TriAlignX
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, setup_device, setup_logging, set_seed, save_checkpoint
from data_processing.dataset_loader import create_dataloaders

def setup_model_and_tokenizer(config, base_model_name):
    """Setup model and tokenizer with LoRA"""
    
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias']
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def train_axis(config, axis, model, tokenizer, train_dataloader, val_dataloader, device, logger):
    """Train model for a specific axis"""
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['stage1']['learning_rate'],
        weight_decay=config['training']['stage1']['weight_decay']
    )
    
    num_training_steps = len(train_dataloader) * config['training']['stage1']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['stage1']['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['stage1']['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['stage1']['num_epochs']}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training {axis}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['stage1']['max_grad_norm']
            )
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{train_loss / (batch_idx + 1):.4f}"
            })
            
            # Log to wandb
            if batch_idx % config['logging']['log_interval'] == 0:
                wandb.log({
                    f'{axis}_train_loss': loss.item(),
                    f'{axis}_learning_rate': scheduler.get_last_lr()[0],
                    f'{axis}_epoch': epoch + batch_idx / len(train_dataloader)
                })
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validating {axis}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_dataloader)
        train_loss /= len(train_dataloader)
        
        logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Log to wandb
        wandb.log({
            f'{axis}_epoch_train_loss': train_loss,
            f'{axis}_epoch_val_loss': val_loss,
            f'{axis}_epoch': epoch + 1
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(
                config['paths']['output_dir'], 
                f"{axis}_stage1_best.pt"
            )
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)
            logger.info(f"Saved best model for {axis} with val loss: {val_loss:.4f}")
    
    # Save final model
    final_save_path = os.path.join(
        config['paths']['output_dir'], 
        f"{axis}_stage1_final.pt"
    )
    save_checkpoint(model, optimizer, config['training']['stage1']['num_epochs'] - 1, val_loss, final_save_path)
    
    return model

def extract_task_vector(base_model, finetuned_model):
    """Extract task vector (delta) between base and fine-tuned model"""
    base_state = base_model.state_dict()
    finetuned_state = finetuned_model.state_dict()
    
    delta = {}
    for key in base_state:
        if key in finetuned_state:
            delta[key] = finetuned_state[key] - base_state[key]
    
    return delta

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Axis-specific fine-tuning")
    parser.add_argument("--axis", type=str, required=True, 
                       choices=["helpful", "harmless", "honest"],
                       help="Axis to fine-tune for")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Base model to fine-tune")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup Hugging Face token
    from utils import setup_huggingface_token
    setup_huggingface_token(config)
    
    # Setup logging
    logger = setup_logging(config['paths']['logs_dir'])
    logger.info(f"Starting Stage 1 fine-tuning for axis: {args.axis}")
    
    # Set seed
    set_seed(config['data']['random_seed'])
    
    # Setup device
    device = setup_device(config['hardware']['device'])
    logger.info(f"Using device: {device}")
    
    # Setup wandb
    wandb.init(
        project=config['logging']['project_name'],
        name=f"stage1_{args.axis}",
        config=config
    )
    
    # Create directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    # Load tokenizer and create dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataloaders = create_dataloaders(config, tokenizer)
    
    # Get dataloaders for the specific axis
    train_dataloader = dataloaders[args.axis]['train']
    val_dataloader = dataloaders[args.axis]['test']  # Using test as validation for now
    
    logger.info(f"Train samples: {len(train_dataloader.dataset)}")
    logger.info(f"Val samples: {len(val_dataloader.dataset)}")
    
    # Setup base model (for task vector extraction)
    base_model, _ = setup_model_and_tokenizer(config, args.base_model)
    
    # Setup model for training
    model, _ = setup_model_and_tokenizer(config, args.base_model)
    model.to(device)
    
    # Train the model
    trained_model = train_axis(
        config, args.axis, model, tokenizer, 
        train_dataloader, val_dataloader, device, logger
    )
    
    # Extract task vector
    logger.info("Extracting task vector...")
    task_vector = extract_task_vector(base_model, trained_model)
    
    # Save task vector
    task_vector_path = os.path.join(
        config['paths']['output_dir'], 
        f"{args.axis}_task_vector.pt"
    )
    torch.save(task_vector, task_vector_path)
    logger.info(f"Saved task vector to {task_vector_path}")
    
    # Log task vector statistics
    total_params = sum(p.numel() for p in base_model.parameters())
    task_vector_params = sum(v.numel() for v in task_vector.values())
    logger.info(f"Task vector contains {task_vector_params} parameters")
    logger.info(f"Task vector size: {task_vector_params / total_params * 100:.2f}% of base model")
    
    wandb.finish()
    logger.info(f"Stage 1 training completed for {args.axis}")

if __name__ == "__main__":
    main() 