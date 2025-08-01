"""
Stage 2: Multi-agent training for TriAlignX
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, setup_device, setup_logging, set_seed, save_checkpoint
from data_processing.dataset_loader import create_combined_dataloader
from models.trialignx import TriAlignX

def create_axis_labels(batch, axis_names):
    """Create axis labels for the batch"""
    batch_size = batch['input_ids'].size(0)
    axis_labels = torch.zeros(batch_size, len(axis_names))
    
    for i, axis in enumerate(axis_names):
        # Set label to 1 for the axis this sample belongs to
        axis_mask = [item['axis'] == axis for item in batch]
        axis_labels[axis_mask, i] = 1
    
    return axis_labels

def train_stage2(config, model, train_dataloader, val_dataloader, device, logger):
    """Train Stage 2: Multi-agent environment"""
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['stage2']['learning_rate'],
        weight_decay=config['training']['stage2']['weight_decay']
    )
    
    num_training_steps = len(train_dataloader) * config['training']['stage2']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['stage2']['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    axis_names = ['helpful', 'harmless', 'honest']
    
    for epoch in range(config['training']['stage2']['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['stage2']['num_epochs']}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_ce_loss = 0.0
        train_select_loss = 0.0
        train_align_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc="Training Stage 2")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Create axis labels
            axis_labels = create_axis_labels(batch, axis_names).to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                axis_labels=axis_labels,
                stage="stage2"
            )
            
            loss = outputs['loss']
            ce_loss = outputs['ce_loss']
            select_loss = outputs['select_loss']
            align_loss = outputs['align_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['stage2']['max_grad_norm']
            )
            
            optimizer.step()
            scheduler.step()
            
            # Accumulate losses
            train_loss += loss.item()
            train_ce_loss += ce_loss.item()
            train_select_loss += select_loss.item()
            train_align_loss += align_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'select': f"{select_loss.item():.4f}",
                'align': f"{align_loss.item():.4f}"
            })
            
            # Log to wandb
            if batch_idx % config['logging']['log_interval'] == 0:
                wandb.log({
                    'stage2_train_loss': loss.item(),
                    'stage2_ce_loss': ce_loss.item(),
                    'stage2_select_loss': select_loss.item(),
                    'stage2_align_loss': align_loss.item(),
                    'stage2_learning_rate': scheduler.get_last_lr()[0],
                    'stage2_epoch': epoch + batch_idx / len(train_dataloader)
                })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_ce_loss = 0.0
        val_select_loss = 0.0
        val_align_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating Stage 2"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                axis_labels = create_axis_labels(batch, axis_names).to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    axis_labels=axis_labels,
                    stage="stage2"
                )
                
                val_loss += outputs['loss'].item()
                val_ce_loss += outputs['ce_loss'].item()
                val_select_loss += outputs['select_loss'].item()
                val_align_loss += outputs['align_loss'].item()
        
        # Average losses
        val_loss /= len(val_dataloader)
        val_ce_loss /= len(val_dataloader)
        val_select_loss /= len(val_dataloader)
        val_align_loss /= len(val_dataloader)
        
        train_loss /= len(train_dataloader)
        train_ce_loss /= len(train_dataloader)
        train_select_loss /= len(train_dataloader)
        train_align_loss /= len(train_dataloader)
        
        logger.info(f"Epoch {epoch + 1}:")
        logger.info(f"  Train - Total: {train_loss:.4f}, CE: {train_ce_loss:.4f}, Select: {train_select_loss:.4f}, Align: {train_align_loss:.4f}")
        logger.info(f"  Val   - Total: {val_loss:.4f}, CE: {val_ce_loss:.4f}, Select: {val_select_loss:.4f}, Align: {val_align_loss:.4f}")
        
        # Log to wandb
        wandb.log({
            'stage2_epoch_train_loss': train_loss,
            'stage2_epoch_train_ce_loss': train_ce_loss,
            'stage2_epoch_train_select_loss': train_select_loss,
            'stage2_epoch_train_align_loss': train_align_loss,
            'stage2_epoch_val_loss': val_loss,
            'stage2_epoch_val_ce_loss': val_ce_loss,
            'stage2_epoch_val_select_loss': val_select_loss,
            'stage2_epoch_val_align_loss': val_align_loss,
            'stage2_epoch': epoch + 1
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(
                config['paths']['output_dir'], 
                "trialignx_stage2_best.pt"
            )
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)
            logger.info(f"Saved best model with val loss: {val_loss:.4f}")
    
    # Save final model
    final_save_path = os.path.join(
        config['paths']['output_dir'], 
        "trialignx_stage2_final.pt"
    )
    save_checkpoint(model, optimizer, config['training']['stage2']['num_epochs'] - 1, val_loss, final_save_path)
    
    return model

def load_task_vectors(config, model):
    """Load task vectors from Stage 1 training"""
    logger = setup_logging(config['paths']['logs_dir'])
    
    for axis in ['helpful', 'harmless', 'honest']:
        task_vector_path = os.path.join(
            config['paths']['output_dir'], 
            f"{axis}_task_vector.pt"
        )
        
        if os.path.exists(task_vector_path):
            task_vector = torch.load(task_vector_path, map_location='cpu')
            model.task_vector_manager.add_task_vector(axis, task_vector)
            logger.info(f"Loaded task vector for {axis}")
        else:
            logger.warning(f"Task vector not found for {axis}: {task_vector_path}")

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Multi-agent training")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Base model to use")
    parser.add_argument("--load_task_vectors", action="store_true",
                       help="Load task vectors from Stage 1")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup Hugging Face token
    from utils import setup_huggingface_token
    setup_huggingface_token(config)
    
    # Setup logging
    logger = setup_logging(config['paths']['logs_dir'])
    logger.info("Starting Stage 2: Multi-agent training")
    
    # Set seed
    set_seed(config['data']['random_seed'])
    
    # Setup device
    device = setup_device(config['hardware']['device'])
    logger.info(f"Using device: {device}")
    
    # Setup wandb
    wandb.init(
        project=config['logging']['project_name'],
        name="stage2_trialignx",
        config=config
    )
    
    # Create directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    # Load tokenizer and create dataloaders
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataloader = create_combined_dataloader(config, tokenizer)
    
    # Create validation dataloader (using a subset of training data for now)
    # In practice, you'd want a proper validation set
    val_dataloader = create_combined_dataloader(config, tokenizer)
    
    logger.info(f"Train samples: {len(train_dataloader.dataset)}")
    logger.info(f"Val samples: {len(val_dataloader.dataset)}")
    
    # Initialize TriAlignX model
    model = TriAlignX(config, args.base_model)
    
    # Load task vectors if requested
    if args.load_task_vectors:
        load_task_vectors(config, model)
    
    model.to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {trainable_params / total_params * 100:.2f}%")
    
    # Train the model
    trained_model = train_stage2(
        config, model, train_dataloader, val_dataloader, device, logger
    )
    
    # Save the complete model
    final_model_path = os.path.join(
        config['paths']['output_dir'], 
        "trialignx_complete.pt"
    )
    trained_model.save_model(final_model_path)
    logger.info(f"Saved complete TriAlignX model to {final_model_path}")
    
    wandb.finish()
    logger.info("Stage 2 training completed")

if __name__ == "__main__":
    main() 