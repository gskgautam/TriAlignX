"""
Main TriAlignX model implementation
Combines base model, PrefSelect, and multi-agent environment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Tuple, Optional, Any

from .prefselect import PrefSelect, TaskVectorManager
from .multi_agent import MultiAgentEnvironment

class TriAlignX(nn.Module):
    """
    Complete TriAlignX model implementation
    """
    
    def __init__(self, config: Dict[str, Any], base_model_name: str = None):
        super().__init__()
        
        self.config = config
        self.base_model_name = base_model_name or config['model']['base_model']
        self.num_axes = 3
        self.axis_names = ['helpful', 'harmless', 'honest']
        
        # Setup Hugging Face token if available
        from utils import setup_huggingface_token
        setup_huggingface_token(config)
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # Get model dimensions
        self.hidden_size = self.base_model.config.hidden_size
        self.vocab_size = self.base_model.config.vocab_size
        
        # Initialize PrefSelect
        self.pref_select = PrefSelect(
            num_axes=self.num_axes,
            hidden_size=self.hidden_size,
            router_hidden_size=config['prefselect']['hidden_size'],
            lambda_reg=config['prefselect']['lambda_reg']
        )
        
        # Initialize multi-agent environment
        self.multi_agent = MultiAgentEnvironment(
            hidden_size=self.hidden_size,
            num_axes=self.num_axes,
            encoding_hidden_size=config['multi_agent']['encoding_agent']['hidden_size'],
            dropout=config['multi_agent']['encoding_agent']['dropout']
        )
        
        # Task vector manager
        self.task_vector_manager = TaskVectorManager(num_axes=self.num_axes)
        
        # Alignment probes for each axis
        self.alignment_probes = nn.ModuleDict({
            axis: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, 1),
                nn.Sigmoid()
            ) for axis in self.axis_names
        })
        
        # Initialize LoRA for fine-tuning
        self.setup_lora()
        
    def setup_lora(self):
        """Setup LoRA configuration"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias']
        )
        
        self.base_model = get_peft_model(self.base_model, lora_config)
        
    def extract_task_vector(self, finetuned_model_path: str, axis: str):
        """
        Extract task vector from fine-tuned model
        
        Args:
            finetuned_model_path: Path to fine-tuned model
            axis: Axis name
        """
        # Load fine-tuned model
        finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)
        
        # Compute delta between fine-tuned and base model
        base_state = self.base_model.state_dict()
        finetuned_state = finetuned_model.state_dict()
        
        delta = {}
        for key in base_state:
            if key in finetuned_state:
                delta[key] = finetuned_state[key] - base_state[key]
        
        # Store task vector
        self.task_vector_manager.add_task_vector(axis, delta)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None, axis_labels: Optional[torch.Tensor] = None,
                stage: str = "stage2") -> Dict[str, torch.Tensor]:
        """
        Forward pass of TriAlignX
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len]
            axis_labels: Axis labels [batch_size, num_axes]
            stage: Training stage ("stage1" or "stage2")
            
        Returns:
            outputs: Dictionary containing model outputs and losses
        """
        batch_size = input_ids.size(0)
        
        if stage == "stage1":
            # Stage 1: Standard fine-tuning
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            return outputs
        
        else:
            # Stage 2: Multi-agent training
            
            # Get base model outputs
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get hidden states from last layer
            hidden_states = base_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # PrefSelect: get routing weights
            z, select_loss = self.pref_select(
                hidden_states, 
                base_outputs.logits
            )
            
            # Get selected task vectors
            selected_vectors = self.task_vector_manager.get_selected_vectors(z)
            
            # Convert selected vectors to list of tensors for multi-agent
            V_star = []
            for axis, task_vector in selected_vectors.items():
                if task_vector:
                    # Flatten task vector parameters
                    flat_vector = torch.cat([v.flatten() for v in task_vector.values()])
                    V_star.append(flat_vector)
            
            # Multi-agent environment
            aligned_embeddings, r = self.multi_agent(V_star, hidden_states, z)
            
            # Create new inputs with aligned embeddings
            aligned_outputs = self.base_model(
                inputs_embeds=aligned_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Compute losses
            ce_loss = aligned_outputs.loss
            
            # Alignment consistency loss
            align_loss = 0.0
            if axis_labels is not None:
                align_loss = self.multi_agent.compute_alignment_loss(
                    aligned_embeddings, axis_labels
                )
            
            # Total loss
            total_loss = (
                ce_loss + 
                select_loss + 
                self.config['multi_agent']['beta_align'] * align_loss
            )
            
            return {
                'loss': total_loss,
                'ce_loss': ce_loss,
                'select_loss': select_loss,
                'align_loss': align_loss,
                'logits': aligned_outputs.logits,
                'routing_weights': z,
                'reward_scores': r,
                'aligned_embeddings': aligned_embeddings
            }
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                max_length: int = 100, temperature: float = 1.0,
                do_sample: bool = True) -> torch.Tensor:
        """
        Generate text with TriAlignX
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            generated_ids: Generated token IDs
        """
        self.eval()
        
        with torch.no_grad():
            # Get base model outputs
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get hidden states
            hidden_states = base_outputs.hidden_states[-1]
            
            # PrefSelect: get routing weights
            z, _ = self.pref_select(hidden_states, base_outputs.logits)
            
            # Get selected task vectors
            selected_vectors = self.task_vector_manager.get_selected_vectors(z)
            
            # Convert to list
            V_star = []
            for axis, task_vector in selected_vectors.items():
                if task_vector:
                    flat_vector = torch.cat([v.flatten() for v in task_vector.values()])
                    V_star.append(flat_vector)
            
            # Multi-agent environment
            aligned_embeddings, _ = self.multi_agent(V_star, hidden_states, z)
            
            # Generate with aligned embeddings
            generated_ids = self.base_model.generate(
                inputs_embeds=aligned_embeddings,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        return generated_ids
    
    def get_axis_predictions(self, input_ids: torch.Tensor, 
                           attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get axis predictions for input
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            predictions: Dictionary of axis predictions
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(input_ids, attention_mask, stage="stage2")
            
            # Get aligned embeddings
            aligned_embeddings = outputs['aligned_embeddings']
            
            # Get predictions from multi-agent
            axis_predictions = self.multi_agent.get_axis_predictions(aligned_embeddings)
            
            # Get individual axis predictions from probes
            probe_predictions = {}
            for i, axis in enumerate(self.axis_names):
                probe_output = self.alignment_probes[axis](
                    aligned_embeddings.mean(dim=1)
                )
                probe_predictions[axis] = probe_output.squeeze()
            
            return {
                'multi_agent': axis_predictions,
                'probes': probe_predictions
            }
    
    def save_model(self, path: str):
        """Save the complete TriAlignX model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'base_model_name': self.base_model_name,
            'task_vectors': self.task_vector_manager.task_vectors
        }, path)
    
    def load_model(self, path: str):
        """Load the complete TriAlignX model"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.task_vector_manager.task_vectors = checkpoint['task_vectors'] 