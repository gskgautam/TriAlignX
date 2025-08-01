"""
PrefSelect module for TriAlignX framework
Implements the routing mechanism for task vector selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class PrefSelect(nn.Module):
    """
    PrefSelect module for routing and selecting appropriate task vectors
    """
    
    def __init__(self, num_axes: int = 3, hidden_size: int = 768, 
                 router_hidden_size: int = 256, lambda_reg: float = 0.1):
        super().__init__()
        
        self.num_axes = num_axes
        self.hidden_size = hidden_size
        self.lambda_reg = lambda_reg
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_size, router_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(router_hidden_size, router_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(router_hidden_size // 2, num_axes),
            nn.Sigmoid()
        )
        
        # Temperature parameter for KL divergence
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, reference_logits: torch.Tensor, 
                current_logits: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PrefSelect
        
        Args:
            x: Input embeddings [batch_size, seq_len, hidden_size]
            reference_logits: Reference logits from base model [batch_size, vocab_size]
            current_logits: Current logits (optional) [batch_size, vocab_size]
            
        Returns:
            z: Routing weights [batch_size, num_axes]
            loss: Regularization loss
        """
        batch_size = x.size(0)
        
        # Compute mean pooling over sequence length
        x_mean = x.mean(dim=1)  # [batch_size, hidden_size]
        
        # Get routing weights
        z = self.router(x_mean)  # [batch_size, num_axes]
        
        # Initialize loss
        total_loss = 0.0
        
        # KL divergence penalty if current_logits provided
        if current_logits is not None:
            kl_loss = F.kl_div(
                F.log_softmax(current_logits / self.temperature, dim=-1),
                F.softmax(reference_logits / self.temperature, dim=-1),
                reduction='batchmean'
            )
            total_loss += kl_loss
        
        # Sparsity regularization
        sparsity_loss = self.lambda_reg * torch.norm(z, p=1, dim=1).mean()
        total_loss += sparsity_loss
        
        # Entropy regularization to encourage clear routing decisions
        entropy_loss = -torch.mean(
            torch.sum(z * torch.log(z + 1e-8), dim=1)
        )
        total_loss += 0.01 * entropy_loss
        
        return z, total_loss
    
    def get_selected_axes(self, z: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get selected axes based on routing weights
        
        Args:
            z: Routing weights [batch_size, num_axes]
            threshold: Threshold for selection
            
        Returns:
            selected: Boolean tensor indicating selected axes [batch_size, num_axes]
        """
        return z > threshold
    
    def get_axis_weights(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get normalized axis weights
        
        Args:
            z: Routing weights [batch_size, num_axes]
            
        Returns:
            weights: Normalized weights [batch_size, num_axes]
        """
        return F.softmax(z, dim=-1)

class TaskVectorManager:
    """
    Manages task vectors for different axes
    """
    
    def __init__(self, num_axes: int = 3):
        self.num_axes = num_axes
        self.task_vectors = {}
        self.axis_names = ['helpful', 'harmless', 'honest']
        
    def add_task_vector(self, axis: str, task_vector: Dict[str, torch.Tensor]):
        """
        Add task vector for a specific axis
        
        Args:
            axis: Axis name ('helpful', 'harmless', 'honest')
            task_vector: Dictionary of parameter deltas
        """
        if axis in self.axis_names:
            self.task_vectors[axis] = task_vector
    
    def get_task_vector(self, axis: str) -> Dict[str, torch.Tensor]:
        """
        Get task vector for a specific axis
        
        Args:
            axis: Axis name
            
        Returns:
            task_vector: Dictionary of parameter deltas
        """
        return self.task_vectors.get(axis, {})
    
    def get_selected_vectors(self, z: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Get selected task vectors based on routing weights
        
        Args:
            z: Routing weights [batch_size, num_axes]
            threshold: Threshold for selection
            
        Returns:
            selected_vectors: Dictionary of selected task vectors
        """
        selected_axes = z > threshold
        
        selected_vectors = {}
        for i, axis in enumerate(self.axis_names):
            if selected_axes[:, i].any():
                selected_vectors[axis] = self.task_vectors.get(axis, {})
        
        return selected_vectors
    
    def apply_task_vectors(self, base_state: Dict[str, torch.Tensor], 
                          selected_vectors: Dict[str, torch.Tensor],
                          z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply selected task vectors to base model state
        
        Args:
            base_state: Base model state dict
            selected_vectors: Selected task vectors
            z: Routing weights [batch_size, num_axes]
            
        Returns:
            modified_state: Modified state dict
        """
        modified_state = base_state.copy()
        
        # Get axis weights
        axis_weights = F.softmax(z, dim=-1)  # [batch_size, num_axes]
        
        # Average weights across batch
        avg_weights = axis_weights.mean(dim=0)  # [num_axes]
        
        for i, axis in enumerate(self.axis_names):
            if axis in selected_vectors and avg_weights[i] > 0:
                task_vector = selected_vectors[axis]
                weight = avg_weights[i]
                
                for param_name, delta in task_vector.items():
                    if param_name in modified_state:
                        modified_state[param_name] = (
                            modified_state[param_name] + weight * delta
                        )
        
        return modified_state 