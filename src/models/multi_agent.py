"""
Multi-agent components for TriAlignX framework
Implements Encoding Agent and Response Agent for Stage 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class EncodingAgent(nn.Module):
    """
    Encoding Agent for processing task vectors and input embeddings
    """
    
    def __init__(self, hidden_size: int = 768, num_axes: int = 3, encoding_hidden_size: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_axes = num_axes
        self.encoding_hidden_size = encoding_hidden_size
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size + num_axes, encoding_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoding_hidden_size, encoding_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoding_hidden_size // 2, num_axes),
            nn.Sigmoid()
        )
        
        # Task vector processor
        self.task_processor = nn.Sequential(
            nn.Linear(hidden_size, encoding_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoding_hidden_size, num_axes)
        )
        
    def forward(self, V_star: List[torch.Tensor], x: torch.Tensor, 
                z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Encoding Agent
        
        Args:
            V_star: List of selected task vectors [num_selected, hidden_size]
            x: Input embeddings [batch_size, seq_len, hidden_size]
            z: Routing weights [batch_size, num_axes]
            
        Returns:
            r: Reward scores [batch_size, num_axes]
        """
        batch_size, seq_len, hidden_size = x.size()
        
        # Process task vectors
        if V_star:
            # Handle single task vector case
            if len(V_star) == 1:
                V_mean = V_star[0].unsqueeze(0)  # [1, hidden_size]
            else:
                # Concatenate all task vectors
                V_combined = torch.cat(V_star, dim=0)  # [total_params, hidden_size]
                V_mean = V_combined.mean(dim=0, keepdim=True)  # [1, hidden_size]
            
            V_processed = self.task_processor(V_mean)  # [1, num_axes]
            V_processed = V_processed.expand(batch_size, -1)  # [batch_size, num_axes]
        else:
            V_processed = torch.zeros(batch_size, self.num_axes, device=x.device)
        
        # Process input embeddings
        x_mean = x.mean(dim=1)  # [batch_size, hidden_size]
        
        # Combine task vectors and input embeddings
        combined = torch.cat([x_mean, V_processed], dim=-1)  # [batch_size, hidden_size + num_axes]
        
        # Encode to get reward scores
        r = self.encoder(combined)  # [batch_size, num_axes]
        
        return r

class ResponseAgent(nn.Module):
    """
    Response Agent for generating aligned responses
    """
    
    def __init__(self, hidden_size: int = 768, num_axes: int = 3, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_axes = num_axes
        
        # Alignment projector
        self.alignment_projector = nn.Sequential(
            nn.Linear(hidden_size + num_axes, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Cross-attention for reward integration
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Response Agent
        
        Args:
            r: Reward scores [batch_size, num_axes]
            x: Input embeddings [batch_size, seq_len, hidden_size]
            
        Returns:
            aligned_embeddings: Aligned embeddings [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.size()
        
        # Expand reward scores to match sequence length
        expanded_r = r.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, num_axes]
        
        # Concatenate embeddings with reward scores
        combined = torch.cat([x, expanded_r], dim=-1)  # [batch_size, seq_len, hidden_size + num_axes]
        
        # Project through alignment network
        aligned = self.alignment_projector(combined)  # [batch_size, seq_len, hidden_size]
        
        # Apply cross-attention
        aligned_norm = self.layer_norm1(aligned)
        attn_output, _ = self.cross_attention(
            aligned_norm, aligned_norm, aligned_norm
        )
        aligned = aligned + attn_output
        
        # Final layer normalization
        aligned = self.layer_norm2(aligned)
        
        return aligned

class MultiAgentEnvironment(nn.Module):
    """
    Complete multi-agent environment combining Encoding and Response agents
    """
    
    def __init__(self, hidden_size: int = 768, num_axes: int = 3, 
                 encoding_hidden_size: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_axes = num_axes
        
        # Initialize agents
        self.encoding_agent = EncodingAgent(
            hidden_size=hidden_size,
            num_axes=num_axes,
            encoding_hidden_size=encoding_hidden_size,
            dropout=dropout
        )
        
        self.response_agent = ResponseAgent(
            hidden_size=hidden_size,
            num_axes=num_axes,
            dropout=dropout
        )
        
        # Alignment consistency layer
        self.alignment_consistency = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_axes)
        )
        
    def forward(self, V_star: List[torch.Tensor], x: torch.Tensor, 
                z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-agent environment
        
        Args:
            V_star: Selected task vectors
            x: Input embeddings [batch_size, seq_len, hidden_size]
            z: Routing weights [batch_size, num_axes]
            
        Returns:
            aligned_embeddings: Aligned embeddings [batch_size, seq_len, hidden_size]
            r: Reward scores [batch_size, num_axes]
        """
        # Encoding Agent: process task vectors and input
        r = self.encoding_agent(V_star, x, z)
        
        # Response Agent: generate aligned embeddings
        aligned_embeddings = self.response_agent(r, x)
        
        return aligned_embeddings, r
    
    def compute_alignment_loss(self, aligned_embeddings: torch.Tensor, 
                             target_axes: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment consistency loss
        
        Args:
            aligned_embeddings: Aligned embeddings [batch_size, seq_len, hidden_size]
            target_axes: Target axis labels [batch_size, num_axes]
            
        Returns:
            alignment_loss: Alignment consistency loss
        """
        # Compute alignment scores
        alignment_scores = self.alignment_consistency(
            aligned_embeddings.mean(dim=1)  # [batch_size, hidden_size]
        )  # [batch_size, num_axes]
        
        # Compute binary cross-entropy loss
        alignment_loss = F.binary_cross_entropy_with_logits(
            alignment_scores, target_axes.float()
        )
        
        return alignment_loss
    
    def get_axis_predictions(self, aligned_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get axis predictions from aligned embeddings
        
        Args:
            aligned_embeddings: Aligned embeddings [batch_size, seq_len, hidden_size]
            
        Returns:
            axis_predictions: Predicted axis scores [batch_size, num_axes]
        """
        alignment_scores = self.alignment_consistency(
            aligned_embeddings.mean(dim=1)
        )
        return torch.sigmoid(alignment_scores) 