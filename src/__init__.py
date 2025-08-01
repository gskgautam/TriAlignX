"""
TriAlignX: A Two-Stage Framework for HHH Alignment
"""

from .models.trialignx import TriAlignX
from .models.prefselect import PrefSelect, TaskVectorManager
from .models.multi_agent import MultiAgentEnvironment, EncodingAgent, ResponseAgent
from .data_processing.dataset_loader import (
    TriAlignXDataset, BeaverTailsDataset, AlpacaDataset, TruthfulQADataset,
    load_beavertails_data, load_alpaca_data, load_truthfulqa_data,
    create_dataloaders, create_combined_dataloader
)
from .utils import load_config, setup_device, setup_logging, set_seed

__version__ = "1.0.0"
__author__ = "TriAlignX Team"

__all__ = [
    "TriAlignX",
    "PrefSelect", 
    "TaskVectorManager",
    "MultiAgentEnvironment",
    "EncodingAgent",
    "ResponseAgent",
    "TriAlignXDataset",
    "BeaverTailsDataset",
    "AlpacaDataset", 
    "TruthfulQADataset",
    "load_beavertails_data",
    "load_alpaca_data",
    "load_truthfulqa_data",
    "create_dataloaders",
    "create_combined_dataloader",
    "load_config",
    "setup_device",
    "setup_logging",
    "set_seed"
] 