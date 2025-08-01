"""
Dataset loader for TriAlignX framework
Handles BeaverTails (Harmlessness), Alpaca (Helpfulness), and TruthfulQA (Honesty)
"""

import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import ast
import numpy as np
from sklearn.model_selection import train_test_split

class TriAlignXDataset(Dataset):
    """Base dataset class for TriAlignX"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        raise NotImplementedError

class BeaverTailsDataset(TriAlignXDataset):
    """Dataset for BeaverTails (Harmlessness)"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        super().__init__(data, tokenizer, max_length)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input as instruction-response pair
        prompt = item['prompt']
        response = item['response']
        is_safe = item['is_safe']
        
        # Create input text
        input_text = f"Instruction: {prompt}\nResponse: {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (1 for safe, 0 for unsafe)
        labels = torch.tensor(1 if is_safe else 0, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
            'axis': 'harmless',
            'original_text': input_text
        }

class AlpacaDataset(TriAlignXDataset):
    """Dataset for Alpaca (Helpfulness)"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        super().__init__(data, tokenizer, max_length)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input as instruction-response pair
        instruction = item['instruction']
        output = item['output']
        
        # Create input text
        input_text = f"Instruction: {instruction}\nResponse: {output}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For helpfulness, we use the instruction as target for generation
        target_text = f"Instruction: {instruction}\nResponse:"
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'axis': 'helpful',
            'original_text': input_text
        }

class TruthfulQADataset(TriAlignXDataset):
    """Dataset for TruthfulQA (Honesty)"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        super().__init__(data, tokenizer, max_length)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input as question-answer pair
        question = item['question']
        answer = item['answer']
        label = item['label']  # 1 for truthful, 0 for misleading
        
        # Create input text
        input_text = f"Question: {question}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels
        labels = torch.tensor(label, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
            'axis': 'honest',
            'original_text': input_text
        }

def load_beavertails_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load BeaverTails dataset"""
    df = pd.read_csv(file_path)
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    data = []
    for _, row in df.iterrows():
        try:
            # Parse the category dictionary
            category_dict = ast.literal_eval(row['category'])
            
            data.append({
                'prompt': row['prompt'],
                'response': row['response'],
                'is_safe': row['is_safe'],
                'category': category_dict
            })
        except:
            continue
    
    return data

def load_alpaca_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load Alpaca dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data

def load_truthfulqa_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load TruthfulQA dataset"""
    df = pd.read_csv(file_path)
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    data = []
    for _, row in df.iterrows():
        data.append({
            'question': row['question'],
            'answer': row['answer'],
            'label': row['label']
        })
    
    return data

def create_dataloaders(config: Dict, tokenizer) -> Dict[str, DataLoader]:
    """Create dataloaders for all axes"""
    
    # Load datasets
    max_samples = config['data']['max_samples_per_axis']
    
    # BeaverTails (Harmlessness)
    beavertails_train = load_beavertails_data(
        "Harmless/BeaverTails_Train.csv", 
        max_samples
    )
    beavertails_test = load_beavertails_data(
        "Harmless/BeaverTails_Test.csv", 
        max_samples // 4
    )
    
    # Alpaca (Helpfulness)
    alpaca_train = load_alpaca_data(
        "Helpfull/Alpaca_Train.json", 
        max_samples
    )
    alpaca_test = load_alpaca_data(
        "Helpfull/Alpaca_Test.json", 
        max_samples // 4
    )
    
    # TruthfulQA (Honesty)
    truthfulqa_train = load_truthfulqa_data(
        "honesty/TruthfulQA_Train.csv", 
        max_samples
    )
    truthfulqa_test = load_truthfulqa_data(
        "honesty/TruthfulQA_Test.csv", 
        max_samples // 4
    )
    
    # Create datasets
    datasets = {
        'harmless': {
            'train': BeaverTailsDataset(beavertails_train, tokenizer, config['model']['max_length']),
            'test': BeaverTailsDataset(beavertails_test, tokenizer, config['model']['max_length'])
        },
        'helpful': {
            'train': AlpacaDataset(alpaca_train, tokenizer, config['model']['max_length']),
            'test': AlpacaDataset(alpaca_test, tokenizer, config['model']['max_length'])
        },
        'honest': {
            'train': TruthfulQADataset(truthfulqa_train, tokenizer, config['model']['max_length']),
            'test': TruthfulQADataset(truthfulqa_test, tokenizer, config['model']['max_length'])
        }
    }
    
    # Create dataloaders
    dataloaders = {}
    for axis in ['harmless', 'helpful', 'honest']:
        dataloaders[axis] = {
            'train': DataLoader(
                datasets[axis]['train'],
                batch_size=config['training']['stage1']['batch_size'],
                shuffle=True,
                num_workers=0
            ),
            'test': DataLoader(
                datasets[axis]['test'],
                batch_size=config['training']['stage1']['batch_size'],
                shuffle=False,
                num_workers=0
            )
        }
    
    return dataloaders

def create_combined_dataloader(config: Dict, tokenizer) -> DataLoader:
    """Create combined dataloader for Stage 2 training"""
    
    # Load all datasets
    max_samples = config['data']['max_samples_per_axis']
    
    beavertails = load_beavertails_data("Harmless/BeaverTails_Train.csv", max_samples)
    alpaca = load_alpaca_data("Helpfull/Alpaca_Train.json", max_samples)
    truthfulqa = load_truthfulqa_data("honesty/TruthfulQA_Train.csv", max_samples)
    
    # Create combined dataset
    class CombinedDataset(Dataset):
        def __init__(self, beavertails_data, alpaca_data, truthfulqa_data, tokenizer, max_length):
            self.beavertails = BeaverTailsDataset(beavertails_data, tokenizer, max_length)
            self.alpaca = AlpacaDataset(alpaca_data, tokenizer, max_length)
            self.truthfulqa = TruthfulQADataset(truthfulqa_data, tokenizer, max_length)
            
            # Create indices for balanced sampling
            self.indices = []
            for i in range(len(beavertails_data)):
                self.indices.append(('harmless', i))
            for i in range(len(alpaca_data)):
                self.indices.append(('helpful', i))
            for i in range(len(truthfulqa_data)):
                self.indices.append(('honest', i))
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            axis, data_idx = self.indices[idx]
            if axis == 'harmless':
                return self.beavertails[data_idx]
            elif axis == 'helpful':
                return self.alpaca[data_idx]
            else:
                return self.truthfulqa[data_idx]
    
    combined_dataset = CombinedDataset(
        beavertails, alpaca, truthfulqa, 
        tokenizer, config['model']['max_length']
    )
    
    return DataLoader(
        combined_dataset,
        batch_size=config['training']['stage2']['batch_size'],
        shuffle=True,
        num_workers=0
    ) 