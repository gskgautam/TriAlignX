# TriAlignX: Three-Way Alignment Framework for LLMs

This repository implements the TriAlignX framework for aligning Large Language Models (LLMs) with HHH (Helpfulness, Harmlessness, Honesty) objectives as described in the AAAI_26.pdf paper.

## Overview

TriAlignX is a two-stage framework that:
1. **Stage 1**: Performs axis-specific fine-tuning and extracts task vectors
2. **Stage 2**: Implements a multi-agent environment for dynamic alignment

## Project Structure

```
HHH/
├── data/                    # Dataset storage
│   ├── Harmless/           # BeaverTails dataset
│   ├── Helpfull/           # Alpaca dataset  
│   └── honesty/            # TruthfulQA dataset
├── src/                    # Source code
│   ├── models/             # Model implementations
│   ├── data_processing/    # Dataset processing
│   ├── training/           # Training scripts
│   └── evaluation/         # Evaluation metrics
├── configs/                # Configuration files
├── logs/                   # Training logs
├── checkpoints/            # Model checkpoints
└── results/                # Evaluation results
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**:
   ```bash
   python src/data_processing/prepare_datasets.py
   ```

2. **Stage 1 - Axis-specific Fine-tuning**:
   ```bash
   python src/training/stage1_finetuning.py --axis helpful
   python src/training/stage1_finetuning.py --axis harmless
   python src/training/stage1_finetuning.py --axis honest
   ```

3. **Stage 2 - Multi-agent Training**:
   ```bash
   python src/training/stage2_trialignx.py
   ```

4. **Evaluation**:
   ```bash
   python src/evaluation/evaluate.py
   ```

## Configuration

Edit `configs/config.yaml` to modify hyperparameters and model settings.

## Results

Evaluation results are saved in `results/` directory with metrics for:
- Helpfulness (Win Rate)
- Harmlessness (Safety Score)  
- Honesty (Truthfulness × Informativeness)

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{trialignx2024,
  title={TriAlignX: Three-Way Alignment Framework for LLMs},
  author={...},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2024}
}
``` 