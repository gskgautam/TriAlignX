# TriAlign

## 📚 Datasets

The following datasets can be accessed from their respective official sources:

- **Alpaca**  
  A strong instruction-following dataset built on top of the Stanford Self-Instruct method.  
  [Access Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

- **BeaverTails**  
  A challenging benchmark designed for evaluating long-context instruction tuning.  
  [Access BeaverTails](https://sites.google.com/view/pku-beavertails)

- **TruthfulQA**  
  A benchmark to measure whether language models produce truthful answers.  
  [Access TruthfulQA](https://github.com/sylinrl/TruthfulQA)


## 🧠 Instruction-Tuned Models

The following instruction-tuned large language models can be downloaded from Hugging Face:

- **LLaMA-2 7B**  
  [Download from Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf)

- **Mistral-7B**  
  [Download from Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1)

 - **Gemma-7B**  
  [Download from Hugging Face](https://huggingface.co/google/gemma-7b)

- **DeepSeek-7B**  
  [Download from Hugging Face](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)



## Module Usage

- **`data_processing/`**  
  Handles dataset loading and preprocessing to ensure consistency across experiments.

- **`evaluation/`**  
  Provides evaluation scripts (`evaluate.py`) for computing alignment scores, baselines, and comparisons.

- **`models/`**  
  Implements model architectures:
  - `multi_agent.py`: Encodes multiple-agent coordination for alignment tasks.  
  - `prefselect.py`: PrefSelect mechanism for query-aware policy selection.  
  - `trialignx.py`: Core TriAlignX implementation.  

- **`training/`**  
  Training pipeline split into stages:  
  - `stage1_finetuning.py`: Axis-wise low-rank fine-tuning.  
  - `stage2_trialignx.py`: Multi-agent steering and final decoding.

- **`utils.py`**  
  General utility functions shared across modules.


### 📈 Evaluation

After applying any of the MoCaE methods, use `Evaluate.py` to assess the performance of the calibrated models.

> ⚠️ Note: Make sure you have the appropriate access to the moderation models used for evaluation. These include:

- GPT-4.0 (via OpenAI API)
- beaver-dam-7b — available here: [PKU-Alignment/beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)
- GPT-Judge (via OpenAI API) 

These evaluators are used to provide automated and/or human-aligned judgment of the calibrated outputs in terms of helpfulness, harmlessness, and honesty.


