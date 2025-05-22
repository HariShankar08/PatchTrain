# GSM-8K Fine-tuning with LoRA

This project implements fine-tuning of language models on the GSM-8K dataset using LoRA (Low-Rank Adaptation) for efficient parameter-efficient fine-tuning.

## Project Structure

```
src/
├── config/
│   └── config.py         # Configuration classes
├── data/
│   └── dataset.py        # Dataset processing
├── models/
│   └── model.py          # Model management and LoRA setup
├── training/
│   └── trainer.py        # Training setup and execution
├── evaluation/
│   └── evaluator.py      # Model evaluation
└── main.py              # Main execution script
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face authentication:
```bash
huggingface-cli login
```

## Usage

The script supports different modes of operation through command-line arguments:

### Command-line Options

- `--mode`: Choose the execution mode
  - `evaluate`: Only evaluate a pretrained model
  - `full_finetune`: Full model fine-tuning
  - `lora_finetune`: LoRA fine-tuning (default)
- `--model_path`: Path to a pretrained model for evaluation or continued training
- `--save_path`: Where to save the model/adapters (default: "./outputs")
- `--eval_only`: Skip training and only evaluate the model

### Usage Examples

1. **Evaluate a pretrained model**:
```bash
python src/main.py --mode evaluate --model_path path/to/model --eval_only
```

2. **Full fine-tuning**:
```bash
python src/main.py --mode full_finetune --save_path ./full_finetuned_model
```

3. **LoRA fine-tuning**:
```bash
python src/main.py --mode lora_finetune --save_path ./lora_adapters
```

4. **Continue training from a checkpoint**:
```bash
python src/main.py --mode lora_finetune --model_path path/to/checkpoint --save_path ./new_adapters
```

### What Each Mode Does

1. **Evaluate Mode**:
   - Loads a pretrained model
   - Evaluates on the GSM-8K test set
   - Reports exact match accuracy
   - Shows example predictions

2. **Full Fine-tuning Mode**:
   - Fine-tunes all model parameters
   - Saves the complete model
   - Evaluates performance after training

3. **LoRA Fine-tuning Mode**:
   - Applies LoRA for parameter-efficient fine-tuning
   - Only updates adapter parameters
   - Saves only the LoRA adapters
   - Evaluates performance after training

## Configuration

You can modify the training parameters in `src/config/config.py`:
- Model configuration (model name, LoRA rank, etc.)
- Training configuration (learning rate, epochs, etc.)
- Data configuration (prompt templates)
- Evaluation configuration (metrics, sampling)

## Output

The script will output:
- Training progress (if training)
- Evaluation metrics
- Example predictions
- Saved model or adapters in the specified output directory 
