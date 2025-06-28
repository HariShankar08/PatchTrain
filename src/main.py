import argparse
import numpy as np
from config.config import (
    ModelConfig, TrainingConfig, DatasetConfig, 
    EvaluationConfig, RunConfig, RunMode
)
from data.dataset import CNNProcessor, WMTProcessor
from models.model import ModelManager
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
import random
import torch
from transformers import set_seed
import csv
import os
import json
from datetime import datetime

from huggingface_hub import login

key = ''
if key:
    login(token=key)  # Login to Hugging Face Hub if key is provided

def parse_args():
    parser = argparse.ArgumentParser(description="Model Training and Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in RunMode],
        default=RunMode.LORA_FINETUNE.value,
        help="Execution mode: evaluate, full_finetune, lora_finetune, patch_train, or patch_peft"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model for evaluation or continued training"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./outputs",
        help="Path to save the model/adapters"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only evaluate the model"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=4,
        help="Size of patches for patch training"
    )
    parser.add_argument(
        "--patch_method",
        type=str,
        choices=["mean", "paraMean"],
        default="mean",
        help="Method to use for patch calculation"
    )
    parser.add_argument(
        "--lambda_ratio",
        type=float,
        default=2/3,
        help="Ratio of training batches to use patch training"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of training runs with different seeds"
    )
    parser.add_argument(
        "--patch_epochs",
        type=int,
        default=None,
        help="Number of epochs to use patch training"
    )
    parser.add_argument(
        "--standard_epochs",
        type=int,
        default=None,
        help="Number of epochs to use standard training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cnn",
        choices=["cnn", "wmt_hi", "wmt_fr"],
        help="Dataset to use for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    
    return parser.parse_args()

def get_dataset_config(args):
    """Get the appropriate dataset configuration based on the dataset argument."""
    if args.dataset == "cnn":
        return DatasetConfig(
            dataset_name="abisee/cnn_dailymail",
            prompt_template="Summarize the following article:\n{article}\n\nSummary:",
            answer_template="{highlights}"
        )
    elif args.dataset == "wmt_hi":
        return DatasetConfig(
            dataset_name="wmt/wmt14",
            subset="hi-en",
            prompt_template="Translate from Hindi to English:\n{source_text}\n\nEnglish:",
            answer_template="{target_text}"
        )
    elif args.dataset == "wmt_fr":
        return DatasetConfig(
            dataset_name="wmt/wmt14",
            subset="fr-en",
            prompt_template="Translate from French to English:\n{source_text}\n\nEnglish:",
            answer_template="{target_text}"
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

def evaluate_model(model, tokenizer, test_dataset, eval_config):
    """Helper function to evaluate model."""
    evaluator = ModelEvaluator(eval_config)
    metrics, predictions, references = evaluator.evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset
    )
    return metrics

def run_training_iteration(run_config, model_config, training_config, data_config, eval_config, seed, args):
    """Run a single training iteration with the given seed."""
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)  # Transformers specific
    
    # Update training config with seed and save path
    training_config.seed = seed
    training_config.save_path = f"{run_config.save_path}"
    
    # Setup data processing
    print(f"\nSetting up data processing for run {seed}...")
    if args.dataset == "cnn":
        data_processor = CNNProcessor(data_config)
    elif args.dataset in ["wmt_hi", "wmt_fr"]:
        data_processor = WMTProcessor(data_config)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    tokenized_dataset = data_processor.preprocess_dataset(None)  # We'll tokenize after model loading
    test_dataset = tokenized_dataset["test"]  # Always use test split for evaluation
    
    # Setup model
    print(f"Setting up model for run {seed}...")
    model_manager = ModelManager(model_config)
    
    if run_config.model_path:
        # Load existing model
        print(f"Loading model from {run_config.model_path}...")
        model, tokenizer = model_manager.load_model_and_tokenizer(run_config.model_path)
    else:
        # Load base model
        model, tokenizer = model_manager.load_model_and_tokenizer()

    if not run_config.eval_only:
        # Retokenize dataset with the correct tokenizer
        tokenized_dataset = data_processor.preprocess_dataset(tokenizer)
        train_dataset = tokenized_dataset["train"]
        val_dataset = tokenized_dataset["validation"]  # Use validation for training evaluation

        if run_config.mode == RunMode.LORA_FINETUNE:
            print(f"Setting up LoRA for run {seed}...")
            model = model_manager.setup_lora()
            
            # Setup training
            print("Setting up training...")
            trainer_manager = ModelTrainer(training_config)
            trainer = trainer_manager.setup_trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,  # Use validation set during training
                tokenizer=tokenizer,
                batch_size=args.batch_size
            )

            # Train model
            trainer, train_time = trainer_manager.train(trainer)
            
        elif run_config.mode == RunMode.FULL_FINETUNE:
            print("Preparing for full fine-tuning...")
            # Setup training
            print("Setting up training for run {seed}...")
            trainer_manager = ModelTrainer(training_config)
            trainer = trainer_manager.setup_trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,  # Use validation set during training
                tokenizer=tokenizer,
                batch_size=args.batch_size
            )

            # Train model
            trainer, train_time = trainer_manager.train(trainer)
            
        elif run_config.mode == RunMode.PATCH_TRAIN:
            print(f"Starting patch training for run {seed}...")
            trainer, model, train_time = model_manager.train_patch_model(
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                patch_size=args.patch_size,
                patch_calculation_method=args.patch_method,
                lambda_ratio=args.lambda_ratio,
                patch_epochs=args.patch_epochs,
                standard_epochs=args.standard_epochs,
                batch_size=args.batch_size,
                training_config=training_config
            )
        
        elif run_config.mode == RunMode.PATCH_PEFT:
            print("Preparing for patch PEFT training...")
            # Train with patch PEFT strategy
            print(f"Starting patch PEFT training with patch_size={args.patch_size}, lambda_ratio={args.lambda_ratio}")
            trainer, model, train_time = model_manager.train_patch_peft(
                train_dataset=train_dataset,
                eval_dataset=val_dataset,  # Use validation set during training
                patch_size=args.patch_size,
                patch_calculation_method=args.patch_method,
                lambda_ratio=args.lambda_ratio,
                num_epochs=training_config.num_train_epochs,
                learning_rate=training_config.learning_rate,
                warmup_steps=training_config.warmup_steps,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                fp16=training_config.fp16,
                bf16=training_config.bf16,
                logging_steps=training_config.logging_steps,
                eval_strategy=training_config.eval_strategy,
                eval_steps=training_config.eval_steps,
                save_strategy=training_config.save_strategy,
                patch_epochs=args.patch_epochs,
                standard_epochs=args.standard_epochs,
                batch_size=args.batch_size
            )

    # Final evaluation on test set
    print(f"Evaluating model on test set for run {seed}...")
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,  # Use test set for final evaluation
        eval_config=eval_config
    )
    
    # Add training time to metrics
    if not run_config.eval_only:
        metrics['train_time'] = train_time
    
    return metrics

def format_metric_stats(mean, std):
    """Format metric statistics as 'mean +/- std'."""
    return f"{mean:.4f} +/- {std:.4f}"

def save_metrics_to_csv(metrics_data, csv_file="training_metrics.csv"):
    """Save metrics data to CSV file."""
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_file)
    
    # Prepare the row data
    row_data = {
        'Model': metrics_data['model_name'],
        'LoRA': 'Yes' if metrics_data['lora'] else 'No',
        'RunMode': metrics_data['run_mode'],
        'Patch Size': metrics_data['patch_size'] if metrics_data['patch_size'] is not None else '',
        'Lambda': metrics_data['lambda_ratio'] if metrics_data['lambda_ratio'] is not None else '',
    }
    
    # Add metrics as a JSON string
    metrics_dict = {}
    for metric_name in metrics_data['metrics'].keys():
        mean = metrics_data['metrics'][metric_name]['mean']
        std = metrics_data['metrics'][metric_name]['std']
        metrics_dict[metric_name] = format_metric_stats(mean, std)
    row_data['Metrics'] = json.dumps(metrics_dict)
    
    # Write to CSV
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'LoRA', 'RunMode', 'Patch Size', 'Lambda', 'Metrics'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create configurations
    run_config = RunConfig(
        mode=RunMode(args.mode),
        model_path=args.model_path,
        save_path=args.save_path,
        eval_only=args.eval_only
    )
    
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = get_dataset_config(args)
    eval_config = EvaluationConfig()
    
    # Run training iterations
    all_metrics = []
    for i in range(args.num_runs):
        seed = i  # Use different seeds for each run
        metrics = run_training_iteration(
            run_config=run_config,
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            eval_config=eval_config,
            seed=seed,
            args=args
        )
        all_metrics.append(metrics)
        
        # Print metrics for this run
        print(f"\nMetrics for run {i+1} (seed {seed}):")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print('==============================================')

    # Calculate and print aggregate statistics
    if args.num_runs > 1:
        print("\nAggregate Statistics:")
        aggregate_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            mean = np.mean(values)
            std = np.std(values)
            aggregate_metrics[metric_name] = {'mean': mean, 'std': std}
            print(f"{metric_name}:")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std:  {std:.4f}")
        
        # Prepare metrics data for CSV
        metrics_data = {
            'model_name': args.model_path,
            'lora': run_config.mode == RunMode.LORA_FINETUNE or run_config.mode == RunMode.PATCH_PEFT,
            'run_mode': run_config.mode.value,
            'patch_size': args.patch_size if run_config.mode in [RunMode.PATCH_TRAIN, RunMode.PATCH_PEFT] else None,
            'lambda_ratio': args.lambda_ratio if run_config.mode in [RunMode.PATCH_TRAIN, RunMode.PATCH_PEFT] else None,
            'metrics': aggregate_metrics
        }
        
        # Save to CSV
        save_metrics_to_csv(metrics_data)

if __name__ == "__main__":
    main() 
