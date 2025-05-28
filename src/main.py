import argparse
from config.config import (
    ModelConfig, TrainingConfig, DataConfig, EvaluationConfig,
    RunConfig, RunMode
)
from data.dataset import GSM8KProcessor
from models.model import ModelManager
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="GSM-8K Model Training and Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in RunMode],
        default=RunMode.LORA_FINETUNE.value,
        help="Execution mode: evaluate, full_finetune, lora_finetune, or patch_train"
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
        "--lambda_ratio",
        type=float,
        default=2/3,
        help="Ratio of epochs to use patch training (e.g., 2/3 means use patch training for 2/3 of epochs)"
    )
    return parser.parse_args()

def evaluate_model(model, tokenizer, test_dataset, eval_config, device):
    """Helper function to evaluate model."""
    evaluator = ModelEvaluator(eval_config)
    metrics, predictions, references = evaluator.evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        device=device
    )
    print(f"\nMetrics: {metrics}")
    evaluator.print_examples(predictions, references)
    return metrics

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize configurations
    run_config = RunConfig(
        mode=RunMode(args.mode),
        model_path=args.model_path,
        save_path=args.save_path,
        eval_only=args.eval_only
    )
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    eval_config = EvaluationConfig()

    # Setup data processing
    print("Setting up data processing...")
    data_processor = GSM8KProcessor(data_config)
    tokenized_dataset = data_processor.preprocess_dataset(None)  # We'll tokenize after model loading
    test_dataset_for_evaluate = tokenized_dataset["test"]
    print(test_dataset_for_evaluate)
    print(test_dataset_for_evaluate[0])
    # exit()
    # Setup model
    print("Setting up model...")
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
        test_dataset = tokenized_dataset["test"]

        if run_config.mode == RunMode.LORA_FINETUNE:
            print("Setting up LoRA...")
            model = model_manager.setup_lora()
            
            # Setup training
            print("Setting up training...")
            trainer_manager = ModelTrainer(training_config)
            trainer = trainer_manager.setup_trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer
            )

            # Train model
            trainer = trainer_manager.train(trainer)

            # Save adapters
            print(f"Saving adapters to {run_config.save_path}...")
            model_manager.save_adapters(run_config.save_path)
            
        elif run_config.mode == RunMode.FULL_FINETUNE:
            print("Preparing for full fine-tuning...")
            # Setup training
            print("Setting up training...")
            trainer_manager = ModelTrainer(training_config)
            trainer = trainer_manager.setup_trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer
            )

            # Train model
            trainer = trainer_manager.train(trainer)

            # Save model
            print(f"Saving model to {run_config.save_path}...")
            trainer.save_model(run_config.save_path)
            
        elif run_config.mode == RunMode.PATCH_TRAIN:
            print("Preparing for patch training...")
            # Train with patch strategy
            print(f"Starting patch training with patch_size={args.patch_size}, lambda_ratio={args.lambda_ratio}")
            trainer, model = model_manager.train_patch_model(
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                patch_size=args.patch_size,
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
            )
            
            # Save model
            print(f"Saving model to {run_config.save_path}...")
            trainer.save_model(run_config.save_path)
            # model.save_pretrained(run_config.save_path)
            # tokenizer.save_pretrained(run_config.save_path)

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset_for_evaluate,
        eval_config=eval_config,
        device=model_manager.device
    )

if __name__ == "__main__":
    main() 
