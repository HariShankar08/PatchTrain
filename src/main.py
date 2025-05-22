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
        help="Execution mode: evaluate, full_finetune, or lora_finetune"
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
    print(f"\nExact Match Score: {metrics['exact_match']:.2%}")
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
    test_dataset = tokenized_dataset["test"]

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

    # Retokenize dataset with the correct tokenizer
    tokenized_dataset = data_processor.preprocess_dataset(tokenizer)
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]

    if not run_config.eval_only:
        if run_config.mode == RunMode.LORA_FINETUNE:
            print("Setting up LoRA...")
            model = model_manager.setup_lora()
        elif run_config.mode == RunMode.FULL_FINETUNE:
            print("Preparing for full fine-tuning...")
            # No special setup needed for full fine-tuning
            pass

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

        # Save model/adapters
        print(f"Saving to {run_config.save_path}...")
        if run_config.mode == RunMode.LORA_FINETUNE:
            model_manager.save_adapters(run_config.save_path)
        else:
            trainer.save_model(run_config.save_path)

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        eval_config=eval_config,
        device=model_manager.device
    )

if __name__ == "__main__":
    main() 