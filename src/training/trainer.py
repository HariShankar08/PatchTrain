from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config.config import TrainingConfig

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def setup_trainer(self, model, train_dataset, eval_dataset, tokenizer, batch_size):
        """Setup the Hugging Face Trainer with the specified configuration."""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=4,  # This should come from model config
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            per_device_eval_batch_size=batch_size,
            per_device_train_batch_size=batch_size,
            # save_strategy=self.config.save_strategy,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        return trainer

    def train(self, trainer):
        """Execute the training process."""
        print("Starting training...")
        trainer.train()
        return trainer 
