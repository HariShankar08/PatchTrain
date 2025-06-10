from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config.config import TrainingConfig
import wandb
import os

wandb.login(key='3ec3e02fc75a1a05f6f949246341161384c0f57b')

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._wandb_run = None

    def _init_wandb(self, model_name: str, training_stage: str = None):
        """Initialize wandb if not already initialized."""
        if not self.config.use_wandb:
            return

        if self._wandb_run is None:
            # Initialize wandb only if no run exists
            run_name = self.config.wandb_run_name
            if run_name is None:
                run_name = f"{os.path.basename(model_name)}"
                if training_stage:
                    run_name += f"_{training_stage}"

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=run_name,
                group=self.config.wandb_group,
                resume="allow"
            )
        return self._wandb_run

    def setup_trainer(self, model, train_dataset, eval_dataset, tokenizer, batch_size, training_stage: str = None):
        """Setup the Hugging Face Trainer with the specified configuration."""
        # Initialize wandb if needed
        if self.config.use_wandb:
            self._init_wandb(model.config._name_or_path, training_stage)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
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
            # Enable wandb logging
            report_to="wandb" if self.config.use_wandb else None,
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

    def finish_wandb(self):
        """Finish the wandb run if it exists."""
        if self._wandb_run is not None:
            wandb.finish()
            self._wandb_run = None 