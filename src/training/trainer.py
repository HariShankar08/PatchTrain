from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from config.config import TrainingConfig
import wandb
import os
import torch
import psutil

wandb.login(key='3ec3e02fc75a1a05f6f949246341161384c0f57b')

class GPUMemoryCallback(TrainerCallback):
    """Callback to log GPU and CPU memory usage to wandb."""
    def __init__(self):
        self.start_time = None

    def _get_gpu_memory(self):
        """Get GPU memory stats in GB."""
        if not torch.cuda.is_available():
            return None
        
        memory_stats = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # Convert to GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }
        return memory_stats

    def _get_cpu_memory(self):
        """Get CPU memory stats in GB."""
        process = psutil.Process(os.getpid())
        memory_stats = {
            'cpu_percent': process.cpu_percent(),
            'cpu_memory_gb': process.memory_info().rss / 1024**3
        }
        return memory_stats

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log memory usage along with other metrics."""
        if not logs:
            return
        
        # Get GPU memory stats
        gpu_memory = self._get_gpu_memory()
        if gpu_memory:
            logs.update({
                'gpu_memory/allocated_gb': gpu_memory['allocated'],
                'gpu_memory/reserved_gb': gpu_memory['reserved'],
                'gpu_memory/max_allocated_gb': gpu_memory['max_allocated']
            })
        
        # Get CPU stats
        cpu_stats = self._get_cpu_memory()
        logs.update({
            'cpu/percent': cpu_stats['cpu_percent'],
            'cpu/memory_gb': cpu_stats['cpu_memory_gb']
        })

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
                resume="allow",
                # Configure custom charts
                config={
                    "custom_charts": {
                        "gpu_memory": {
                            "metrics": ["gpu_memory/allocated_gb", "gpu_memory/reserved_gb", "gpu_memory/max_allocated_gb"],
                            "title": "GPU Memory Usage"
                        },
                        "cpu_usage": {
                            "metrics": ["cpu/percent", "cpu/memory_gb"],
                            "title": "CPU Usage"
                        }
                    }
                }
            )
            
            # Define metrics for step alignment
            wandb.define_metric("gpu_memory/allocated_gb", step_metric="train/global_step")
            wandb.define_metric("gpu_memory/reserved_gb", step_metric="train/global_step")
            wandb.define_metric("gpu_memory/max_allocated_gb", step_metric="train/global_step")
            wandb.define_metric("cpu/percent", step_metric="train/global_step")
            wandb.define_metric("cpu/memory_gb", step_metric="train/global_step")
            
            # Log initial GPU memory state as a summary
            if torch.cuda.is_available():
                wandb.run.summary.update({
                    'gpu_info/device_name': torch.cuda.get_device_name(0),
                    'gpu_info/device_count': torch.cuda.device_count(),
                    'gpu_info/initial_memory_gb': torch.cuda.memory_allocated() / 1024**3
                })
                
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
            # Disable saving checkpoints during training
            save_strategy="no",
            # Enable wandb logging
            report_to="wandb" if self.config.use_wandb else None,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        callbacks = []
        if self.config.use_wandb:
            callbacks.append(GPUMemoryCallback())

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
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
            # Log final GPU memory state
            if torch.cuda.is_available():
                wandb.log({
                    'gpu_info/final_memory_gb': torch.cuda.memory_allocated() / 1024**3,
                    'gpu_info/max_memory_gb': torch.cuda.max_memory_allocated() / 1024**3
                })
            wandb.finish()
            self._wandb_run = None 