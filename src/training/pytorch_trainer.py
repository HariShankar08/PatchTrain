import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler
import wandb
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Any, Union
from .trainer import GPUMemoryCallback

class PyTorchTrainer:
    def __init__(self, config):
        self.config = config
        self._wandb_run = None
        self.gpu_memory_callback = GPUMemoryCallback()

    def _init_wandb(self, model_name: str, training_stage: str = None):
        """Initialize wandb if not already initialized."""
        if not self.config.use_wandb:
            return

        if self._wandb_run is None:
            # Initialize wandb only if no run exists
            run_name = self.config.wandb_run_name
            if run_name is None:
                # Create a descriptive run name
                base_name = os.path.basename(model_name)
                run_name = f"{base_name}"
                
                # Add training stage if provided
                if training_stage:
                    run_name += f"_{training_stage}"
                
                # Add seed for uniqueness
                if hasattr(self.config, 'seed'):
                    run_name += f"_seed{self.config.seed}"
                
                # Add timestamp for absolute uniqueness
                timestamp = datetime.now().strftime("%m%d_%H%M")
                run_name += f"_{timestamp}"

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=run_name,
                group=self.config.wandb_group,
                resume="allow",
                config={
                    "architecture": base_name,
                    "training_stage": training_stage,
                    "seed": getattr(self.config, 'seed', None),
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.num_train_epochs,
                    "batch_size": self.config.per_device_train_batch_size,
                }
            )

    def create_dataloaders(
        self,
        train_dataset,
        eval_dataset,
        batch_size: int,
        collate_fn=None
    ):
        """Create training and evaluation dataloaders."""
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        eval_dataloader = None
        if eval_dataset is not None:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
        return train_dataloader, eval_dataloader

    def train(
        self,
        model: nn.Module,
        train_dataset,
        eval_dataset=None,
        collate_fn=None,
        training_stage: str = None,
        max_steps: Optional[int] = None,
    ):
        """Execute the training process using PyTorch."""
        print("Starting PyTorch training...")
        
        # Initialize wandb
        if self.config.use_wandb:
            self._init_wandb(model.config._name_or_path, training_stage)

        # Create dataloaders
        train_dataloader, eval_dataloader = self.create_dataloaders(
            train_dataset,
            eval_dataset,
            self.config.per_device_train_batch_size,
            collate_fn
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        # Setup learning rate scheduler
        num_training_steps = len(train_dataloader) * self.config.num_train_epochs
        if max_steps is not None:
            num_training_steps = min(num_training_steps, max_steps)
            
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )

        # Training loop
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(int(self.config.num_train_epochs)):
            model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(model.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Log metrics
                if self.config.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    })
                    self.gpu_memory_callback.on_step_end(global_step)
                
                global_step += 1
                
                if max_steps is not None and global_step >= max_steps:
                    break
                    
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss}")
            
            # Evaluation
            if eval_dataloader is not None and self.config.eval_strategy == "epoch":
                eval_loss = self.evaluate(model, eval_dataloader)
                print(f"Evaluation loss: {eval_loss}")
                
                if self.config.use_wandb:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/epoch": epoch + 1,
                    })
                    
                # Save best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if hasattr(model, "save_pretrained"):
                        model.save_pretrained(f"{self.config.output_dir}/best_model")
            
            if max_steps is not None and global_step >= max_steps:
                break
                
        return model

    def evaluate(self, model: nn.Module, eval_dataloader: DataLoader) -> float:
        """Evaluate the model on the evaluation dataset."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
        return total_loss / len(eval_dataloader)

    def finish_wandb(self, model_path: str = None):
        """Finish the wandb run and save model artifact if specified."""
        if not self.config.use_wandb or self._wandb_run is None:
            return
            
        if model_path:
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description="Trained model weights and config"
            )
            artifact.add_dir(model_path)
            wandb.log_artifact(artifact)
            
        wandb.finish() 