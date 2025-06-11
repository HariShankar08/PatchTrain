import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from config.config import ModelConfig, TrainingConfig
from .patch_model import PatchTrainModel
from typing import Union, Optional
from training.trainer import ModelTrainer, GPUMemoryCallback

class ModelManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        # Determine the device
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def load_model_and_tokenizer(self, model_path: str = None):
        """Load the base model and tokenizer."""
        # Load tokenizer
        if model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if model_path:
            try:
                # Try loading as a PEFT model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float32 if self.device == "mps" else torch.bfloat16,
                )
                base_model = base_model.to(self.device)
                self.model = PeftModel.from_pretrained(
                    base_model,
                    model_path
                )
            except:
                # If not a PEFT model, load as regular model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32 if self.device == "mps" else torch.bfloat16,
                )
                self.model = self.model.to(self.device)
        else:
            if self.config.use_4bit and self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    load_in_4bit=True,
                    torch_dtype=torch.float32 if self.device == "mps" else torch.bfloat16,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float32 if self.device == "mps" else torch.bfloat16,
                )
                self.model = self.model.to(self.device)

        return self.model, self.tokenizer

    def setup_lora(self):
        """Configure and apply LoRA to the model."""
        if self.model is None:
            self.load_model_and_tokenizer()

        # If model is already a PEFT model, return it
        if hasattr(self.model, "peft_config"):
            return self.model

        # Prepare model for k-bit training if using 4-bit
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        peft_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        return self.model

    def train_patch_model(
        self,
        train_dataset,
        eval_dataset=None,
        patch_size: int = 4,
        patch_calculation_method: str = "paraMean",
        lambda_ratio: float = 2/3,
        num_epochs: int = 3,
        patch_epochs: Union[int, None] = None,
        standard_epochs: Union[int, None] = None,
        batch_size: int = 8,
        training_config: Optional[TrainingConfig] = None,
        **trainer_kwargs
    ):
        """
        Train a model using the patch training strategy.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            patch_size: Size of patches for patch training
            patch_calculation_method: Method to calculate patch values
            lambda_ratio: Ratio of training to use patch training
            num_epochs: Total number of epochs
            patch_epochs: Number of epochs for patch training (overrides lambda_ratio)
            standard_epochs: Number of epochs for standard training (overrides lambda_ratio)
            batch_size: Batch size for training
            training_config: Configuration for training (if None, a default config will be used)
            **trainer_kwargs: Additional arguments to pass to the Trainer
        """
        if training_config is None:
            training_config = TrainingConfig()

        # Calculate number of epochs for each phase
        if patch_epochs is None:
            patch_epochs = int(num_epochs * lambda_ratio)
        if standard_epochs is None:
            standard_epochs = num_epochs - patch_epochs

        total_batches = self.calculate_total_batches(train_dataset, batch_size, gradient_accumulation_steps=4)
        print(f"Total batches: {total_batches * num_epochs}")
        patch_steps = int(total_batches * lambda_ratio)
        standard_steps = total_batches - patch_steps

        patch_steps = patch_steps * num_epochs
        standard_steps = standard_steps * num_epochs
        print(f"Patch steps: {patch_steps}")
        print(f"Standard steps: {standard_steps}")
        print(f'Both equal: {patch_steps == standard_steps}')
        trainer_manager = ModelTrainer(training_config)
        
        try:
            # Phase 1: Patch Training
            if patch_epochs > 0:
                print(f"Starting Phase 1: Patch Training (patch_size={patch_size}) for {patch_epochs} epochs")
                patch_model = PatchTrainModel(
                    self.model,
                    patch_size=patch_size,
                    patch_calculation_method=patch_calculation_method
                )
                
                # Create training arguments for patch phase
                patch_training_args = {
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": batch_size,
                    "per_device_eval_batch_size": batch_size,
                    "max_steps": patch_steps,
                    **trainer_kwargs
                }
                
                # Train with patch model
                trainer = trainer_manager.setup_trainer(
                    model=patch_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=self.tokenizer,
                    batch_size=batch_size,
                    training_stage="patch_phase",
                    training_args=patch_training_args
                )
                trainer.train()
                
                # Get the trained model
                self.model = patch_model.base_model
            
            # Phase 2: Standard Training
            if standard_epochs > 0:
                print(f"Starting Phase 2: Standard Training (patch_size=1) for {standard_epochs} epochs")
                standard_model = self.model
                
                # Create training arguments for standard phase
                standard_training_args = {
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": batch_size,
                    "per_device_eval_batch_size": batch_size,
                    "max_steps": standard_steps,
                    **trainer_kwargs
                }
                torch.cuda.empty_cache()
                # Train with standard model
                trainer = trainer_manager.setup_trainer(
                    model=standard_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=self.tokenizer,
                    batch_size=batch_size,
                    training_stage="standard_phase",
                    training_args=standard_training_args
                )
                trainer.train()
                
                # Get the final trained model
                self.model = standard_model
            
            return trainer, self.model
        finally:
            # Make sure to finish the wandb run
            trainer_manager.finish_wandb()

    def save_adapters(self, path: str):
        """Save the LoRA adapters."""
        if self.model is not None:
            self.model.save_pretrained(path) 

    def train_patch_peft(
        self,
        train_dataset,
        eval_dataset=None,
        patch_size: int = 4,
        patch_calculation_method: str = "mean",
        lambda_ratio: float = 2/3,
        num_epochs: int = 3,
        patch_epochs: Union[int, None] = None,
        standard_epochs: Union[int, None] = None,
        batch_size: int = 8,
        training_config: Optional[TrainingConfig] = None,
        **trainer_kwargs
    ):
        """
        Train a model using the patch training strategy with PEFT.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            patch_size: Size of patches for patch training
            patch_calculation_method: Method to calculate patch values
            lambda_ratio: Ratio of training to use patch training
            num_epochs: Total number of epochs
            patch_epochs: Number of epochs for patch training (overrides lambda_ratio)
            standard_epochs: Number of epochs for standard training (overrides lambda_ratio)
            batch_size: Batch size for training
            training_config: Configuration for training (if None, a default config will be used)
            **trainer_kwargs: Additional arguments to pass to the Trainer
        """
        if training_config is None:
            training_config = TrainingConfig()

        if self.model is None:
            self.setup_lora()
        
        # Calculate number of epochs for each phase
        if patch_epochs is None:
            patch_epochs = int(num_epochs * lambda_ratio)
        if standard_epochs is None:
            standard_epochs = num_epochs - patch_epochs

        total_batches = self.calculate_total_batches(train_dataset, batch_size, gradient_accumulation_steps=4)
        print(f"Total batches: {total_batches * num_epochs}")
        patch_steps = int(total_batches * lambda_ratio)
        standard_steps = total_batches - patch_steps

        patch_steps = patch_steps * num_epochs
        standard_steps = standard_steps * num_epochs
        print(f"Patch steps: {patch_steps}")
        print(f"Standard steps: {standard_steps}")
        print(f'Both equal: {patch_steps == standard_steps}')
        
        trainer_manager = ModelTrainer(training_config)
        
        try:
            # Phase 1: Patch Training
            if patch_epochs > 0:
                print(f"Starting Phase 1: Patch Training with PEFT (patch_size={patch_size}) for {patch_epochs} epochs")
                patch_model = PatchTrainModel(
                    self.model,
                    patch_size=patch_size,
                    patch_calculation_method=patch_calculation_method
                )
                
                # Create training arguments for patch phase
                patch_training_args = {
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": batch_size,
                    "per_device_eval_batch_size": batch_size,
                    "max_steps": patch_steps,
                    **trainer_kwargs
                }
                
                # Train with patch model
                trainer = trainer_manager.setup_trainer(
                    model=patch_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=self.tokenizer,
                    batch_size=batch_size,
                    training_stage="patch_phase_peft",
                    training_args=patch_training_args
                )
                trainer.train()
                
                # Get the trained model
                self.model = patch_model.base_model
            
            # Phase 2: Standard Training
            if standard_epochs > 0:
                print(f"Starting Phase 2: Standard Training with PEFT (patch_size=1) for {standard_epochs} epochs")
                standard_model = self.model
                
                # Create training arguments for standard phase
                standard_training_args = {
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": batch_size,
                    "per_device_eval_batch_size": batch_size,
                    "max_steps": standard_steps,
                    **trainer_kwargs
                }
                torch.cuda.empty_cache()
                # Train with standard model
                trainer = trainer_manager.setup_trainer(
                    model=standard_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=self.tokenizer,
                    batch_size=batch_size,
                    training_stage="standard_peft_phase",
                    training_args=standard_training_args
                )
                trainer.train()
                
                # Get the final trained model
                self.model = standard_model
            
            return trainer, self.model
        finally:
            # Make sure to finish the wandb run
            trainer_manager.finish_wandb()

    def calculate_total_batches(self, train_dataset, batch_size, gradient_accumulation_steps):
        return len(train_dataset) // (batch_size * gradient_accumulation_steps)
        
