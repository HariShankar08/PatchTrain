import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from config.config import ModelConfig, TrainingConfig
from .patch_model import PatchTrainModel
from typing import Union, Optional
from training.trainer import ModelTrainer, GPUMemoryCallback
import bitsandbytes as bnb

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
                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float32 if self.device in ("mps", "cpu") else torch.bfloat16,
                )
            except:
                # If not a PEFT model, load as regular model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float32 if self.device in ("mps", "cpu") else torch.bfloat16,
                )
        else:
            load_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float32 if self.device in ("mps", "cpu") else torch.bfloat16,
            }
            
            # Handle quantization based on config
            if self.config.use_qlora:
                # QLoRA always uses 4-bit quantization
                load_kwargs.update({
                    "load_in_4bit": True,
                    "quantization_config": {
                        "bnb_4bit_compute_dtype": torch.bfloat16,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4",
                        "llm_int8_threshold": 6.0,
                        "llm_int8_has_fp16_weight": False,
                    }
                })
            elif self.config.use_4bit:
                # Regular 4-bit quantization without QLoRA optimizations
                load_kwargs["load_in_4bit"] = True
            elif self.config.use_8bit:
                load_kwargs["load_in_8bit"] = True
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **load_kwargs
            )

        return self.model, self.tokenizer

    def setup_lora(self, r=8, alpha=32, target_modules=None, lora_dropout=0.05):
        """Configure and apply LoRA/QLoRA to the model."""
        if self.model is None:
            self.load_model_and_tokenizer()

        # If model is already a PEFT model, return it
        if hasattr(self.model, "peft_config"):
            return self.model

        # Prepare model for quantized training if using QLoRA or other quantization
        if self.config.use_qlora or self.config.use_4bit or self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True,
            )

        if target_modules is None:
            if self.config.use_qlora:
                # QLoRA typically targets all linear layers
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"]
            else:
                # Regular LoRA targets attention layers
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # LoRA configuration
        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        # Print trainable parameters info
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

        # Define save_path before the try block
        save_path = f"{training_config.save_path}_seed{training_config.seed}"
        
        try:
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
            
            from training.pytorch_trainer import PyTorchTrainer
            trainer_manager = PyTorchTrainer(training_config)
            
            # Phase 1: Patch Training
            if patch_epochs > 0:
                print(f"Starting Phase 1: Patch Training (patch_size={patch_size}) for {patch_epochs} epochs")
                patch_model = PatchTrainModel(
                    self.model,
                    patch_size=patch_size,
                    patch_calculation_method=patch_calculation_method
                )
                
                # Train with patch model
                patch_model = trainer_manager.train(
                    model=patch_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    training_stage="patch_phase",
                    max_steps=patch_steps
                )
                
                # Get the trained model
                self.model = patch_model.base_model
            
            # Phase 2: Standard Training
            if standard_epochs > 0:
                print(f"Starting Phase 2: Standard Training (patch_size=1) for {standard_epochs} epochs")
                standard_model = self.model
                
                torch.cuda.empty_cache()
                # Train with standard model
                standard_model = trainer_manager.train(
                    model=standard_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    training_stage="standard_phase",
                    max_steps=standard_steps
                )
                
                # Get the final trained model
                self.model = standard_model
            
            return trainer_manager, self.model
        finally:
            # Make sure to finish the wandb run and save artifact
            trainer_manager.finish_wandb(model_path=save_path)

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
        
        from training.pytorch_trainer import PyTorchTrainer
        trainer_manager = PyTorchTrainer(training_config)
        
        try:
            # Phase 1: Patch Training
            if patch_epochs > 0:
                print(f"Starting Phase 1: Patch Training with PEFT (patch_size={patch_size}) for {patch_epochs} epochs")
                patch_model = PatchTrainModel(
                    self.model,
                    patch_size=patch_size,
                    patch_calculation_method=patch_calculation_method
                )
                
                # Train with patch model
                patch_model = trainer_manager.train(
                    model=patch_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    training_stage="patch_phase_peft",
                    max_steps=patch_steps
                )
                
                # Get the trained model
                self.model = patch_model.base_model
            
            # Phase 2: Standard Training
            if standard_epochs > 0:
                print(f"Starting Phase 2: Standard Training with PEFT (patch_size=1) for {standard_epochs} epochs")
                standard_model = self.model
                
                torch.cuda.empty_cache()
                # Train with standard model
                standard_model = trainer_manager.train(
                    model=standard_model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    training_stage="standard_peft_phase",
                    max_steps=standard_steps
                )
                
                # Get the final trained model
                self.model = standard_model
            
            # Save the PEFT model
            save_path = f'{training_config.save_path}_seed{training_config.seed}'
            self.save_adapters(save_path)
            
            return trainer_manager, self.model
        finally:
            # Make sure to finish the wandb run
            trainer_manager.finish_wandb(model_path=save_path)

    def calculate_total_batches(self, train_dataset, batch_size, gradient_accumulation_steps):
        return len(train_dataset) // (batch_size * gradient_accumulation_steps)
        
