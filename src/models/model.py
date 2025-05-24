import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from ..config.config import ModelConfig
from .patch_model import PatchTrainModel

class ModelManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

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
                self.model = PeftModel.from_pretrained(
                    AutoModelForCausalLM.from_pretrained(
                        self.config.model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                    ),
                    model_path
                )
            except:
                # If not a PEFT model, load as regular model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
        else:
            if self.config.use_4bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    load_in_4bit=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name
                ).to(self.device)

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
        **trainer_kwargs
    ):
        """
        Train a model using the patch training strategy.
        
        This method implements a two-phase training approach:
        1. First phase: Train with patch_size for (lambda_ratio * num_epochs) epochs
        2. Second phase: Train with patch_size=1 for the remaining epochs
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            patch_size: Size of patches for the first phase of training
            lambda_ratio: Ratio of epochs to use patch training (e.g., 2/3 means use patch training for 2/3 of epochs)
            num_epochs: Total number of training epochs
            **trainer_kwargs: Additional arguments to pass to the Trainer
        """
        if self.model is None:
            self.load_model_and_tokenizer()
            
        # Calculate number of epochs for each phase
        patch_epochs = int(num_epochs * lambda_ratio)
        standard_epochs = num_epochs - patch_epochs
        
        # Phase 1: Patch Training
        if patch_epochs > 0:
            print(f"Starting Phase 1: Patch Training (patch_size={patch_size}) for {patch_epochs} epochs")
            patch_model = PatchTrainModel(
                self.model,
                patch_size=patch_size,
                patch_calculation_method=patch_calculation_method
            )
            
            # Create training arguments for patch phase
            patch_training_args = TrainingArguments(
                num_train_epochs=patch_epochs,
                **trainer_kwargs
            )
            
            # Train with patch model
            trainer = Trainer(
                model=patch_model,
                args=patch_training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                **trainer_kwargs
            )
            trainer.train()
            
            # Get the trained model
            self.model = patch_model.base_model
        
        # Phase 2: Standard Training
        if standard_epochs > 0:
            print(f"Starting Phase 2: Standard Training (patch_size=1) for {standard_epochs} epochs")
            standard_model = PatchTrainModel(
                self.model,
                patch_size=1,
                patch_calculation_method="mean"
            )
            
            # Create training arguments for standard phase
            standard_training_args = TrainingArguments(
                num_train_epochs=standard_epochs,
                **trainer_kwargs
            )
            
            # Train with standard model
            trainer = Trainer(
                model=standard_model,
                args=standard_training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                **trainer_kwargs
            )
            trainer.train()
            
            # Get the final trained model
            self.model = standard_model.base_model
            
        return self.model

    def save_adapters(self, path: str):
        """Save the LoRA adapters."""
        if self.model is not None:
            self.model.save_pretrained(path) 