import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from ..config.config import ModelConfig

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

    def save_adapters(self, path: str):
        """Save the LoRA adapters."""
        if self.model is not None:
            self.model.save_pretrained(path) 