from dataclasses import dataclass
from typing import Optional
from enum import Enum

class RunMode(Enum):
    EVALUATE = "evaluate"  # Only evaluate a pretrained model
    FULL_FINETUNE = "full_finetune"  # Full model finetuning
    LORA_FINETUNE = "lora_finetune"  # LoRA finetuning
    PATCH_TRAIN = "patch_train"
    PATCH_PEFT = "patch_peft"

@dataclass
class RunConfig:
    mode: RunMode
    model_path: Optional[str]
    save_path: str
    eval_only: bool = False  # Whether to skip training and only evaluate

@dataclass
class ModelConfig:
    model_name: str = "facebook/opt-350m"
    use_4bit: bool = True
    lora_rank: int = 8
    max_seq_length: int = 512
    batch_size: int = 8

@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    fp16: bool = False
    bf16: bool = True
    max_grad_norm=1.0
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "no"

@dataclass
class GSM8kDataConfig:
    dataset_name: str = "gsm8k"
    prompt_template: str = "Question: {question}\nLet's think step by step to solve the problem.\nAnswer:"
    answer_template: str = " {answer}\nThe final answer is: {final_answer}"

@dataclass
class EvaluationConfig:
    num_samples: int = 50
    max_new_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True 

@dataclass
class TinyStoriesDataConfig:
    dataset_name: str = "roneneldan/TinyStories"
    prompt_template: str = "Generate a story: {text}"

