from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Any
from config.config import GSM8kDataConfig, TinyStoriesDataConfig

class GSM8KProcessor:
    def __init__(self, config: GSM8kDataConfig):
        self.config = config
        self.dataset = None

    def load_dataset(self):
        """Load the GSM-8K dataset."""
        self.dataset = load_dataset(self.config.dataset_name, "main")
        return self.dataset

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format a single example with the prompt template."""
        final_answer = example["answer"].split("#### ")[-1]
        prompt = self.config.prompt_template.format(question=example["question"])
        answer = self.config.answer_template.format(
            answer=example["answer"],
            final_answer=final_answer
        )
        return {"text": prompt + answer}

    def preprocess_dataset(self, tokenizer: PreTrainedTokenizer):
        """Preprocess the entire dataset."""
        if self.dataset is None or tokenizer is None:
            return self.load_dataset()

        # Format examples
        tokenized_dataset = self.dataset.map(
            self.format_example,
            remove_columns=["question", "answer"],
        )

        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        tokenized_dataset = tokenized_dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset

class TinyStoriesProcessor:
    def __init__(self, config: TinyStoriesDataConfig):
        self.config = config
        self.dataset = None

    def load_dataset(self):
        """Load the TinyStories dataset."""
        self.dataset = load_dataset("roneneldan/TinyStories")
        return self.dataset

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format a single example with the prompt template."""
        prompt = self.config.prompt_template.format(text=example["text"])
        return {"text": prompt}

    def preprocess_dataset(self, tokenizer: PreTrainedTokenizer):
        """Preprocess the entire dataset."""
        if self.dataset is None or tokenizer is None:
            return self.load_dataset()
        
        # Format examples
        tokenized_dataset = self.dataset.map(
            self.format_example,
            remove_columns=["text"]
        )

        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        tokenized_dataset = tokenized_dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset