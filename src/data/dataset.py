from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    dataset_name: str
    subset: str = None
    prompt_template: str = None
    answer_template: str = None

class BaseProcessor:
    def __init__(self, config: 'DatasetConfig'):
        self.config = config
        self.dataset = None

    def format_as_conversation(self, user_content: str, assistant_content: str = None) -> list:
        """Format the input and output as a conversation list for chat templates."""
        messages = [{"role": "user", "content": user_content}]
        if assistant_content is not None:
            messages.append({"role": "assistant", "content": assistant_content})
        return messages

    def tokenize_with_chat_template(self, tokenizer: PreTrainedTokenizer, messages: list) -> Dict:
        """Apply the chat template and tokenize."""
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return tokenizer(
            chat_text,
            truncation=True,
            max_length=512,
            padding="max_length",
        )

class CNNProcessor(BaseProcessor):
    def load_dataset(self):
        """Load the CNN DailyMail dataset."""
        self.dataset = load_dataset(self.config.dataset_name, "3.0.0")
        return self.dataset

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format a single example with the prompt template."""
        user_prompt = self.config.prompt_template.format(article=example["article"])
        assistant_response = self.config.answer_template.format(highlights=example["highlights"])
        return {"messages": self.format_as_conversation(user_prompt, assistant_response)}

    def preprocess_dataset(self, tokenizer: PreTrainedTokenizer):
        """Preprocess the entire dataset."""
        if self.dataset is None:
            self.dataset = self.load_dataset()
        
        if tokenizer is None:
            return self.dataset

        # Format examples
        formatted_dataset = self.dataset.map(
            self.format_example,
            remove_columns=["article", "highlights", "id"],
        )

        # Tokenize with chat template
        def tokenize(examples):
            return self.tokenize_with_chat_template(tokenizer, examples["messages"])

        tokenized_dataset = formatted_dataset.map(
            tokenize,
            batched=True,
            remove_columns=["messages"]
        )

        return tokenized_dataset

class WMTProcessor(BaseProcessor):
    def load_dataset(self):
        """Load the WMT14 dataset with appropriate subsetting."""
        dataset = load_dataset(self.config.dataset_name, self.config.subset)
        
        # Apply subsetting for French-English
        if self.config.subset == "fr-en":
            # Subset the splits
            dataset = dataset.filter(
                lambda _, idx: idx < 24000 if _ == "train" else idx < 3000,
                with_indices=True
            )
        
        self.dataset = dataset
        return self.dataset

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format a single example with the prompt template."""
        source_lang = self.config.subset.split("-")[0]
        target_lang = self.config.subset.split("-")[1]
        
        user_prompt = self.config.prompt_template.format(
            source_text=example["translation"][source_lang]
        )
        assistant_response = self.config.answer_template.format(
            target_text=example["translation"][target_lang]
        )
        return {"messages": self.format_as_conversation(user_prompt, assistant_response)}

    def preprocess_dataset(self, tokenizer: PreTrainedTokenizer):
        """Preprocess the entire dataset."""
        if self.dataset is None:
            self.dataset = self.load_dataset()

        if tokenizer is None:
            return self.dataset

        # Format examples
        formatted_dataset = self.dataset.map(
            self.format_example,
            remove_columns=["translation"],
        )

        # Tokenize with chat template
        def tokenize(examples):
            return self.tokenize_with_chat_template(tokenizer, examples["messages"])

        tokenized_dataset = formatted_dataset.map(
            tokenize,
            batched=True,
            remove_columns=["messages"]
        )

        return tokenized_dataset