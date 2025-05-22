import torch
import evaluate
from typing import Dict, List
from ..config.config import EvaluationConfig

class ModelEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.exact_match = evaluate.load("exact_match")

    def evaluate_model(self, model, tokenizer, dataset, device: str):
        """Evaluate the model on the test dataset."""
        model.eval()
        predictions: List[str] = []
        references: List[str] = []

        for example in dataset.select(range(min(self.config.num_samples, len(dataset)))):
            # Create prompt
            prompt = f"Question: {example['question']}\nLet's think step by step:\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Process prediction
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            try:
                pred_answer = full_output.split("The final answer is:")[-1].strip()
            except:
                pred_answer = ""

            # Get reference answer
            ref_answer = example["answer"].split("####")[-1].strip()

            predictions.append(pred_answer)
            references.append(ref_answer)

        # Calculate metrics
        metrics = self.exact_match.compute(
            predictions=predictions,
            references=references
        )

        return metrics, predictions, references

    def print_examples(self, predictions: List[str], references: List[str], num_examples: int = 3):
        """Print example predictions and their references."""
        print("\nExample Predictions:")
        for i in range(min(num_examples, len(predictions))):
            print(f"\nExample {i+1}:")
            print(f"Predicted: {predictions[i]}")
            print(f"Reference: {references[i]}") 