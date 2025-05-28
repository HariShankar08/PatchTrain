import torch
import evaluate
from typing import Dict, List
from config.config import EvaluationConfig
import numpy as np
from math import exp

class ModelEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.rouge = evaluate.load('rouge')
        self.perplexity = evaluate.load('perplexity', module_type='metric')

    def evaluate_model(self, model, tokenizer, dataset, device: str):
        """Evaluate the model on the test dataset."""
        model.eval()
        predictions: List[str] = []
        references: List[str] = []
        all_losses = []

        for example in dataset.select(range(min(self.config.num_samples, len(dataset)))):
            print(example)
            # Create prompt
            prompt = f"Question: {example['question']}\nLet's think step by step:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Get reference answer
            ref_answer = example["answer"].split("####")[-1].strip()
            ref_answer = f'{example["answer"]}\nThe final answer is: {ref_answer}'
            references.append(ref_answer)
            
            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
                # Calculate perplexity
                # Get logits from the output
                #loss_out = model(inputs)
                #logits = torch.stack(outputs.scores, dim=1).to(device)
                # Shift tokens and logits to align them
                #shift_logits = logits[..., :-1, :].contiguous()
                #shift_labels = outputs.sequences[..., 1:].contiguous()
                # Flatten the tokens and logits
                # loss_fct = torch.nn.CrossEntropyLoss()
                # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                all_losses.append(0)

            # Process prediction
            full_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            try:
                pred_answer = full_output.split("Let's think step by step:")[-1].strip()
            except:
                pred_answer = ""
            predictions.append(pred_answer)

        # Calculate ROUGE metrics
        rouge_metrics = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        # Calculate average perplexity
        avg_loss = np.mean(all_losses)
        avg_perplexity = exp(avg_loss)
        
        # Combine all metrics
        metrics = {
            **rouge_metrics,
            "average_perplexity": avg_perplexity,
            "average_loss": avg_loss
        }

        return metrics, predictions, references

    def print_examples(self, predictions: List[str], references: List[str], num_examples: int = 3):
        """Print example predictions and their references."""
        print("\nExample Predictions:")
        for i in range(min(num_examples, len(predictions))):
            print(f"\nExample {i+1}:")
            print(f"Predicted: {predictions[i]}")
            print(f"Reference: {references[i]}")
