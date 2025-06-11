import torch
import evaluate
from typing import Dict, List
from config.config import EvaluationConfig
import numpy as np
from math import exp
from tqdm import tqdm
import math

class ModelEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        # Initialize all metrics
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')
        self.sacrebleu = evaluate.load('sacrebleu')
        self.perplexity = evaluate.load('perplexity', module_type='metric')

    def format_as_conversation(self, user_content: str, assistant_content: str = None) -> list:
        """Format the input and output as a conversation list for chat templates."""
        messages = [{"role": "user", "content": user_content}]
        if assistant_content is not None:
            messages.append({"role": "assistant", "content": assistant_content})
        return messages

    def compute_translation_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute BLEU and SacreBLEU scores for translation tasks."""
        # For BLEU, we need to tokenize the strings into lists of tokens
        
        try:
            # Compute BLEU score with properly formatted inputs
            bleu_score = self.bleu.compute(
                predictions=[pred.split() for pred in predictions],  # Pass untokenized predictions
                references=[[ref.split()] for ref in references]  # Format as list of list of references
            )
            
            # SacreBLEU takes untokenized strings
            sacrebleu_score = self.sacrebleu.compute(
                predictions=[pred.split() for pred in predictions],
                references=[[ref.split()] for ref in references]
            )
            
            return {
                "bleu": bleu_score["bleu"],
                "sacrebleu": sacrebleu_score["score"]
            }
        except Exception as e:
            print(f"Warning: Error computing translation metrics: {str(e)}")
            print(f"Sample prediction: {predictions[0] if predictions else 'No predictions'}")
            print(f"Sample reference: {references[0] if references else 'No references'}")
            return {
                "bleu": 0.0,
                "sacrebleu": 0.0
            }

    def compute_summarization_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE scores for summarization tasks."""
        return self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )

    def evaluate_model(self, model, tokenizer, dataset):
        """Evaluate the model on the test dataset."""
        model.eval()
        predictions: List[str] = []
        references: List[str] = []
        all_losses = []

        # Get the device the model is on
        device = next(model.parameters()).device

        # Determine dataset type from first example
        first_example = dataset[0]
        is_translation = "translation" in first_example

        for example in tqdm(dataset.select(range(min(self.config.num_samples, len(dataset)))), desc="Evaluating"):
            # Get source and target based on dataset type
            if not is_translation:  # CNN DailyMail
                user_prompt = f"Summarize the following article:\n{example['article']}\n\nSummary:"
                reference = example["highlights"]
            else:  # WMT
                source_lang = next(k for k in example["translation"].keys() if k != "en")
                user_prompt = f"Translate from {source_lang.title()} to English:\n{example['translation'][source_lang]}\n\nEnglish:"
                reference = example["translation"]["en"]
            
            # Format as chat and tokenize
            messages = self.format_as_conversation(user_prompt)
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat_text, return_tensors="pt")
            # Move inputs to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            references.append(reference)
            
            # Generate prediction
            with torch.no_grad():
                # Generate with optimized parameters for translation/summarization
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    num_beams=4,  # Using beam search for better quality
                    length_penalty=0.6,  # Favor shorter sequences for summarization
                    no_repeat_ngram_size=3,  # Prevent repetition of n-grams
                    early_stopping=True,  # Stop when all beams are finished
                    diversity_penalty=0.1,  # Add some diversity to beam search
                    num_beam_groups=1,  # Default beam groups
                    top_k=50,  # Keep top 50 tokens for sampling
                    top_p=0.9,  # Nucleus sampling threshold
                    repetition_penalty=1.2,  # Additional repetition prevention
                )
                
                # Calculate perplexity
                full_messages = self.format_as_conversation(user_prompt, reference)
                full_chat_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
                full_inputs = tokenizer(full_chat_text, return_tensors="pt")
                # Move full_inputs to the same device as model
                full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
                
                # Get the logits
                logits = model(**full_inputs).logits
                # Get the loss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), full_inputs['input_ids'].view(-1))
                all_losses.append(loss.item())

            # Process prediction
            full_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            # Extract the generated response after the chat template
            try:
                pred_answer = full_output.split("assistant")[-1].strip()
                if pred_answer.startswith(":"):
                    pred_answer = pred_answer[1:].strip()
            except:
                pred_answer = ""
            predictions.append(pred_answer)

        # Calculate metrics based on dataset type
        if is_translation:
            task_metrics = self.compute_translation_metrics(predictions, references)
        else:
            task_metrics = self.compute_summarization_metrics(predictions, references)
        
        # Calculate average perplexity
        avg_loss = np.mean(all_losses)
        task_metrics["perplexity"] = math.exp(avg_loss)
        
        return task_metrics, predictions, references

    def print_examples(self, predictions: List[str], references: List[str], num_examples: int = 3):
        """Print example predictions and their references."""
        print("\nExample Predictions:")
        for i in range(min(num_examples, len(predictions))):
            print(f"\nExample {i+1}:")
            print(f"Predicted: {predictions[i]}")
            print(f"Reference: {references[i]}")
