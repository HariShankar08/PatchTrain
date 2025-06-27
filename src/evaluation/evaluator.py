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
        # self.bertscore = evaluate.load('bertscore', module_type='metric')
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
                predictions=predictions, # [pred.split() for pred in predictions],  # Pass untokenized predictions
                references=[[ref] for ref in references]  # Format as list of list of references
            )
            
            # SacreBLEU takes untokenized strings
            sacrebleu_score = self.sacrebleu.compute(
                predictions=predictions, #[pred.split() for pred in predictions],
                references=[[ref] for ref in references]
            )

            # Compute BERTScore
            # bertscore_score = self.bertscore.compute(
            #     predictions=predictions,
            #     references=[[ref] for ref in references],  # Pass untokenized references
            #     lang="en",
            #     rescale_with_baseline=True
            # )
            
            return {
                "bleu": bleu_score["bleu"],
                "sacrebleu": sacrebleu_score["score"],
            #     "bertscore_precision": np.mean(bertscore_score["precision"]),
            #     "bertscore_recall": np.mean(bertscore_score["recall"]),
            #     "bertscore_f1": np.mean(bertscore_score["f1"]),
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
        """Evaluate the model on the test dataset with correct perplexity calculation."""
        model.eval()
        predictions = []
        references = []
        perplexities = []
        
        device = next(model.parameters()).device
        
        # Determine task type from dataset structure
        is_translation = "translation" in dataset[0]
        
        for example in tqdm(dataset.select(range(min(self.config.num_samples, len(dataset)))), 
                            desc="Evaluating"):
            try:
                # Prepare input prompt and reference
                if is_translation:
                    source_lang = next(k for k in example["translation"].keys() if k != "en")
                    prompt = f"Translate from {source_lang.title()} to English:\n{example['translation'][source_lang]}\n\nEnglish:"
                    reference = example["translation"]["en"]
                else:
                    prompt = f"Summarize the following article:\n{example['article']}\n\nSummary:"
                    reference = example["highlights"]
                
                # Tokenize the complete prompt+reference
                messages = self.format_as_conversation(prompt, reference)
                full_text = tokenizer.apply_chat_template(messages, tokenize=False)
                inputs = tokenizer(full_text, return_tensors="pt").to(device)
                
                # Calculate perplexity (only on reference text)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Shift for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()
                    
                    # Find where reference starts in tokenized sequence
                    prompt_only = tokenizer.apply_chat_template(
                        self.format_as_conversation(prompt),
                        tokenize=False
                    )
                    prompt_tokens = tokenizer(prompt_only, return_tensors="pt").input_ids
                    ref_start = prompt_tokens.shape[1] - 1  # account for shift
                    
                    # Compute loss only on reference portion
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    ).view(shift_labels.shape)
                    
                    ref_loss = losses[0, ref_start:].mean()
                    perplexity = math.exp(ref_loss.item())
                    perplexities.append(perplexity)
                
                # Generate prediction
                messages = self.format_as_conversation(prompt)
                chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(chat_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        num_beams=4,
                        length_penalty=0.6,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                        repetition_penalty=1.2,
                        return_dict_in_generate=True
                    )
                    
                    pred_text = tokenizer.decode(
                        outputs.sequences[0][inputs["input_ids"].shape[1]:], 
                        skip_special_tokens=True
                    )
                    predictions.append(pred_text.strip())
                    references.append(reference)
                    
            except Exception as e:
                print(f"Error processing example: {e}")
                continue
        
        # Calculate task-specific metrics
        if is_translation:
            metrics = self.compute_translation_metrics(predictions, references)
        else:
            metrics = self.compute_summarization_metrics(predictions, references)
        
        # Add perplexity metrics
        metrics.update({
            "perplexity": np.mean(perplexities),
            "perplexity_std": np.std(perplexities),
            # "perplexities": perplexities  # raw values for analysis
        })
        
        return metrics, predictions, references
        
    def print_examples(self, predictions: List[str], references: List[str], num_examples: int = 3):
        """Print example predictions and their references."""
        print("\nExample Predictions:")
        for i in range(min(num_examples, len(predictions))):
            print(f"\nExample {i+1}:")
            print(f"Predicted: {predictions[i]}")
            print(f"Reference: {references[i]}")
