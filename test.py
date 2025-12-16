import argparse
import json
import os
from datetime import datetime
from typing import List, Dict

import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

import sacrebleu
from bert_score import score as bert_score
from comet import download_model, load_from_checkpoint


# =========================
# PROMPT (STRICT)
# =========================
def create_chat_messages(source_text, src_code, tgt_code):
    return [
        {
            "role": "system",
            "content": (
                f"You are a strict machine translation system. "
                f"Translate the user's text from {src_code} to {tgt_code}.\n"
                "Output ONLY the translated text.\n"
                "Do not provide explanations, notes, header text, or multiple options.\n"
                "Do not enclose the output in quotes."
            )
        },
        {
            "role": "user",
            "content": source_text
        }
    ]


def flatten_chat(messages):
    """Simple chat → plain text conversion (Gemma/LLaMA friendly)"""
    text = ""
    for m in messages:
        text += f"{m['role'].upper()}: {m['content']}\n"
    return text.strip()


# =========================
# MODEL LOADING
# =========================
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


# =========================
# GENERATION
# =========================
@torch.no_grad()
def translate_batch(
    model,
    tokenizer,
    inputs: List[str],
    max_new_tokens=256
):
    enc = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    outputs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        top_k=None
    )

    decoded = tokenizer.batch_decode(
        outputs[:, enc["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return [d.strip() for d in decoded]


# =========================
# METRICS
# =========================
def compute_metrics(preds, refs, comet_model):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    P, R, F1 = bert_score(
        preds,
        refs,
        lang="en",
        rescale_with_baseline=True
    )

    comet_score = comet_model.predict(
        [{"src": "", "mt": p, "ref": r} for p, r in zip(preds, refs)],
        batch_size=8,
        gpus=1
    )["system_score"]

    return {
        "BLEU": bleu,
        "chrF": chrf,
        "BERTScore_F1": F1.mean().item(),
        "COMET": comet_score
    }


# =========================
# MAIN EVAL LOOP
# =========================
def evaluate_dataset(
    dataset_name,
    model,
    tokenizer,
    comet_model,
    output_dir
):
    ds = load_dataset(dataset_name)
    test_split = ds["test"]
    
    # Get all Indic language columns (exclude English)
    lang_columns = [col for col in test_split.column_names 
                   if "_" in col and col[0].islower() and col != "eng_Latn"]
    
    print(f"\n▶ Dataset: {dataset_name}")
    print(f"Languages to evaluate: {', '.join(lang_columns)}")
    print(f"Total samples: {len(test_split)}")

    all_logs = []
    per_lang_results = {}

    # Process each source language separately
    for src_lang in lang_columns:
        print(f"\n▶ Processing: {src_lang} → eng_Latn")
        
        lang_preds = []
        lang_refs = []
        
        for sample in tqdm(test_split, desc=src_lang):
            src = sample[src_lang]
            ref = sample["eng_Latn"]
            
            # Skip if source or reference is empty
            if not src or not ref:
                continue

            chat = create_chat_messages(src, src_lang, "eng_Latn")
            prompt = flatten_chat(chat)

            pred = translate_batch(
                model,
                tokenizer,
                [prompt]
            )[0]

            lang_preds.append(pred)
            lang_refs.append(ref)

            all_logs.append({
                "dataset": dataset_name,
                "src_lang": src_lang,
                "tgt_lang": "eng_Latn",
                "source": src,
                "prediction": pred,
                "reference": ref
            })
        
        # Compute metrics for this language pair
        if lang_preds:
            lang_metrics = compute_metrics(lang_preds, lang_refs, comet_model)
            per_lang_results[src_lang] = lang_metrics
            print(f"\n{src_lang} → eng_Latn Results:")
            for metric, score in lang_metrics.items():
                print(f"  {metric}: {score:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/{dataset_name.replace('/', '_')}_logs_{timestamp}.json", "w") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)

    with open(f"{output_dir}/{dataset_name.replace('/', '_')}_metrics_{timestamp}.json", "w") as f:
        json.dump(per_lang_results, f, indent=2)

    return per_lang_results


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="HF model name (e.g. google/gemma-1.1-1b-it or meta-llama/Llama-3.2-1B-Instruct)"
    )
    parser.add_argument(
        "--output_dir",
        default="mt_eval_logs"
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)

    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)

    # Evaluate only IN22-Gen for now
    dataset = "ai4bharat/IN22-Gen"
    results = evaluate_dataset(
        dataset,
        model,
        tokenizer,
        comet_model,
        args.output_dir
    )

    print("\n====== FINAL RESULTS: IN22-Gen ======")
    for lang, metrics in results.items():
        print(f"\n{lang} → eng_Latn:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()
