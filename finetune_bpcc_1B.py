import argparse
import torch
import os
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)

# =========================
# ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ai4bharat/BPCC",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="bpcc-seed-latest",
    )

    parser.add_argument("--output_dir", type=str, default="./llama1b-bpcc-chat-en")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)

    # Note: Logic inside main() will auto-select BF16 if available
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=False)

    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=500)

    return parser.parse_args()


# =========================
# STRICT CHAT TEMPLATE
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
            ),
        },
        {
            "role": "user",
            "content": source_text,
        },
    ]


# =========================
# LOAD ALL SPLITS
# =========================
def load_all_indic_to_en(dataset_name, subset):
    ds = load_dataset(dataset_name, subset)
    streams = []

    print("\nDataset splits found:")
    for split_name in ds.keys():
        split = ds[split_name]
        # Filter slightly to ensure we have data
        if len(split) == 0:
            continue
            
        print(f"  Split: {split_name}, Size: {len(split)}")
        streams.append(split)
        
        # break 

    return streams


# =========================
# TOKENIZATION (CHAT LOSS MASKING)
# =========================
def tokenize_example(example, tokenizer, max_length):
    # Intentional SWAP: 
    # src in dataset (English) -> Output
    # tgt in dataset (Indic)   -> Input
    
    src_text = example["tgt"]        # Indic text (input)
    tgt_text = example["src"]        # English text (output)

    src_lang = example["tgt_lang"]   # e.g., ben_Beng
    tgt_lang = "eng_Latn"            # English

    messages = create_chat_messages(
        src_text, src_lang, tgt_lang
    )

    # 1. Format the prompt (System + User)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2. Format the full sequence (Prompt + Output)
    full_text = prompt_text + tgt_text + tokenizer.eos_token

    # 3. Tokenize Full Sequence (Input IDs & Attention Mask)
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=False # apply_chat_template usually handles special tokens
    )

    # 4. Tokenize Prompt Only (To calculate masking length)
    # We use the same settings to ensure length consistency
    prompt_tokenized = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False 
    )
    prompt_len = len(prompt_tokenized["input_ids"])

    # 5. Create Labels
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    # Mask the prompt tokens so the model doesn't learn to generate the prompt
    if prompt_len < len(labels):
        labels[:prompt_len] = [-100] * prompt_len
    else:
        # In rare case prompt is longer than max_length (truncated), mask everything
        labels[:] = [-100] * len(labels)
        
    # Mask padding tokens (where input_ids is pad_token)
    for i, token_id in enumerate(input_ids):
        if token_id == tokenizer.pad_token_id:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    # --- Hardware / Precision Auto-Configuration ---
    # Llama works best with BF16. FP16 often causes gradient unscale errors.
    use_bf16 = False
    use_fp16 = False
    
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print(">> BF16 hardware detected. Switching to BF16 for Llama stability.")
        use_bf16 = True
        model_dtype = torch.bfloat16
    else:
        print(">> BF16 not supported. Falling back to FP16.")
        use_fp16 = True
        # CRITICAL FIX: If using FP16, load model in float32 initially.
        # This allows the Trainer to create FP32 master weights for the optimizer.
        # Loading in float16 directly causes the 'Attempting to unscale FP16 gradients' error.
        model_dtype = torch.float32 

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Fix padding token issues
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Trainer expects right padding usually

    print(f"Loading model: {args.model_name} with dtype {model_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        device_map="auto",
        attn_implementation="sdpa" if use_bf16 else "eager" # optimize attention
    )

    print(f"Loading dataset: {args.dataset_name} ({args.subset})")
    dataset_streams = load_all_indic_to_en(args.dataset_name, args.subset)

    if not dataset_streams:
        raise ValueError("No valid dataset splits found.")

    print(f"Found {len(dataset_streams)} dataset splits.")
    
    print("Tokenizing datasets...")
    tokenized_streams = []
    for idx, ds in enumerate(dataset_streams):
        # Optional: sample specific languages or limit size for debugging
        print(f"  Processing split {idx+1}/{len(dataset_streams)}: {len(ds)} examples")
        
        tok = ds.map(
            lambda x: tokenize_example(x, tokenizer, args.max_length),
            remove_columns=ds.column_names,
            desc=f"Tokenizing split {idx+1}",
            num_proc=24, # Speed up tokenization
        )
        tokenized_streams.append(tok)

    print("Combining datasets...")
    combined_dataset = concatenate_datasets(tokenized_streams)
    print(f"Total examples: {len(combined_dataset)}")
    
    # Split into train and eval (95/5 split for larger data)
    print("Splitting into train/eval...")
    split_dataset = combined_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        warmup_ratio=0.03,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()