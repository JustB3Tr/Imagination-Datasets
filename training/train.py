#!/usr/bin/env python3
"""
LoRA (full bf16 base, not quantized) finetune of Qwen3-Coder-30B-A3B-Instruct
on the Imagination 2.1 Pro dataset.

Designed to "just run": upload this file plus train.jsonl / eval.jsonl
(from dedup_and_split.py) into the same directory on your Vast.ai instance
(Unsloth Studio template, A100 80GB recommended -- the bf16 base weights
alone are ~60GB, won't fit on a 24-48GB card) and run:

    python train.py

It will:
  1. auto-find train.jsonl / eval.jsonl (checks ./ and ./data/)
  2. download Qwen3-Coder-30B-A3B-Instruct from Hugging Face (first run only,
     cached after that)
  3. apply LoRA on the full-precision base (attention + expert FFN layers,
     router frozen -- see comment below on why). Pass --load_in_4bit to
     fall back to QLoRA if you ever need to run on a smaller/cheaper GPU --
     the quality gap is normally small, this just gets you the extra bit
     when the budget/hardware allows it.
  4. render each example with the model's own chat template (so tool-calling
     formatting matches what the base model already learned in pretraining)
  5. mask loss to assistant turns only (critical -- without this the model
     also trains to predict user/tool messages, which dilutes the signal and
     can teach it to hallucinate fake tool outputs)
  6. train with checkpointing (resumes automatically if the run gets
     interrupted and you re-launch the script -- spot instances can die)
  7. save the LoRA adapter when done

Do a smoke test first:  python train.py --max_steps 20
Then the real run:      python train.py
"""
import argparse
import glob
import json
import os

# Unsloth must be imported before transformers/torch/trl -- it patches them
# at import time for the memory/speed optimizations to take effect.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig


def find_dataset_file(name: str) -> str:
    candidates = [
        name,
        os.path.join("data", name),
        os.path.join(os.path.dirname(__file__), name),
        os.path.join(os.path.dirname(__file__), "data", name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Couldn't find {name}. Looked in: {candidates}. "
        f"Upload it alongside this script (or in a data/ subfolder)."
    )


def load_jsonl_messages(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # meta (domain/level/mode/seed_task) isn't needed for training,
            # only the actual conversation.
            examples.append({"messages": row["messages"]})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--output_dir", default="./lora_out")
    ap.add_argument("--load_in_4bit", action="store_true",
                     help="Fall back to QLoRA (4-bit base) instead of full "
                          "bf16 LoRA -- use this if you're on a 24-48GB card "
                          "instead of an 80GB one. Quality difference vs full "
                          "LoRA is normally small.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=-1,
                     help="Set to a small number (e.g. 20) for a smoke test "
                          "before committing to the full run.")
    ap.add_argument("--train_file", default="train.jsonl",
                     help="e.g. train_low_medium.jsonl for the base pass, "
                          "train_high.jsonl for a later adapter pass.")
    ap.add_argument("--eval_file", default="eval.jsonl",
                     help="e.g. eval_low_medium.jsonl or eval_high.jsonl, "
                          "matching --train_file.")
    args = ap.parse_args()

    # ---- 1. Load base model (downloads from HF on first run) ----
    precision = "4-bit (QLoRA)" if args.load_in_4bit else "bf16 (full LoRA)"
    print(f"Loading {args.model_name} in {precision} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=None,  # auto-detect bf16
    )

    # ---- 2. Apply LoRA ----
    # target_modules covers attention + each expert's own FFN weights.
    # Deliberately NOT touching the MoE router (the network that decides
    # which experts fire per token) -- training the router alongside the
    # experts is a well-known way to destabilize MoE routing and wreck the
    # finetune. Only the experts' own weights and attention get adapted.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # Must be 0, not just "low" -- Unsloth's MoE-expert LoRA path (the
        # grouped mlp.experts.gate_up_proj/down_proj parameters used for
        # Qwen3's 128-expert MoE) dispatches to PEFT's ParamWrapper, which
        # raises ValueError on any nonzero dropout. Attention layers would
        # accept dropout fine on their own, but LoRA applies one dropout
        # value across all target_modules in a single get_peft_model call.
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",       # expert FFN (per-expert, not the router)
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ---- 3. Load + format dataset ----
    train_path = find_dataset_file(args.train_file)
    print(f"Loading train set from {train_path}")
    train_rows = load_jsonl_messages(train_path)

    eval_rows = None
    try:
        eval_path = find_dataset_file(args.eval_file)
        print(f"Loading eval set from {eval_path}")
        eval_rows = load_jsonl_messages(eval_path)
    except FileNotFoundError:
        print("No eval.jsonl found, training without a held-out eval set.")

    def to_text(example):
        # Uses the tokenizer's own chat template so tool_calls/tool-result
        # formatting matches what the base model already learned during
        # pretraining, instead of fighting its priors with something custom.
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}

    train_ds = Dataset.from_list(train_rows).map(to_text)
    eval_ds = Dataset.from_list(eval_rows).map(to_text) if eval_rows else None

    # Sanity check: print + save the first rendered example so you can
    # visually confirm the chat template is producing something sane before
    # burning GPU hours on a misformatted dataset.
    preview_path = os.path.join(args.output_dir, "formatted_preview.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(preview_path, "w") as f:
        f.write(train_ds[0]["text"])
    print(f"\n--- First formatted training example (also saved to {preview_path}) ---")
    print(train_ds[0]["text"][:2000])
    print("--- end preview ---\n")

    # ---- 4. Resume from checkpoint if one exists (spot instances can die) ----
    resume_from = None
    existing_checkpoints = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")))
    if existing_checkpoints:
        resume_from = existing_checkpoints[-1]
        print(f"Found existing checkpoint, resuming from {resume_from}")

    # ---- 5. Train ----
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,  # keep each conversation as its own sample (no cross-example bleed)
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # Mask loss to assistant turns only -- see module docstring for why this
    # matters. Qwen models use ChatML-style turn markers.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ---- 6. Save the LoRA adapter ----
    adapter_dir = os.path.join(args.output_dir, "lora_adapter_final")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nDone. LoRA adapter saved to {adapter_dir}")
    print("Next steps: merge the adapter into the base weights, convert to "
          "GGUF via llama.cpp's convert_hf_to_gguf.py, quantize to Q4_K_M, "
          "then build a Modelfile with a TEMPLATE block matching this exact "
          "chat format for Ollama (see project README for the known "
          "--jinja / template gotcha).")


if __name__ == "__main__":
    main()
