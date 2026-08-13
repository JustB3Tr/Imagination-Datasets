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


# A real fraction of generated rows have a tool_call function.arguments
# string that isn't valid JSON -- usually an under-escaped backslash inside
# a regex/command/path (e.g. a regex \K written as \K instead of \\K) or a
# raw literal newline where \n was needed. Both are recoverable without
# guessing at content. Rows with a genuinely unescaped quote character
# (e.g. `grep -R "pip install"` with the inner quotes never escaped) are
# NOT recoverable this way -- the string boundary itself is ambiguous, and
# guessing where a quote belongs risks silently corrupting the example, so
# those still get dropped below. This mirrors repair_tool_call_arguments in
# dedup_and_split.py -- kept as a fallback here too, for anyone running
# train.py directly against freshly-generated raw data that hasn't gone
# through dedup_and_split.py yet.
_JSON_VALID_ESCAPES = set('"\\/bfnrtu')


def _try_parse_dict(s: str, strict: bool = True):
    try:
        obj = json.loads(s, strict=strict)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _repair_tool_call_arguments(s: str) -> str | None:
    if _try_parse_dict(s) is not None or _try_parse_dict(s, strict=False) is not None:
        return s
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] not in _JSON_VALID_ESCAPES:
            out.append("\\\\")
            i += 1
            continue
        out.append(c)
        i += 1
    fixed = "".join(out)
    if _try_parse_dict(fixed) is not None or _try_parse_dict(fixed, strict=False) is not None:
        return fixed
    return None


def normalize_message(m: dict) -> dict:
    """Coerce a message dict to a fixed, consistent shape. A handful of
    earlier-generated rows have divergent shapes -- some tool_calls store
    "arguments" as a raw dict instead of the required JSON-encoded string,
    and some messages carry stray extra keys ("name", "status",
    "additional_props", or "message" instead of "content"). datasets.
    Dataset.from_list infers one PyArrow struct schema across every message
    in the column, so any shape variance raises
    "ArrowInvalid: cannot mix struct and non-struct, non-null values" --
    normalizing every message to the same key set/types here fixes it
    without needing to regenerate the affected rows."""
    role = m.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        # e.g. a stray "tool_calls" role on one bad row -- should always be
        # "assistant" in practice (nothing else emits tool_calls), and an
        # unrecognized role falls into the chat template's generic catch-all
        # branch, which does unconditional content concatenation same as the
        # None-content-no-tool_calls case below.
        role = "assistant"

    content = m.get("content")
    if content is None and "message" in m:
        content = m.get("message")  # rare typo'd field from one bad row
    tool_calls = m.get("tool_calls")
    if content is None and not tool_calls:
        # A message with neither content nor tool_calls can't be rendered --
        # the chat template does `message.content` string concatenation
        # unconditionally for any assistant/user/system message that isn't
        # in the tool_calls branch, and None + str crashes with
        # "can only concatenate str (not 'NoneType') to str". None content
        # is only valid when tool_calls is present (the template's
        # tool_calls branch explicitly guards for that case).
        content = ""
    out = {"role": role, "content": content}
    if tool_calls:
        fixed_calls = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, (dict, list)):
                args = json.dumps(args)
            elif isinstance(args, str) and _try_parse_dict(args) is None:
                repaired = _repair_tool_call_arguments(args)
                if repaired is not None:
                    args = repaired
            fixed_calls.append({
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {"name": fn.get("name"), "arguments": args},
            })
        out["tool_calls"] = fixed_calls
    if m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    return out


def has_unparseable_arguments(messages: list[dict]) -> bool:
    """True if any tool_call's function.arguments string isn't valid JSON
    that decodes to a dict, after normalize_message's repair pass. Qwen's
    chat_template does `tool_call.arguments|items`, which requires a real
    dict at render time -- rows with a genuinely unescaped quote character
    aren't safely repairable, so those still get dropped here rather than
    crashing the whole training run or feeding the tokenizer malformed
    data."""
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            args = (tc.get("function") or {}).get("arguments")
            if not isinstance(args, str):
                continue
            if _try_parse_dict(args) is None:
                return True
    return False


def load_jsonl_messages(path: str) -> list[dict]:
    examples = []
    n_dropped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # meta (domain/level/mode/seed_task) isn't needed for training,
            # only the actual conversation.
            messages = [normalize_message(m) for m in row["messages"]]
            if has_unparseable_arguments(messages):
                n_dropped += 1
                continue
            examples.append({"messages": messages})
    if n_dropped:
        print(f"Dropped {n_dropped} example(s) from {path}: "
              f"tool_call arguments weren't valid JSON (under-escaped backslashes).")
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
    ap.add_argument("--save_total_limit", type=int, default=None,
                     help="Max checkpoints to keep (oldest deleted first). "
                          "Default keeps ALL checkpoints -- fine if you have "
                          "the disk space (each is a few GB), and means "
                          "nothing is ever lost to rotation. Pass a number "
                          "(e.g. 3) to cap it if space is tight.")
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

    def for_template(messages):
        # Qwen's chat_template.jinja does
        # `{% for args_name, args_value in tool_call.arguments|items %}`,
        # which requires tool_call.arguments to be an actual dict -- but the
        # dataset stores it as a JSON-encoded string (the correct format for
        # the JSONL/Arrow data itself, and what OpenAI's tool-calling API
        # expects on the wire). Parse it back to a dict here, at render time
        # only, so this doesn't fight the stored format load_jsonl_messages
        # normalized everything to.
        rendered = []
        for m in messages:
            m2 = dict(m)
            tool_calls = m2.get("tool_calls")
            if tool_calls:
                new_calls = []
                for tc in tool_calls:
                    tc2 = dict(tc)
                    fn = dict(tc2.get("function") or {})
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            pass  # leave as-is; template will surface the bad row
                    tc2["function"] = fn
                    new_calls.append(tc2)
                m2["tool_calls"] = new_calls
            rendered.append(m2)
        return rendered

    def to_text(example):
        # Uses the tokenizer's own chat template so tool_calls/tool-result
        # formatting matches what the base model already learned during
        # pretraining, instead of fighting its priors with something custom.
        return {"text": tokenizer.apply_chat_template(
            for_template(example["messages"]), tokenize=False, add_generation_prompt=False
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
    existing_checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
    if existing_checkpoints:
        # Sort numerically by step count, not lexicographically -- plain
        # string sort would rank "checkpoint-50" after "checkpoint-100" and
        # "checkpoint-150" (since '5' > '1' as the first differing char),
        # so with save_total_limit rotation there's a real window right
        # after crossing a digit-length boundary (e.g. just after step 150
        # saves, {50, 100, 150} briefly coexist) where the old string sort
        # would silently resume from the wrong, older checkpoint.
        existing_checkpoints.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
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
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        # Without this, the trainer just keeps whatever checkpoint happens to
        # be last, which is NOT the same as the best one -- a real run on
        # this dataset bottomed out on eval_loss around 1 epoch in (~step
        # 300) and then overfit for the remaining 2 epochs, ending 24% worse
        # on eval_loss than the best point it had already passed. With this
        # set, the trainer tracks eval_loss across every eval_steps interval
        # and reloads the best checkpoint's weights at the very end of
        # training, before save_pretrained() writes lora_adapter_final --
        # and it protects that checkpoint from save_total_limit's rotation
        # even if it's no longer among the most recent saves.
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
