# QLoRA training — Imagination 2 Pro

## 1. Rent the GPU

On [Vast.ai](https://vast.ai): pick the **RTX A6000 (48GB)**, and select the
**Unsloth Studio** template when creating the instance. That gets you CUDA /
bitsandbytes / flash-attn / Unsloth already installed correctly — don't pick
a generic PyTorch image and pip-install Unsloth yourself, compiling
flash-attn from source can eat 20-40 minutes of billed GPU time for nothing.

## 2. Upload files

Upload these three files into the same directory on the instance:

- `train_qlora.py` (this script)
- `train.jsonl` (from `dedup_and_split.py`)
- `eval.jsonl` (from `dedup_and_split.py`)

## 3. Smoke test first

```bash
python train_qlora.py --max_steps 20
```

This runs 20 training steps and exits. Check:
- `qlora_out/formatted_preview.txt` — does the rendered chat template look
  right? Tool calls, `<think>` blocks, system prompt all present and sane?
- Did it OOM? If so, lower `--max_seq_length` (e.g. 2048) or confirm you're
  actually on the 48GB card, not a 24GB one.
- Is the loss printing and trending downward, not NaN?

## 4. Real run

Once the smoke test looks right, run it for real (in `tmux` or with
`nohup`, so it survives an SSH disconnect):

```bash
tmux new -s train
python train_qlora.py
# Ctrl+B then D to detach; `tmux attach -t train` to check back in
```

It checkpoints every 50 steps (`--save_steps`) and **auto-resumes** from the
latest checkpoint if you re-run the same command after an interruption
(spot/interruptible instances can get preempted) — just run the exact same
`python train_qlora.py` again.

## 5. After training

The LoRA adapter lands in `qlora_out/lora_adapter_final/`. From there:

1. Merge the adapter into the base weights
2. Convert to GGUF via llama.cpp's `convert_hf_to_gguf.py`
3. Quantize to Q4_K_M
4. Build an Ollama `Modelfile` with a custom `TEMPLATE` block that matches
   this exact chat format — Ollama's default template won't match your
   custom reasoning-level / orchestrator schema, and the model will work
   fine in raw eval but break once Ollama's default templating gets
   applied. Whatever chat template `train_qlora.py` used at training time
   (Qwen's own, via `tokenizer.apply_chat_template`) is the same one that
   needs to end up in the `Modelfile`.

## Hyperparameters, and why

| Param | Value | Why |
|---|---|---|
| Quantization | 4-bit NF4 | Fits a 30B model's weights in ~15GB, leaves headroom for the rest |
| LoRA rank | 16 | This dataset teaches behavior/format, not new knowledge — don't need a huge rank |
| LoRA alpha | 32 (2x rank) | Standard heuristic |
| Target modules | attention + expert FFN, **not the router** | Training the MoE router alongside experts destabilizes routing — well-known failure mode |
| Epochs | 3 | Small dataset (~4.5k examples), more risks overfitting |
| Effective batch size | 16 (1 × 16 grad accum) | VRAM-constrained at batch=1 for a 30B model even in 4-bit |
| Learning rate | 2e-4 | Standard LoRA rate — higher than full finetune since only adapter weights update |
| Loss masking | assistant turns only | Without this the model also trains to predict user/tool messages, diluting signal and risking it learning to hallucinate fake tool outputs |
