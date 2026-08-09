# LoRA training — Imagination 2 Pro

## 1. Rent the GPU

On [Vast.ai](https://vast.ai): pick an **A100 80GB**, and select the
**Unsloth Studio** template when creating the instance. That gets you CUDA /
bitsandbytes / flash-attn / Unsloth already installed correctly — don't pick
a generic PyTorch image and pip-install Unsloth yourself, compiling
flash-attn from source can eat 20-40 minutes of billed GPU time for nothing.

Why 80GB and not the cheaper 24-48GB cards: this runs full LoRA on the
**bf16** (not quantized) base model, which is ~60GB for the frozen weights
alone before adapters/activations/optimizer state — doesn't fit on a 24-48GB
card. Current Vast.ai rates put an A100 80GB around $0.67-1.00/hr; a full
training run is realistically a few hours, so this is cheap relative to the
project's overall compute budget. If you ever want to run on a smaller/
cheaper card instead, pass `--load_in_4bit` to fall back to QLoRA — the
quality difference vs full LoRA is normally small, this setup just spends a
bit more to get the extra edge since the budget allows it.

## 2. Upload files

Upload these three files into the same directory on the instance:

- `train.py` (this script)
- `train.jsonl` (from `dedup_and_split.py`)
- `eval.jsonl` (from `dedup_and_split.py`)

## 3. Smoke test first

```bash
python train.py --max_steps 20
```

This runs 20 training steps and exits. Check:
- `lora_out/formatted_preview.txt` — does the rendered chat template look
  right? Tool calls, `<think>` blocks, system prompt all present and sane?
- Did it OOM? If so, lower `--max_seq_length` (e.g. 2048), or fall back to
  `--load_in_4bit` if you're not actually on an 80GB card.
- Is the loss printing and trending downward, not NaN?

## 4. Real run

Once the smoke test looks right, run it for real (in `tmux` or with
`nohup`, so it survives an SSH disconnect):

```bash
tmux new -s train
python train.py
# Ctrl+B then D to detach; `tmux attach -t train` to check back in
```

It checkpoints every 50 steps (`--save_steps`) and **auto-resumes** from the
latest checkpoint if you re-run the same command after an interruption
(spot/interruptible instances can get preempted) — just run the exact same
`python train.py` again.

## 5. After training

The LoRA adapter lands in `lora_out/lora_adapter_final/`. From there:

1. Merge the adapter into the base weights
2. Convert to GGUF via llama.cpp's `convert_hf_to_gguf.py`
3. Quantize to Q4_K_M
4. Build an Ollama `Modelfile` with a custom `TEMPLATE` block that matches
   this exact chat format — Ollama's default template won't match your
   custom reasoning-level / orchestrator schema, and the model will work
   fine in raw eval but break once Ollama's default templating gets
   applied. Whatever chat template `train.py` used at training time
   (Qwen's own, via `tokenizer.apply_chat_template`) is the same one that
   needs to end up in the `Modelfile`.

## Hyperparameters, and why

| Param | Value | Why |
|---|---|---|
| Precision | bf16 base, full LoRA | Budget allows an 80GB card; avoids the small quality cost of 4-bit quantization. Pass `--load_in_4bit` to fall back to QLoRA on a cheaper/smaller GPU if needed. |
| LoRA rank | 16 | This dataset teaches behavior/format, not new knowledge — don't need a huge rank |
| LoRA alpha | 32 (2x rank) | Standard heuristic |
| Target modules | attention + expert FFN, **not the router** | Training the MoE router alongside experts destabilizes routing — well-known failure mode |
| Epochs | 3 | Small dataset (~4.5k examples), more risks overfitting |
| Effective batch size | 16 (1 × 16 grad accum) | VRAM-constrained at batch=1 for a 30B model regardless of base precision |
| Learning rate | 2e-4 | Standard LoRA rate — higher than full finetune since only adapter weights update |
| Loss masking | assistant turns only | Without this the model also trains to predict user/tool messages, diluting signal and risking it learning to hallucinate fake tool outputs |
