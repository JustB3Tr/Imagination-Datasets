# Training on Google Colab Pro

Copy each cell below into a Colab notebook, in order.

## 0. Before you run anything

**Runtime → Change runtime type** → GPU: **A100** → check **High-RAM** (this
is what gives you the 80GB variant instead of 40GB — needed for full bf16
LoRA, see `README.md` for why).

## Cell 1 — Mount Drive

Checkpoints and the final adapter need to live somewhere that survives a
disconnect. Colab's local disk doesn't — Drive does.

```python
from google.colab import drive
drive.mount('/content/drive')

OUTPUT_DIR = "/content/drive/MyDrive/imagination2_lora_out"
```

## Cell 2 — Get the code

```python
!git clone https://github.com/JustB3Tr/Imagination-Datasets.git
%cd Imagination-Datasets/training
```

## Cell 3 — Install dependencies

```python
!pip install -q unsloth unsloth_zoo "trl>=0.12.0" "transformers>=4.46.0" datasets accelerate bitsandbytes peft
```

## Cell 4 — Upload the dataset

`train.jsonl` / `eval.jsonl` are gitignored (not in the repo, generated
locally by `dedup_and_split.py`), so they need to come from you directly.
Easiest: upload them straight into this cell's file picker.

```python
from google.colab import files
uploaded = files.upload()  # select train.jsonl and eval.jsonl from your machine
```

If you'd rather not re-upload every session, put them in Drive once instead
(e.g. `/content/drive/MyDrive/imagination2_data/`) and skip this cell — the
script auto-finds `train.jsonl`/`eval.jsonl` in the current directory or a
`data/` subfolder, so just `%cd` there or symlink them in.

## Cell 5 — Smoke test

Always run this before the real thing — confirms the chat template is
rendering correctly and nothing OOMs.

```python
!python train.py --max_steps 20 --output_dir {OUTPUT_DIR}
```

Check the output above for:
- The printed preview of the first formatted example — tool calls, `<think>`
  blocks, system prompt all present and sane?
- No OOM
- Loss printing and not NaN

Then actually look at the saved preview file:

```python
print(open(f"{OUTPUT_DIR}/formatted_preview.txt").read()[:2000])
```

## Cell 6 — Real run (backgrounded, so you can keep using the notebook)

`nohup ... &` inside a Colab cell backgrounds the process instead of
blocking the cell forever — same trick as running it in `tmux` on a real
box. The training itself keeps going even if you close the tab, **as long
as the Colab runtime itself isn't disconnected/recycled** (Pro's idle
timeout is around 90 minutes of tab inactivity, so don't walk away for too
long without checking in).

```python
!nohup python train.py --output_dir {OUTPUT_DIR} > {OUTPUT_DIR}/train.log 2>&1 &
print("Training started in the background. Run the next cell to check on it.")
```

## Cell 7 — Check on it

Run this cell any time to see progress:

```python
!tail -n 40 {OUTPUT_DIR}/train.log
```

## If the runtime disconnects

Re-run Cells 1-3 (mount, clone, install), then Cell 6 again with the exact
same `--output_dir` — the script auto-detects the latest checkpoint under
that directory and resumes from there instead of starting over.

## When it's done

The adapter is at `{OUTPUT_DIR}/lora_adapter_final/`, already durable in
Drive. From there: merge into the base weights, convert to GGUF, quantize
— see the "After training" section in `README.md`.
