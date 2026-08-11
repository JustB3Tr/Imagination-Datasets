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

`data/train_low_medium.jsonl` / `data/eval_low_medium.jsonl` (and the
`_high` pair, for the later adapter pass) ARE checked into the repo, so
Cell 2's clone already has them under `Imagination-Datasets/data/` — no
upload needed. This cell is only for when you're working from a dataset
version that isn't pushed yet:

```python
from google.colab import files
uploaded = files.upload()  # select the .jsonl files from your machine
```

## Cell 5 — Smoke test

Always run this before the real thing — confirms the chat template is
rendering correctly and nothing OOMs. This first pass trains on the
low+medium split (the cleaner set per the judge audit); the high split is
a separate adapter pass, see "Stage 2" below.

```python
!python train.py --max_steps 20 --output_dir {OUTPUT_DIR} \
  --train_file ../data/train_low_medium.jsonl \
  --eval_file ../data/eval_low_medium.jsonl
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

Use Python's own `subprocess.Popen` to launch it, not `!nohup ... &`.
Colab's `!` shell-out magic doesn't reliably detach a background job the
way a real terminal does -- `!nohup ... &` can still leave the cell
blocking for the full training run instead of returning immediately.
`subprocess.Popen` sidesteps that entirely and returns control right away:

```python
import subprocess

log_path = f"{OUTPUT_DIR}/train.log"
proc = subprocess.Popen(
    ["python", "train.py", "--output_dir", OUTPUT_DIR,
     "--train_file", "../data/train_low_medium.jsonl",
     "--eval_file", "../data/eval_low_medium.jsonl"],
    stdout=open(log_path, "w"),
    stderr=subprocess.STDOUT,
)
print(f"Training started in background, PID {proc.pid}")
```

The cell finishes instantly (you'll see the PID print right away) while
training keeps running behind it. The process survives closing the tab,
**as long as the Colab runtime itself isn't disconnected/recycled** (Pro's
idle timeout is around 90 minutes of tab inactivity, so don't walk away
for too long without checking in).

## Cell 7 — Check on it

Run this cell any time to see progress:

```python
!tail -n 50 {OUTPUT_DIR}/train.log
```

## If the runtime disconnects

Re-run Cells 1-3 (mount, clone, install), then Cell 6 again with the exact
same `--output_dir` — the script auto-detects the latest checkpoint under
that directory and resumes from there instead of starting over.

## When it's done

The adapter is at `{OUTPUT_DIR}/lora_adapter_final/`, already durable in
Drive. From there: merge into the base weights, convert to GGUF, quantize
— see the "After training" section in `README.md`.

## Stage 2 (later) — high-level adapter on top

The judge audit found `high`-level examples noticeably weaker on
level_match than low/medium (the `<think>` blocks often skip the required
"name an alternative, reject it" step) — see `data/train_high.jsonl` /
`data/eval_high.jsonl`, held out of stage 1 for this reason. Once stage 1
is done and you're happy with it, train a second adapter on top instead of
starting over:

**Cell A — merge stage 1's adapter into the base weights** (makes it the
new "base" for stage 2; run once, save the merged model to Drive so you
don't redo it):

```python
from unsloth import FastLanguageModel

MERGED_DIR = "/content/drive/MyDrive/imagination2_merged_low_medium"
model, tokenizer = FastLanguageModel.from_pretrained(f"{OUTPUT_DIR}/lora_adapter_final")
model = model.merge_and_unload()
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
```

**Cell B — train the high-level adapter on top of the merged model:**

```python
STAGE2_OUTPUT_DIR = "/content/drive/MyDrive/imagination2_lora_out_high"
!python train.py --model_name {MERGED_DIR} --output_dir {STAGE2_OUTPUT_DIR} \
  --train_file ../data/train_high.jsonl \
  --eval_file ../data/eval_high.jsonl \
  --max_steps 20   # smoke test first, then drop this flag for the real run
```

Same smoke-test-then-real-run, same backgrounding/resume rules as stage 1.
Repeat this same Cell A/B pattern for any future "extra high" tier — merge
whatever the current best checkpoint is, then train the next adapter on
top of that.
