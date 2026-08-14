#!/usr/bin/env python3
"""
Merge + quant: pulls the low+medium and high LoRA adapters, merges both onto
a freshly-downloaded base model (in memory, back-to-back), converts to f16
GGUF, and quantizes to Q4_K_M -- all producing "Imagination-2.1-Pro-LMH".

Designed to run as a single Colab cell (needs google.colab.drive), not as a
standalone script -- paste the body below into a cell, or run this file
directly in a Colab environment (`!python mq.py`) with Drive already
expected at /content/drive.

Deliberately does almost everything on local Colab disk (/content) instead
of reading/writing through the Drive mount: pulling a ~60GB merged model
back off Drive to convert it to GGUF was the slow part in practice. Here,
only the small adapters get pulled from Drive, the base model comes fresh
from Hugging Face, both adapter merges happen in memory (the low+medium
merge never touches disk at all), and only the final ~18GB Q4_K_M file gets
copied back to Drive at the end. Resumable at every stage -- rerun the
same cell after a disconnect or restart and it picks up where it left off.
"""
import os, shutil, subprocess, sys
from google.colab import drive

drive.mount('/content/drive')

DRIVE = "/content/drive/MyDrive"
DRIVE_LM_ADAPTER_DIR = f"{DRIVE}/imagination2_lora_out/lora_adapter_final"
DRIVE_HIGH_ADAPTER_DIR = f"{DRIVE}/imagination2_lora_out_high/lora_adapter_final"
DRIVE_FINAL_GGUF = f"{DRIVE}/Imagination-2.1-Pro-LMH.gguf"

# Base model this was actually trained on -- matches train.py's --model_name
# default. If you passed a different --model_name for stage 1, change this
# to match, or the merge will silently apply the adapters to the wrong base.
BASE_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

LOCAL_DIR = "/content/imagination2_build"
LOCAL_LM_ADAPTER = f"{LOCAL_DIR}/lm_adapter"
LOCAL_HIGH_ADAPTER = f"{LOCAL_DIR}/high_adapter"
LOCAL_MERGED_LMH = f"{LOCAL_DIR}/merged_lmh"
LOCAL_F16_GGUF = f"{LOCAL_DIR}/Imagination-2.1-Pro-LMH-f16.gguf"
LOCAL_FINAL_GGUF = f"{LOCAL_DIR}/Imagination-2.1-Pro-LMH.gguf"
LLAMACPP_DIR = "/content/llama.cpp"

os.makedirs(LOCAL_DIR, exist_ok=True)


def run(cmd):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def have(pkg):
    return subprocess.run(f"pip show {pkg}", shell=True, capture_output=True).returncode == 0


def dir_complete(d, min_files=2):
    # Local adapter/model dirs are a straight local->local copy or a single
    # save_pretrained call, not resumable multi-part transfers -- "exists
    # and has more than just a stray partial file" is enough of a sanity
    # check here, unlike merge_is_complete's shard-index check for the old
    # Drive-resident merges.
    return os.path.isdir(d) and len(os.listdir(d)) >= min_files


# ---- deps ----
if not have("unsloth"):
    run("pip install -q unsloth unsloth_zoo")
    print("\nFresh install pulled in new package versions -- restarting runtime so they "
          "actually take effect. Just rerun this exact cell after it restarts.")
    os.kill(os.getpid(), 9)

if not os.path.isdir(LLAMACPP_DIR):
    run(f"git clone https://github.com/ggml-org/llama.cpp.git {LLAMACPP_DIR}")

if not have("gguf"):
    run(f"pip install -q -r {LLAMACPP_DIR}/requirements.txt")

if not os.path.isfile(f"{LLAMACPP_DIR}/build/bin/llama-quantize"):
    run(f"cmake -B {LLAMACPP_DIR}/build -S {LLAMACPP_DIR} -DGGML_CUDA=OFF")
    run(f"cmake --build {LLAMACPP_DIR}/build --config Release -j --target llama-quantize")

# llama.cpp's requirements.txt (or the unsloth install) can leave a numpy
# version installed that's binary-incompatible with the numpy ABI torch/
# unsloth's compiled extensions were already loaded against in this runtime.
# Gating on package presence (like the unsloth branch above) doesn't work
# here because pip installs persist across reruns even though the runtime
# process doesn't restart, so a marker file (wiped on a fresh runtime, same
# lifecycle as everything else in /content) tracks whether we've actually
# restarted since the version changed.
NUMPY_FIX_MARKER = "/content/.numpy_fixed"
if not os.path.isfile(NUMPY_FIX_MARKER):
    run("pip install -q 'numpy<2'")
    open(NUMPY_FIX_MARKER, "w").close()
    print("\nPinned numpy to a version compatible with torch/unsloth's compiled "
          "extensions -- restarting runtime so it actually takes effect. Just rerun "
          "this exact cell after it restarts.")
    os.kill(os.getpid(), 9)

# ---- figure out what's already done, resume from there ----
if os.path.isfile(DRIVE_FINAL_GGUF):
    print(f"\nAlready done: {DRIVE_FINAL_GGUF}")
    run(f"ls -la '{DRIVE_FINAL_GGUF}'")
    sys.exit(0)

if not os.path.isfile(LOCAL_FINAL_GGUF):
    if not os.path.isfile(LOCAL_F16_GGUF):
        # ---- Step 1: pull just the (small) adapters from Drive ----
        if not dir_complete(LOCAL_LM_ADAPTER):
            print("\n=== Step 1a: copying low+medium adapter from Drive ===")
            shutil.rmtree(LOCAL_LM_ADAPTER, ignore_errors=True)
            shutil.copytree(DRIVE_LM_ADAPTER_DIR, LOCAL_LM_ADAPTER)
        else:
            print(f"\n{LOCAL_LM_ADAPTER} already present locally, skipping copy.")

        if not dir_complete(LOCAL_HIGH_ADAPTER):
            print("\n=== Step 1b: copying high adapter from Drive ===")
            shutil.rmtree(LOCAL_HIGH_ADAPTER, ignore_errors=True)
            shutil.copytree(DRIVE_HIGH_ADAPTER_DIR, LOCAL_HIGH_ADAPTER)
        else:
            print(f"\n{LOCAL_HIGH_ADAPTER} already present locally, skipping copy.")

        # ---- Step 2: base model fresh from HF, both adapters merged in
        # memory back-to-back -- merged_lm never touches disk at all, only
        # the final low+medium+high result gets saved, once, locally ----
        if not dir_complete(LOCAL_MERGED_LMH):
            print(f"\n=== Step 2: downloading base ({BASE_MODEL_NAME}) and merging both adapters ===")
            from unsloth import FastLanguageModel
            from peft import PeftModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                BASE_MODEL_NAME, max_seq_length=4096, load_in_4bit=False,
            )
            model = PeftModel.from_pretrained(model, LOCAL_LM_ADAPTER)
            model = model.merge_and_unload()
            print("Low+medium merged in memory.")

            model = PeftModel.from_pretrained(model, LOCAL_HIGH_ADAPTER)
            model = model.merge_and_unload()
            print("High merged in memory.")

            model.save_pretrained(LOCAL_MERGED_LMH, safe_serialization=True)
            tokenizer.save_pretrained(LOCAL_MERGED_LMH)
            print(f"Final merged model saved locally to {LOCAL_MERGED_LMH}")
            if not dir_complete(LOCAL_MERGED_LMH):
                raise RuntimeError(f"Merge finished but {LOCAL_MERGED_LMH} still looks incomplete -- check local disk space.")

            del model
            import torch
            torch.cuda.empty_cache()
        else:
            print(f"\nMerged low+medium+high model already found locally at {LOCAL_MERGED_LMH}, skipping.")

        print("\n=== Step 3: converting to f16 GGUF (local read, fast) ===")
        run(f"python {LLAMACPP_DIR}/convert_hf_to_gguf.py '{LOCAL_MERGED_LMH}' --outfile '{LOCAL_F16_GGUF}' --outtype f16")

        # Free space before quantizing -- LOCAL_MERGED_LMH (~60GB bf16) is no
        # longer needed once the f16 GGUF exists, and default Colab local
        # disk is tight once you're holding a base model + merged model +
        # f16 GGUF + Q4_K_M GGUF all at once.
        print(f"\nRemoving {LOCAL_MERGED_LMH} to free local disk space (f16 GGUF has everything needed now).")
        shutil.rmtree(LOCAL_MERGED_LMH, ignore_errors=True)
    else:
        print(f"\nf16 GGUF already found locally at {LOCAL_F16_GGUF}, skipping merge+convert.")

    print("\n=== Step 4: quantizing to Q4_K_M (local read+write, fast) ===")
    run(f"{LLAMACPP_DIR}/build/bin/llama-quantize '{LOCAL_F16_GGUF}' '{LOCAL_FINAL_GGUF}' Q4_K_M")

    print(f"\nRemoving {LOCAL_F16_GGUF} to free local disk space (Q4_K_M has everything needed now).")
    os.remove(LOCAL_F16_GGUF)
else:
    print(f"\nQ4_K_M GGUF already found locally at {LOCAL_FINAL_GGUF}, skipping straight to Drive copy.")

# ---- Step 5: only the small quantized file goes back to Drive ----
print(f"\n=== Step 5: copying final Q4_K_M GGUF to Drive ===")
shutil.copy2(LOCAL_FINAL_GGUF, DRIVE_FINAL_GGUF)

print("\n=== DONE ===")
run(f"ls -la '{DRIVE_FINAL_GGUF}'")
