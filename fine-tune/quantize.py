"""
quantize.py — AWQ 4-bit quantization of the merged fp16 Squee model.

Reads the merged fp16 model from the Modal Volume, runs activation-aware
weight quantization using 128 samples from our training set as calibration,
writes the AWQ artifact back to the volume.

Output: AWQ-quantized model on the volume at /full/squee-awq-4bit (~5 GB).
        Loadable by vLLM (next step) for serving.

Usage:
  modal run fine-tune/quantize.py

Cost: ~$1.50-$3 (A100 40GB for ~30-60 min — fp16 8B during calibration plus
      activation memory peaks at ~25-30 GB; A10's 22 GB would be tight).
"""

from pathlib import Path

import modal

APP_NAME = "squee-finetune"
VOLUME_NAME = "squee-lora-checkpoints"
HF_SECRET_NAME = "huggingface-secret"

MERGED_PATH = "/outputs/full/squee-merged-16bit"
AWQ_PATH = "/outputs/full/squee-awq-4bit"
N_CALIBRATION_SAMPLES = 128
# Must be >= the longest calibration sample's token count, NOT a truncation cap.
# autoawq's get_calib_dataset *drops* samples longer than this (doesn't truncate).
# Our samples are ~1,240 tokens (1,160 system prompt + user + assistant + template),
# so 2048 keeps all of them. Matches our training max_seq_length too.
CALIB_MAX_SEQ_LEN = 2048

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "autoawq",
        "transformers",
        "accelerate",
        "packaging",
        "wheel",
    )
    .add_local_dir(str(DATA_DIR), remote_path="/data")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3 * 3600,
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def quantize() -> None:
    import json
    import os

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"[quantize] loading tokenizer + merged model from {MERGED_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MERGED_PATH,
        token=os.environ.get("HF_TOKEN"),
    )
    model = AutoAWQForCausalLM.from_pretrained(
        MERGED_PATH,
        safetensors=True,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[quantize] model loaded")

    # Build calibration dataset: first N samples from train.jsonl,
    # rendered through the model's chat template (the format the served
    # model will see at inference time).
    print(f"[quantize] preparing {N_CALIBRATION_SAMPLES} calibration samples")
    calib_data: list[str] = []
    with open("/data/train.jsonl") as f:
        for i, line in enumerate(f):
            if i >= N_CALIBRATION_SAMPLES:
                break
            ex = json.loads(line)
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            calib_data.append(text)
    print(f"[quantize] {len(calib_data)} calibration samples ready")

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",  # GEMM kernel — fastest at inference, broadly supported by vLLM
    }

    print(f"[quantize] running AWQ (calibration + quantization, ~30-60 min)...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
        max_calib_seq_len=CALIB_MAX_SEQ_LEN,
    )
    print(f"[quantize] quantization complete. saving...")

    model.save_quantized(AWQ_PATH)
    tokenizer.save_pretrained(AWQ_PATH)

    print(f"[quantize] committing volume")
    volume.commit()
    print(f"[quantize] done. AWQ artifact at Volume:'{VOLUME_NAME}':{AWQ_PATH}")


@app.local_entrypoint()
def main() -> None:
    quantize.remote()
