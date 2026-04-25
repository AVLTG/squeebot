"""
merge.py — Merge the LoRA adapter into the base Llama 3.1 8B weights, save
as a single fp16 model. This produces the input artifact for the AWQ quantize
step that follows.

Output: fp16 merged model on the Modal Volume at /full/squee-merged-16bit/
        (~16 GB on disk: model + tokenizer files)

Usage:
  modal run fine-tune/merge.py

Cost: ~$1 (A100 40GB for ~15-20 min — picked for safety since fp16 8B is
      ~16 GB in VRAM during merge, tight on A10's 22 GB).
"""

from pathlib import Path

import modal

APP_NAME = "squee-finetune"
VOLUME_NAME = "squee-lora-checkpoints"
HF_SECRET_NAME = "huggingface-secret"

ADAPTER_PATH = "/outputs/full/squee-lora"
MERGED_PATH = "/outputs/full/squee-merged-16bit"
MAX_SEQ_LENGTH = 2048

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
        "unsloth",
        "packaging",
        "wheel",
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    gpu="A100-40GB",  # fp16 8B uses ~16 GB; A100 gives plenty of headroom for merge ops
    timeout=3600,
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def merge() -> None:
    import os

    from unsloth import FastLanguageModel

    print(f"[merge] loading adapter from {ADAPTER_PATH}")
    # Load with 4-bit base (matches how the LoRA was trained). Unsloth's
    # save_pretrained_merged with merged_16bit dequantizes during save.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[merge] adapter loaded. Now merging + saving as fp16...")

    # Unsloth handles the dequant + LoRA merge + fp16 save in one call
    model.save_pretrained_merged(
        MERGED_PATH,
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"[merge] saved. Committing volume...")
    volume.commit()
    print(f"[merge] done. Merged fp16 model at Volume:'{VOLUME_NAME}':{MERGED_PATH}")


@app.local_entrypoint()
def main() -> None:
    merge.remote()
