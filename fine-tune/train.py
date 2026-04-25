"""
train.py — Fine-tune Llama 3.1 8B Instruct on Squee's voice via Unsloth QLoRA.

Runs on Modal (A10G GPU). Uses the Unsloth pre-quantized 4-bit base so we skip
the quantization-at-load step. Output LoRA adapter is saved to a persistent
Modal Volume.

Usage:
  # Smoke test — 10 steps on 100 examples, ~5 min, ~$0.10 of credits
  modal run fine-tune/train.py --smoke

  # Full training — 3 epochs on 9,497 examples, ~45-75 min, ~$1-2 of credits
  modal run fine-tune/train.py

Prereqs:
  1. `modal token new` has been run and you can see your workspace
  2. Data prepared: `python3 fine-tune/prepare_data.py`
  3. Modal secret `huggingface-secret` with HF_TOKEN set
"""

from pathlib import Path

import modal

APP_NAME = "squee-finetune"
VOLUME_NAME = "squee-lora-checkpoints"
HF_SECRET_NAME = "huggingface-secret"

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

FINE_TUNE_DIR = Path(__file__).resolve().parent
DATA_DIR = FINE_TUNE_DIR / "data"

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
    .add_local_dir(str(DATA_DIR), remote_path="/data")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,  # 24h, Modal's max — full run takes ~12h
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def train(smoke: bool = False) -> dict:
    import os

    # Unsloth must be imported before trl/transformers/peft for its patches to apply
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    print(f"[train] smoke={smoke} | base={BASE_MODEL}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        token=os.environ.get("HF_TOKEN"),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    ds = load_dataset(
        "json",
        data_files={
            "train": "/data/train.jsonl",
            "eval": "/data/eval.jsonl",
        },
    )
    print(f"[train] loaded: train={len(ds['train'])} eval={len(ds['eval'])}")

    if smoke:
        ds["train"] = ds["train"].select(range(100))
        ds["eval"] = ds["eval"].select(range(20))
        print(f"[train] smoke-trimmed: train={len(ds['train'])} eval={len(ds['eval'])}")

    def formatting(examples):
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in examples["messages"]
        ]
        return {"text": texts}

    ds = ds.map(
        formatting,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    output_subdir = "smoke" if smoke else "full"
    output_dir = f"/outputs/{output_subdir}"

    common_sft_kwargs = dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        # Dataset/packing args live on SFTConfig in newer TRL
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
    )

    if smoke:
        sft_args = SFTConfig(
            warmup_steps=2,
            max_steps=10,
            logging_steps=1,
            output_dir=f"{output_dir}/checkpoints",
            **common_sft_kwargs,
        )
    else:
        sft_args = SFTConfig(
            warmup_steps=50,
            num_train_epochs=3,
            logging_steps=10,
            output_dir=f"{output_dir}/checkpoints",
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            **common_sft_kwargs,
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        args=sft_args,
    )

    # Commit volume on every checkpoint save so a container kill never loses
    # more than `save_steps` of progress.
    from transformers.trainer_callback import TrainerCallback

    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            try:
                volume.commit()
                print(f"[volume] committed at step {state.global_step}")
            except Exception as e:
                print(f"[volume] commit failed at step {state.global_step}: {e}")

    trainer.add_callback(VolumeCommitCallback())

    # Resume from the latest committed checkpoint if one exists on the volume.
    checkpoints_dir = f"{output_dir}/checkpoints"
    has_checkpoint = (
        os.path.isdir(checkpoints_dir)
        and any(d.startswith("checkpoint-") for d in os.listdir(checkpoints_dir))
    )
    if has_checkpoint:
        print(f"[train] resuming from latest checkpoint in {checkpoints_dir}")
    stats = trainer.train(resume_from_checkpoint=has_checkpoint)

    adapter_dir = f"{output_dir}/squee-lora"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    volume.commit()

    print(f"[train] done. Adapter saved to Modal Volume '{VOLUME_NAME}' at {adapter_dir}")
    return {
        "mode": "smoke" if smoke else "full",
        "adapter_path": adapter_dir,
        "train_runtime_sec": stats.metrics.get("train_runtime"),
        "train_loss": stats.metrics.get("train_loss"),
    }


@app.local_entrypoint()
def main(smoke: bool = False):
    if not (DATA_DIR / "train.jsonl").exists() or not (DATA_DIR / "eval.jsonl").exists():
        raise FileNotFoundError(
            f"Missing train.jsonl or eval.jsonl in {DATA_DIR}. "
            "Run `python3 fine-tune/prepare_data.py` first."
        )
    result = train.remote(smoke=smoke)
    print("\n=== Training result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
