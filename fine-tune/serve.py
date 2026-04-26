"""
serve.py — vLLM Modal app serving the Squee AWQ model.

Exposes an OpenAI-compatible HTTP endpoint at /v1/chat/completions (and
the other standard OpenAI routes). Bearer-auth via an API key stored in
the Modal secret `squee-vllm-key`.

Scales to zero after 5 min idle. Cold start is ~45-90 sec (model load +
CUDA graph compile). Warm requests are sub-second.

Usage:
  # Deploy (run once; URL is permanent until you stop the app):
  modal deploy fine-tune/serve.py

  # The URL Modal prints will look like:
  #   https://<workspace>--squee-vllm-serve.modal.run
  # That's the base URL the bot will hit at /v1/chat/completions.

  # Stop the deployment:
  modal app stop squee-vllm

Cost: A10 at ~$1.10/hr while warm. With ~50-100 req/day clustered in
      active hours, expect $0.10-$0.50/day.
"""

import os
import subprocess

import modal

APP_NAME = "squee-vllm"
VOLUME_NAME = "squee-lora-checkpoints"
HF_SECRET_NAME = "huggingface-secret"
VLLM_KEY_SECRET = "squee-vllm-key"

AWQ_PATH = "/outputs/full/squee-awq-4bit"
SERVED_NAME = "squee"   # the model name the bot will reference in API requests
PORT = 8000
MAX_MODEL_LEN = 2048    # matches our training max_seq_length

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm",
        "packaging",
        "wheel",
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/outputs": volume},
    secrets=[
        modal.Secret.from_name(HF_SECRET_NAME),
        modal.Secret.from_name(VLLM_KEY_SECRET),
    ],
    scaledown_window=300,        # idle for 5 min → scale to zero
    timeout=24 * 3600,           # request timeout ceiling (we'll never hit this)
)
@modal.concurrent(max_inputs=10) # vLLM batches internally; 10 concurrent reqs / container
@modal.web_server(port=PORT, startup_timeout=300)
def serve() -> None:
    api_key = os.environ["VLLM_API_KEY"]
    cmd = [
        "vllm", "serve", AWQ_PATH,
        "--quantization", "awq_marlin",  # faster AWQ kernel; works on the same artifact
        "--dtype", "float16",             # AWQ requires fp16 (model config has bf16 from Unsloth)
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--served-model-name", SERVED_NAME,
        "--api-key", api_key,
        "--gpu-memory-utilization", "0.9",
    ]
    print(f"[serve] launching vLLM (model={SERVED_NAME}, port={PORT})")
    # Popen, not run — Modal's web_server decorator expects this function to
    # return after launching the server, then waits for the port to come up.
    subprocess.Popen(cmd)
