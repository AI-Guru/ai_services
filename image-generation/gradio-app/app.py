"""Gradio frontend for the self-hosted text-to-image services.

Talks to one or more diffusers model containers over their OpenAI-compatible
`/v1/images/generations` endpoints (see ../diffusers-server/). The model list is
config-driven via the MODELS_JSON env var, so the dropdown grows as containers
are added — no code change needed.

  MODELS_JSON  JSON list of {"name": ..., "url": "http://host:port/v1"} entries
"""
import base64
import io
import json
import os

import gradio as gr
import requests
from PIL import Image

DEFAULT_MODELS = [{"name": "z-image-turbo", "url": "http://z-image-turbo:8101/v1"}]
MODELS = json.loads(os.environ.get("MODELS_JSON", json.dumps(DEFAULT_MODELS)))
URL_BY_NAME = {m["name"]: m["url"] for m in MODELS}
SIZES = ["512x512", "768x768", "1024x1024", "1024x1536", "1536x1024"]
MAX_BATCH = int(os.environ.get("MAX_BATCH", "8"))


def model_defaults(model):
    """Pull per-model generation defaults so the sliders match the selected model
    (e.g. Z-Image: 9 steps / CFG 0; Qwen-Image: 50 steps / true-CFG 4)."""
    base = URL_BY_NAME.get(model)
    try:
        c = requests.get(f"{base}/config", timeout=5).json()
        label = "Guidance (true-CFG)" if c["guidance_param"] == "true_cfg_scale" else "Guidance (CFG)"
        return (
            gr.update(value=c["default_steps"]),
            gr.update(value=c["default_guidance"], label=label),
            gr.update(value=c["default_size"]),
        )
    except requests.exceptions.RequestException:
        # container for this model may be down (one-at-a-time on the GPU) — leave sliders as-is
        return gr.update(), gr.update(), gr.update()


def generate(model, prompt, negative_prompt, size, batch, steps, guidance, seed):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    base = URL_BY_NAME.get(model)
    if not base:
        raise gr.Error(f"Unknown model: {model}")
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": int(batch),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "response_format": "b64_json",
    }
    if negative_prompt.strip():
        payload["negative_prompt"] = negative_prompt
    if int(seed) >= 0:
        payload["seed"] = int(seed)

    try:
        r = requests.post(f"{base}/images/generations", json=payload, timeout=600)
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Could not reach {model} at {base} — is the container up? ({e})")
    if r.status_code != 200:
        raise gr.Error(f"{model} returned {r.status_code}: {r.text[:300]}")
    return [Image.open(io.BytesIO(base64.b64decode(d["b64_json"]))) for d in r.json()["data"]]


with gr.Blocks(title="AI Services — Text to Image") as demo:
    gr.Markdown("# 🖼️ AI Services — Text to Image\nSelf-hosted open-weights image generation.")
    with gr.Row():
        with gr.Column(scale=2):
            model = gr.Dropdown(
                choices=list(URL_BY_NAME), value=list(URL_BY_NAME)[0], label="Model"
            )
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="A photorealistic ...")
            negative_prompt = gr.Textbox(label="Negative prompt (optional)", lines=1)
            with gr.Row():
                size = gr.Dropdown(choices=SIZES, value="1024x1024", label="Size")
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
            batch = gr.Slider(1, MAX_BATCH, value=1, step=1, label="Batch size (parallel images)")
            with gr.Row():
                steps = gr.Slider(1, 50, value=9, step=1, label="Steps")
                guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Guidance (CFG)")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            out = gr.Gallery(label="Results", height=640, columns=2, object_fit="contain")

    gr.Markdown(
        "**Tips** — Z-Image-Turbo: keep Steps≈9 and Guidance=0.0 (it's distilled). "
        "Other models may need more steps and Guidance≈3–5. "
        "Batch size generates N images in one parallel pass — larger batches use "
        "more VRAM. A fixed seed makes the whole batch reproducible (images still "
        "differ within the batch)."
    )
    go.click(
        generate,
        inputs=[model, prompt, negative_prompt, size, batch, steps, guidance, seed],
        outputs=out,
    )
    # adapt sliders to the model's own defaults on switch and on initial load
    model.change(model_defaults, inputs=model, outputs=[steps, guidance, size])
    demo.load(model_defaults, inputs=model, outputs=[steps, guidance, size])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
