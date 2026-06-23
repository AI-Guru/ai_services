#!/usr/bin/env python3
"""Simple Gradio test app for the self-hosted image pipeline.

Three tabs, all talking to the local OpenAI-compatible / REST endpoints:
  - Generate : Z-Image-Turbo text-to-image   (POST /v1/images/generations)
  - Edit     : Z-Image-Turbo img2img edit     (POST /v1/images/edits)
  - Upscale  : Real-ESRGAN x2/x4              (POST /upscale)

It's a thin HTTP client — no GPU, no model code here. Run it anywhere that can
reach the services. Deps come from the repo-root pyproject's optional group:

    pip install -e ".[gradio]"      # run from repo root
    python models/z-image/gradio_app.py
    # open http://localhost:7861

Override endpoints if not on the same host:
    GEN_URL=http://192.168.0.161:11476  UPSCALE_URL=http://192.168.0.161:11477  python gradio_app.py
"""
from __future__ import annotations

import base64
import io
import os

import gradio as gr
import requests
from PIL import Image

GEN_URL = os.environ.get("GEN_URL", "http://localhost:11476").rstrip("/")
UPSCALE_URL = os.environ.get("UPSCALE_URL", "http://localhost:11477").rstrip("/")
MODEL = os.environ.get("GEN_MODEL", "z-image-turbo")
TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))


def _decode(resp: requests.Response) -> Image.Image:
    resp.raise_for_status()
    b64 = resp.json()["data"][0]["b64_json"]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def generate(prompt, negative, size, steps, guidance, seed):
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "size": size,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "negative_prompt": negative or "",
        "n": 1,
    }
    if int(seed) >= 0:
        payload["seed"] = int(seed)
    return _decode(requests.post(f"{GEN_URL}/v1/images/generations", json=payload, timeout=TIMEOUT))


def edit(image, prompt, steps, guidance, seed):
    if image is None:
        raise gr.Error("Upload an image to edit.")
    if not prompt.strip():
        raise gr.Error("Enter an edit instruction.")
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    data = {
        "model": MODEL,
        "prompt": prompt,
        "num_inference_steps": str(int(steps)),
        "guidance_scale": str(float(guidance)),
    }
    if int(seed) >= 0:
        data["seed"] = str(int(seed))
    files = {"image": ("input.png", buf, "image/png")}
    return _decode(requests.post(f"{GEN_URL}/v1/images/edits", data=data, files=files, timeout=TIMEOUT))


def upscale(image, scale):
    if image is None:
        raise gr.Error("Upload an image to upscale.")
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("input.png", buf, "image/png")}
    resp = requests.post(
        f"{UPSCALE_URL}/upscale", files=files, data={"scale": int(scale)}, timeout=TIMEOUT
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


with gr.Blocks(title="ai_services image playground") as demo:
    gr.Markdown(f"# 🖼️ ai_services image playground\nGenerator: `{MODEL}` @ {GEN_URL} · Upscaler @ {UPSCALE_URL}")

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column():
                g_prompt = gr.Textbox(label="Prompt", lines=3, value="cyberpunk church interior, neon stained glass, volumetric fog")
                g_negative = gr.Textbox(label="Negative prompt", value="blurry, low quality, distorted")
                g_size = gr.Dropdown(["512x512", "768x768", "1024x1024", "1280x1280"], value="1024x1024", label="Size")
                with gr.Row():
                    g_steps = gr.Slider(1, 30, value=8, step=1, label="Steps (Turbo ≈ 8)")
                    g_guidance = gr.Slider(0.0, 7.5, value=1.0, step=0.1, label="Guidance (Turbo ≈ 1.0)")
                g_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                g_btn = gr.Button("Generate", variant="primary")
            g_out = gr.Image(label="Result", type="pil")
        g_btn.click(generate, [g_prompt, g_negative, g_size, g_steps, g_guidance, g_seed], g_out)

    with gr.Tab("Edit (img2img)"):
        with gr.Row():
            with gr.Column():
                e_in = gr.Image(label="Input image", type="pil")
                e_prompt = gr.Textbox(label="Edit instruction", lines=2, value="make it neon cyberpunk, glowing")
                with gr.Row():
                    e_steps = gr.Slider(1, 30, value=8, step=1, label="Steps")
                    e_guidance = gr.Slider(0.0, 7.5, value=1.0, step=0.1, label="Guidance")
                e_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                e_btn = gr.Button("Edit", variant="primary")
            e_out = gr.Image(label="Result", type="pil")
        e_btn.click(edit, [e_in, e_prompt, e_steps, e_guidance, e_seed], e_out)

    with gr.Tab("Upscale"):
        with gr.Row():
            with gr.Column():
                u_in = gr.Image(label="Input image", type="pil")
                u_scale = gr.Radio([2, 4], value=4, label="Scale")
                u_btn = gr.Button("Upscale", variant="primary")
            u_out = gr.Image(label="Result", type="pil")
        u_btn.click(upscale, [u_in, u_scale], u_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7861")))
