#!/usr/bin/env python3
"""
Gradio front-end for the resident Stable Audio 3 Small-SFX server.

Mounted onto the SAME FastAPI app as the REST API (server.py calls mount_ui at import
time), so the UI shares the one loaded model and the one generation lock.

  http://localhost:8400/       -> redirect
  http://localhost:8400/ui     -> this
  http://localhost:8400/v1/... -> REST, unchanged

Nothing here reaches into the model: mount_ui receives the state dict, the request model
and the generate function as arguments. That keeps `python3 server.py` (module name
__main__) from being re-imported as a second module with a second STATE.
"""
import asyncio, json, os, shutil, time, traceback

import gradio as gr

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a")

# Prompting for SFX is not prompting for music. The model responds to the physical event
# and the space it happens in — material, action, distance, room — far more than to genre
# or mood words. These follow the shape of upstream's prompting guide.
EXAMPLES = [
    "heavy wooden door creaking open slowly on rusted hinges, echoing stone hallway",
    "single glass bottle shattering on a concrete floor, close mic, sharp transient",
    "distant thunder rolling across a valley, low rumble, light rain on leaves",
    "sword being drawn from a leather scabbard, metallic ring, dry studio",
    "old film projector running, mechanical clatter, steady rhythm, small room",
    "footsteps on gravel, slow deliberate pace, outdoors at night, crickets",
    "large spaceship door hissing open, pneumatic release, deep metallic resonance",
    "campfire crackling, occasional pop, wind in nearby trees",
]


def list_audio(d):
    if not os.path.isdir(d):
        return []
    return sorted((f for f in os.listdir(d) if f.lower().endswith(AUDIO_EXT)), reverse=True)


def stash_upload(path, inputs_dir):
    """Gradio hands us a temp file that dies with the browser session. Copy it into the
    mounted inputs/ so the same clip stays addressable by name from the REST API."""
    os.makedirs(inputs_dir, exist_ok=True)
    name = f"upload_{int(time.time())}_{os.path.basename(path)}"
    shutil.copy(path, os.path.join(inputs_dir, name))
    return name


def mount_ui(app, *, state, generate, request_cls, inputs_dir, outputs_dir,
             model_id, post_trained, max_duration, path="/ui"):

    def resolve(dropdown, upload):
        """Upload wins over the dropdown. Returns a name relative to inputs_dir, which is
        what SfxRequest wants."""
        if upload:
            return stash_upload(upload, inputs_dir)
        return dropdown or None

    def health_md():
        if state.get("model") is None:
            return ("### ⏳ loading\nThe model is still coming up — the first start also "
                    "downloads the checkpoint and the T5Gemma text encoder.")
        try:
            import torch
            alloc = torch.cuda.memory_allocated() / 1e9
            resv = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return (f"### ✅ resident\n`{alloc:.2f} GB` allocated · `{resv:.2f} GB` reserved · "
                    f"`{total:.0f} GB` total · `{state.get('sr')} Hz`\n\n"
                    f"Small-SFX is ~2 GB, so it can share the card with a resident LLM.")
        except Exception as e:
            return f"### ✅ resident\n(VRAM unavailable: {type(e).__name__})"

    def refresh_inputs():
        return gr.update(choices=list_audio(inputs_dir))

    def refresh_outputs():
        return gr.update(choices=list_audio(outputs_dir))

    def load_output(name):
        return os.path.join(outputs_dir, name) if name else None

    def on_mode(mode):
        return (gr.update(visible=mode == "Variation"),
                gr.update(visible=mode == "Inpaint / Extend"))

    async def run(mode, prompt, duration, steps, seed, batch, fmt, prefix,
                  src_dd, src_up, noise, ip_dd, ip_up, ip_start, ip_end):
        blank = (None, gr.update(), gr.update())
        if not (prompt or "").strip():
            yield "⚠ Nothing to generate — the prompt is empty.", *blank
            return
        if state.get("model") is None:
            yield "⚠ Model is still loading. Watch `docker logs -f stable-audio-3-sfx`.", *blank
            return

        kw = dict(prompt=prompt.strip(), duration=float(duration), steps=int(steps),
                  seed=int(seed), batch_size=int(batch), format=fmt, prefix=prefix or "ui")

        if mode == "Variation":
            src = resolve(src_dd, src_up)
            if not src:
                yield "⚠ Variation mode needs a source clip.", *blank
                return
            kw.update(init_audio=src, init_noise_level=float(noise))
        elif mode == "Inpaint / Extend":
            src = resolve(ip_dd, ip_up)
            if not src:
                yield "⚠ Inpaint mode needs a source clip.", *blank
                return
            if float(ip_end) <= float(ip_start):
                yield "⚠ Mask end must be after mask start.", *blank
                return
            kw.update(inpaint_audio=src,
                      inpaint_mask_start_seconds=float(ip_start),
                      inpaint_mask_end_seconds=float(ip_end))

        try:
            req = request_cls(**kw)
        except Exception as e:
            yield f"❌ **{type(e).__name__}**: {e}", *blank
            return

        async def locked():
            # Same lock the REST endpoints take, so a UI click and a /v1/audio job queue
            # behind each other instead of interleaving through the same weights.
            async with state["gen_lock"]:
                return await asyncio.to_thread(generate, req)

        t0 = time.time()
        task = asyncio.create_task(locked())
        yield "⏳ queued …", *blank

        # 8 steps on a 433M model is fast enough that a progress bar would be theatre.
        # An honest wall clock is more useful, and it also shows queueing behind the API.
        while not task.done():
            await asyncio.sleep(0.5)
            yield f"⏳ sampling — **{time.time()-t0:.1f} s** elapsed", *blank

        try:
            info = task.result()
        except Exception as e:
            traceback.print_exc()
            oom = "\n\nThis looks like an OOM — stop other GPU containers and retry." \
                if "out of memory" in str(e).lower() else ""
            yield f"❌ **{type(e).__name__}**: {e}{oom}", *blank
            return

        note = f"\n\n⚠ {info['ignored']}" if "ignored" in info else ""
        extra = f" · batch of {len(info['files'])}" if len(info["files"]) > 1 else ""
        yield (f"✅ **{info['file']}** — {info['duration_s']} s rendered in "
               f"{info['generation_s']} s ({info['realtime_factor']}× realtime), peak "
               f"{info['peak_vram_gb']} GB VRAM{extra}{note}"), \
              info["path"], json.dumps(info, indent=2), \
              gr.update(choices=list_audio(outputs_dir), value=info["file"])

    # -- layout --------------------------------------------------------------------------
    with gr.Blocks(title=f"Stable Audio 3 — {model_id}", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# Stable Audio 3 · {model_id}\n"
            "433M latent-diffusion sound-effects model — 44.1 kHz stereo, up to "
            f"{max_duration:.0f} s, 8 sampling steps. Source clips live in the mounted "
            "`inputs/`, results in `outputs/`.\n\n"
            "*Licensed under the Stability AI Community License — free for research and "
            "for commercial use below the revenue threshold; read it before shipping "
            "outputs (<https://stability.ai/license>).*"
        )

        with gr.Row():
            # ---------------- left: what to make ----------------
            with gr.Column(scale=3):
                mode = gr.Radio(["Text → SFX", "Variation", "Inpaint / Extend"],
                                value="Text → SFX", label="Mode")
                prompt = gr.Textbox(
                    label="Prompt", lines=3,
                    placeholder="Describe the physical event and the space it happens in — "
                                "material, action, distance, room.")
                gr.Examples(EXAMPLES, inputs=prompt, label="Example prompts")

                with gr.Group(visible=False) as var_box:
                    gr.Markdown("**Variation** — noise the source, then denoise toward the "
                                "prompt. `1.0` ignores the source entirely; `0.1` is a near "
                                "copy; `0.5` is a halfway blend.")
                    with gr.Row():
                        src_dd = gr.Dropdown(list_audio(inputs_dir), label="Source from inputs/",
                                             scale=3)
                        src_rescan = gr.Button("↻", scale=1)
                    src_up = gr.Audio(label="…or upload", type="filepath")
                    noise = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="init_noise_level")

                with gr.Group(visible=False) as ip_box:
                    gr.Markdown("**Inpaint / Extend** — regenerate the masked region and keep "
                                "the rest. To *extend* a clip, set mask start to the clip's "
                                "length and raise Duration past it.")
                    with gr.Row():
                        ip_dd = gr.Dropdown(list_audio(inputs_dir), label="Source from inputs/",
                                            scale=3)
                        ip_rescan = gr.Button("↻", scale=1)
                    ip_up = gr.Audio(label="…or upload", type="filepath")
                    with gr.Row():
                        ip_start = gr.Number(value=0.0, label="Mask start (s)")
                        ip_end = gr.Number(value=2.0, label="Mask end (s)")

            # ---------------- right: how to make it ----------------
            with gr.Column(scale=2):
                duration = gr.Slider(0.5, max_duration, value=7.0, step=0.5, label="Duration (s)")
                steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                gr.Markdown(
                    "*8 steps is the tuned default for this post-trained checkpoint — going "
                    "higher does not buy quality here (that advice is for `-base` weights, "
                    "which want ~50). Lower trades quality for speed.*"
                    + ("" if post_trained else
                       "\n\n*This is a **base** checkpoint: CFG and negative prompts do apply.*"))
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    batch = gr.Slider(1, 8, value=1, step=1, label="Batch")
                with gr.Row():
                    fmt = gr.Dropdown(["wav", "flac", "ogg", "mp3"], value="wav", label="Format")
                    prefix = gr.Textbox(value="ui", label="Filename prefix")
                health = gr.Markdown(health_md())
                refresh_health = gr.Button("↻ refresh GPU status", size="sm")

        go = gr.Button("Generate", variant="primary", size="lg")
        status = gr.Markdown()
        audio_out = gr.Audio(label="Result", type="filepath", interactive=False)

        with gr.Accordion("Past outputs", open=False):
            with gr.Row():
                past = gr.Dropdown(list_audio(outputs_dir), label="outputs/", scale=3)
                refresh_past = gr.Button("↻", scale=1)
        with gr.Accordion("Response detail", open=False):
            info_json = gr.Code(language="json", label="")

        # -- wiring ----------------------------------------------------------------------
        mode.change(on_mode, mode, [var_box, ip_box])
        src_rescan.click(refresh_inputs, outputs=src_dd)
        ip_rescan.click(refresh_inputs, outputs=ip_dd)
        refresh_health.click(health_md, outputs=health)
        refresh_past.click(refresh_outputs, outputs=past)
        past.change(load_output, past, audio_out)

        go.click(run,
                 [mode, prompt, duration, steps, seed, batch, fmt, prefix,
                  src_dd, src_up, noise, ip_dd, ip_up, ip_start, ip_end],
                 [status, audio_out, info_json, past])

        demo.load(health_md, outputs=health)

    # Generation already serializes on state["gen_lock"]; the queue is here so a second
    # browser tab waits in line instead of erroring out.
    demo.queue(default_concurrency_limit=4, max_size=32)
    gr.mount_gradio_app(app, demo, path=path, allowed_paths=[outputs_dir, inputs_dir])
    print(f"UI mounted at {path}", flush=True)
    return demo
