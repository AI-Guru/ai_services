#!/usr/bin/env python3
"""Compare upscaling approaches on identical inputs, against known ground truth.

The honest way to score an upscaler is to hand it a problem whose answer you
already have: take a high-quality source, degrade it on purpose, restore it, and
score the restoration against the original. That is what this does.

    source (HQ)  --downscale /N + JPEG--> degraded  --restore xN-->  candidate
                                                                        |
                                       PSNR / SSIM  <-- compared vs --- source

Methods compared:
  lanczos      classical resampling, no model — the floor everything must beat
  realesrgan   GAN, ../docker-compose.upscaler-realesrgan-rtx.yml   (port 11477)
  supir        SDXL diffusion, supir/docker-compose.supir-*.yml     (port 11478)
  seedvr2      SeedVR2-3B one-step DiT, ../models/seedvr2/          (port 11480)

READ THE NUMBERS CAREFULLY. PSNR and SSIM are *distortion* metrics: they reward
staying close to the original pixels. Blau & Michaeli (CVPR 2018) proved
distortion and perceptual quality are mathematically at odds, so a generative
restorer that produces the better-looking image is EXPECTED to score worse here.
A low SUPIR PSNR is the trade-off working as designed, not a failure — which is
exactly why this script also emits a side-by-side HTML page to look at.

Usage:
    python3 compare_upscalers.py --images church1.png scripts/testimage.jpg
    python3 compare_upscalers.py --scale 4 --jpeg-quality 40 --edm-steps 50
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REALESRGAN_URL = "http://localhost:11477"
SUPIR_URL = "http://localhost:11478"
SEEDVR2_URL = "http://localhost:11480"


# --------------------------------------------------------------------------
# metrics (numpy-only: this box has no scipy/skimage outside the containers)
# --------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax**2) / (2 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def _filter2(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Valid-mode 2D correlation via stride tricks — avoids a scipy dependency."""
    kh, kw = k.shape
    view = np.lib.stride_tricks.sliding_window_view(img, (kh, kw))
    return np.einsum("ijkl,kl->ij", view, k)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Wang et al. SSIM on luma, 11x11 gaussian window, mean over map."""
    def luma(x: np.ndarray) -> np.ndarray:
        return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)

    x, y = luma(a), luma(b)
    k = _gaussian_kernel()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu_x, mu_y = _filter2(x, k), _filter2(y, k)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x2 = _filter2(x * x, k) - mu_x2
    sigma_y2 = _filter2(y * y, k) - mu_y2
    sigma_xy = _filter2(x * y, k) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(num / den))


# --------------------------------------------------------------------------
# degradation + backends
# --------------------------------------------------------------------------
def degrade(src: Image.Image, scale: int, jpeg_quality: int | None) -> Image.Image:
    """Downscale by `scale`, optionally round-tripping through JPEG for artifacts.

    `jpeg_quality=None` gives a PURE resolution round-trip — the only thing lost is
    detail below the new Nyquist limit, with no compression damage on top. That
    isolates "can the upscaler invent back what downsampling removed?" from "can it
    also clean up compression?", which are different questions.
    """
    small = src.resize((src.width // scale, src.height // scale), Image.LANCZOS)
    if jpeg_quality is None:
        return small.convert("RGB")
    buf = io.BytesIO()
    small.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def run_lanczos(img: Image.Image, scale: int, **_: object) -> tuple[Image.Image, float]:
    t0 = time.time()
    out = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    return out, time.time() - t0


def run_realesrgan(img: Image.Image, scale: int, **_: object) -> tuple[Image.Image, float]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    t0 = time.time()
    r = requests.post(
        f"{REALESRGAN_URL}/upscale",
        files={"file": ("in.png", buf.getvalue(), "image/png")},
        data={"scale": scale} if scale in (2, 4) else {"outscale": scale},
        timeout=300,
    )
    dt = time.time() - t0
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)), dt


def run_supir(
    img: Image.Image, scale: int, prompt: str = "", edm_steps: int = 50, s_cfg: float = 4.0, **_: object
) -> tuple[Image.Image, float]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    t0 = time.time()
    r = requests.post(
        f"{SUPIR_URL}/upscale/json",
        json={
            "image": base64.b64encode(buf.getvalue()).decode(),
            "upscale": scale,
            "prompt": prompt,
            "edm_steps": edm_steps,
            "s_cfg": s_cfg,
        },
        timeout=1800,
    )
    dt = time.time() - t0
    r.raise_for_status()
    body = r.json()
    return Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"]))), dt


def run_seedvr2(img: Image.Image, scale: int, seed: int = 666, **_: object) -> tuple[Image.Image, float]:
    """SeedVR2-3B (models/seedvr2). One-step; takes no prompt (frozen conditioning)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    t0 = time.time()
    r = requests.post(
        f"{SEEDVR2_URL}/upscale/json",
        json={"image": base64.b64encode(buf.getvalue()).decode(), "scale": scale, "seed": seed},
        timeout=1800,
    )
    dt = time.time() - t0
    r.raise_for_status()
    body = r.json()
    return Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"]))), dt


METHODS = {
    "lanczos": run_lanczos,
    "realesrgan": run_realesrgan,
    "supir": run_supir,
    "seedvr2": run_seedvr2,
}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _b64_png(img: Image.Image, max_side: int = 512) -> str:
    im = img.copy()
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _crop(img: Image.Image, frac: float = 0.28, size: int = 256) -> Image.Image:
    """Centre-ish crop at native resolution — where the differences actually live."""
    cx, cy = int(img.width * 0.5), int(img.height * frac)
    half = min(size, img.width, img.height) // 2
    box = (max(0, cx - half), max(0, cy - half), min(img.width, cx + half), min(img.height, cy + half))
    return img.crop(box)


def write_html(results: list[dict], out_path: Path, cfg: dict) -> None:
    order = ["source", "degraded", *METHODS]
    rows = []
    for r in results:
        cells = "".join(
            f"<figure><img src='data:image/png;base64,{r['thumbs'][m]}' alt='{m}'>"
            f"<figcaption><b>{m}</b>{r['captions'][m]}</figcaption></figure>"
            for m in order if m in r["thumbs"]
        )
        crops = "".join(
            f"<figure><img src='data:image/png;base64,{r['crops'][m]}' alt='{m} crop'>"
            f"<figcaption>{m}</figcaption></figure>"
            for m in order if m in r["crops"]
        )
        rows.append(
            f"<section><h2>{r['name']}</h2>"
            f"<div class='strip'>{cells}</div>"
            f"<h3>1:1 crop</h3><div class='strip'>{crops}</div></section>"
        )

    table_rows = "".join(
        f"<tr><td>{r['name']}</td><td>{m}</td>"
        f"<td class='n'>{r['metrics'][m]['psnr']:.2f}</td>"
        f"<td class='n'>{r['metrics'][m]['ssim']:.4f}</td>"
        f"<td class='n'>{r['metrics'][m]['seconds']:.2f}</td></tr>"
        for r in results for m in METHODS if m in r["metrics"]
    )

    out_path.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Upscaler comparison</title><style>
 body{{font:15px/1.55 system-ui,sans-serif;margin:0;padding:2rem;background:#0f1115;color:#e8eaed}}
 h1{{margin:0 0 .3rem}} h2{{margin:2.5rem 0 .6rem;font-size:1.15rem}}
 h3{{font-size:.85rem;text-transform:uppercase;letter-spacing:.07em;color:#9aa0a6;margin:1.2rem 0 .5rem}}
 .strip{{display:flex;gap:1rem;flex-wrap:wrap}}
 figure{{margin:0}} figure img{{display:block;border-radius:6px;background:#1b1e24;max-width:100%}}
 figcaption{{font-size:.78rem;color:#9aa0a6;margin-top:.35rem;line-height:1.4}}
 table{{border-collapse:collapse;margin:1rem 0 2rem;font-size:.88rem;width:100%;max-width:760px}}
 th,td{{text-align:left;padding:.45rem .8rem;border-bottom:1px solid #272b33}}
 th{{color:#9aa0a6;font-weight:600}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
 .note{{background:#171a20;border-left:3px solid #f0a04b;padding:.9rem 1.1rem;border-radius:0 6px 6px 0;
        max-width:760px;font-size:.9rem;color:#c8ccd2}}
 .wrap{{overflow-x:auto}}
</style></head><body>
<h1>Upscaler comparison — &times;{cfg['scale']}</h1>
<p style="color:#9aa0a6;margin:.2rem 0 1.4rem">
  Degradation: &divide;{cfg['scale']} Lanczos" + (" + JPEG q%s" % cfg['jpeg_quality'] if cfg['jpeg_quality'] else " (no JPEG)") + " &middot;
  SUPIR: {cfg['edm_steps']} steps, s_cfg {cfg['s_cfg']} &middot; scored against the undegraded source
</p>
<div class="note"><b>PSNR and SSIM measure distortion, not beauty.</b> They reward pixel-fidelity to
the original, so a generative restorer that invents convincing detail is <i>expected</i> to score
lower while looking better. Blau &amp; Michaeli (CVPR 2018) proved the two objectives are formally
at odds. Use the numbers for fidelity and your eyes for quality — that is why both are here.</div>
<div class="wrap"><table><thead><tr><th>image</th><th>method</th><th>PSNR&nbsp;dB</th><th>SSIM</th>
<th>sec</th></tr></thead><tbody>{table_rows}</tbody></table></div>
{"".join(rows)}
</body></html>""")


def write_montage(stem: str, imgs: dict[str, Image.Image], out_path: Path, size: int = 256) -> None:
    """One strip of 1:1 crops per source, labelled — the committed evidence.

    The full-size outputs and the base64-embedded HTML are large and gitignored;
    these strips are small enough to live in the repo next to the write-up.
    Crops are NEAREST-resampled so the comparison isn't itself smoothed.
    """
    from PIL import ImageDraw

    order = [n for n in ("source", "degraded", *METHODS) if n in imgs]
    tiles = []
    for n in order:
        im = imgs[n]
        if n == "degraded":  # bring the LQ input up to the same framing, unsmoothed
            im = im.resize((im.width * 4, im.height * 4), Image.NEAREST)
        cx, cy, half = im.width // 2, int(im.height * 0.35), size // 2
        box = (max(0, cx - half), max(0, cy - half), min(im.width, cx + half), min(im.height, cy + half))
        tiles.append((n, im.crop(box).resize((size, size), Image.NEAREST)))

    canvas = Image.new("RGB", (size * len(tiles), size + 22), (20, 22, 26))
    draw = ImageDraw.Draw(canvas)
    for i, (n, tile) in enumerate(tiles):
        canvas.paste(tile, (i * size, 22))
        draw.text((i * size + 6, 5), n, fill=(230, 230, 230))
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="high-quality source images")
    ap.add_argument("--scale", type=int, default=4, help="degrade by /N, restore by xN")
    ap.add_argument("--jpeg-quality", type=int, default=40)
    ap.add_argument("--no-jpeg", action="store_true",
                    help="pure resolution round-trip: skip the JPEG degradation entirely")
    ap.add_argument("--edm-steps", type=int, default=50, help="SUPIR diffusion steps")
    ap.add_argument("--s-cfg", type=float, default=4.0, help="SUPIR guidance scale")
    ap.add_argument("--prompt", default="", help="SUPIR restoration prompt")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--outdir", default="comparison_out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "images").mkdir(parents=True, exist_ok=True)
    results = []

    for path in args.images:
        src = Image.open(path).convert("RGB")
        # crop to a multiple of scale so the degrade/restore round-trip is exact
        w, h = (src.width // args.scale) * args.scale, (src.height // args.scale) * args.scale
        src = src.crop((0, 0, w, h))
        lq = degrade(src, args.scale, None if args.no_jpeg else args.jpeg_quality)
        name = Path(path).name
        print(f"\n=== {name}  {src.size} -> degraded {lq.size} ===")

        src_np = np.asarray(src)
        entry = {
            "name": name,
            "source_size": list(src.size),
            "degraded_size": list(lq.size),
            "metrics": {},
            "thumbs": {"source": _b64_png(src), "degraded": _b64_png(lq)},
            "crops": {"source": _b64_png(_crop(src), 256), "degraded": _b64_png(_crop(lq.resize(src.size, Image.NEAREST)), 256)},
            "captions": {"source": "<br>ground truth", "degraded": f"<br>{lq.width}&times;{lq.height}" + ('' if args.no_jpeg else f' q{args.jpeg_quality}')},
        }
        src.save(outdir / "images" / f"{Path(path).stem}_source.png")
        lq.save(outdir / "images" / f"{Path(path).stem}_degraded.png")
        full = {"source": src, "degraded": lq}

        for m in args.methods:
            try:
                out, dt = METHODS[m](
                    lq, args.scale, prompt=args.prompt, edm_steps=args.edm_steps, s_cfg=args.s_cfg
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {m:11s} FAILED: {exc}")
                continue
            if out.size != src.size:
                out = out.resize(src.size, Image.LANCZOS)
            out_np = np.asarray(out.convert("RGB"))
            p, s = psnr(src_np, out_np), ssim(src_np, out_np)
            entry["metrics"][m] = {"psnr": p, "ssim": s, "seconds": dt}
            entry["thumbs"][m] = _b64_png(out)
            entry["crops"][m] = _b64_png(_crop(out), 256)
            entry["captions"][m] = f"<br>PSNR {p:.2f} dB · SSIM {s:.4f}<br>{dt:.2f}s"
            out.save(outdir / "images" / f"{Path(path).stem}_{m}.png")
            full[m] = out
            print(f"  {m:11s} PSNR {p:6.2f} dB   SSIM {s:.4f}   {dt:7.2f}s")

        write_montage(Path(path).stem, full, outdir / f"montage_{Path(path).stem}.png")
        results.append(entry)

    cfg = {
        "scale": args.scale,
        "jpeg_quality": None if args.no_jpeg else args.jpeg_quality,
        "edm_steps": args.edm_steps,
        "s_cfg": args.s_cfg,
        "prompt": args.prompt,
    }
    (outdir / "results.json").write_text(
        json.dumps(
            {"config": cfg, "results": [{k: v for k, v in r.items() if k not in ("thumbs", "crops", "captions")} for r in results]},
            indent=2,
        )
    )
    write_html(results, outdir / "comparison.html", cfg)
    print(f"\nwrote {outdir}/comparison.html  and  {outdir}/results.json")


if __name__ == "__main__":
    main()
