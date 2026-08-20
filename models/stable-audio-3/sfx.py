#!/usr/bin/env python3
"""
CLI for the resident Stable Audio 3 Small-SFX server.

Thin client over the REST API — the request body is the API, so a server version bump
surfaces as a 422 with the offending field named rather than a silent misparse.

  ./sfx.py "heavy wooden door creaking open, echoing stone hallway" --duration 6
  ./sfx.py "rain on a tin roof" --batch 4 --seed 1234 --format flac
  ./sfx.py "metallic scrape" --init-audio upload_x.wav --noise 0.6
  ./sfx.py "distant thunder tail" --inpaint-audio storm.wav --mask 8 14 --duration 20
  ./sfx.py --health

Writes land in the server's outputs/ regardless; -o additionally saves a local copy.
"""
import argparse, json, os, sys, urllib.error, urllib.request

DEFAULT_HOST = os.environ.get("SA3_HOST", "http://localhost:8400")


def call(host, path, payload=None, method=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{host}{path}", data=data, method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"} if data else {})
    return urllib.request.urlopen(req, timeout=timeout)


def main():
    p = argparse.ArgumentParser(description="Generate SFX on the resident Stable Audio 3 server.")
    p.add_argument("prompt", nargs="?", help="what to generate")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--health", action="store_true", help="print server status and exit")

    p.add_argument("--duration", type=float, default=7.0, help="seconds (max 120)")
    p.add_argument("--steps", type=int, default=8,
                   help="8 is tuned for the post-trained weights; only '-base' wants ~50")
    p.add_argument("--seed", type=int, default=-1, help="-1 = random")
    p.add_argument("--batch", type=int, default=1, help="clips per request; cheaper than N requests")
    p.add_argument("--format", default="wav", choices=["wav", "flac", "ogg", "mp3"])
    p.add_argument("--prefix", default="sfx", help="output filename prefix")
    p.add_argument("-o", "--out", help="also save the first clip here")

    p.add_argument("--init-audio", help="filename under the server's inputs/ — restyle it")
    p.add_argument("--noise", type=float, default=0.7,
                   help="init_noise_level: 1.0 ignores the source, 0.1 is a near copy")

    p.add_argument("--inpaint-audio", help="filename under the server's inputs/")
    p.add_argument("--mask", nargs=2, type=float, metavar=("START", "END"),
                   help="region to regenerate, in seconds; for continuation set START to "
                        "the clip length and raise --duration past it")

    # Base-checkpoint only. Sent through so '-base' deployments work; the server reports
    # them back as ignored on post-trained weights rather than silently dropping them.
    p.add_argument("--cfg", type=float, default=1.0, help="base checkpoints only")
    p.add_argument("--negative", help="base checkpoints only")

    a = p.parse_args()

    try:
        if a.health:
            print(json.dumps(json.load(call(a.host, "/health")), indent=2))
            return
        if not a.prompt:
            p.error("a prompt is required unless --health is given")

        body = {"prompt": a.prompt, "duration": a.duration, "steps": a.steps,
                "seed": a.seed, "batch_size": a.batch, "format": a.format,
                "prefix": a.prefix, "cfg_scale": a.cfg}
        if a.negative:
            body["negative_prompt"] = a.negative
        if a.init_audio:
            body.update(init_audio=a.init_audio, init_noise_level=a.noise)
        if a.inpaint_audio:
            if not a.mask:
                p.error("--inpaint-audio requires --mask START END")
            body.update(inpaint_audio=a.inpaint_audio,
                        inpaint_mask_start_seconds=a.mask[0],
                        inpaint_mask_end_seconds=a.mask[1])

        resp = call(a.host, "/v1/audio/sync", body)
        blob = resp.read()
        info = json.loads(resp.headers.get("X-SA3-Info", "{}"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        sys.exit(f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {a.host}: {e.reason}\n"
                 f"Is the container up?  docker ps --filter name=stable-audio-3-sfx")

    if a.out:
        with open(a.out, "wb") as f:
            f.write(blob)

    print(f"{info.get('file')} — {info.get('duration_s')}s in {info.get('generation_s')}s "
          f"({info.get('realtime_factor')}x realtime), peak {info.get('peak_vram_gb')} GB")
    for extra in info.get("files", [])[1:]:
        print(f"  + {extra['file']}")
    if "ignored" in info:
        print(f"note: {info['ignored']}", file=sys.stderr)
    if a.out:
        print(f"saved local copy: {a.out}")


if __name__ == "__main__":
    main()
