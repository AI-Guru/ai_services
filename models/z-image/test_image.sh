#!/usr/bin/env bash
# Smoke-test the Z-Image-Turbo vLLM-Omni endpoint.
# Usage: ./test_image.sh [prompt] [outfile]
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:11476}"
PROMPT="${1:-a photorealistic red panda barista pulling an espresso shot, warm morning light}"
OUT="${2:-z-image-out.png}"

# Z-Image-Turbo is distilled: 8 steps, CFG ~1.0. The response is OpenAI-shaped
# (data[0].b64_json), so decode it to a PNG.
curl -sS -X POST "${BASE_URL}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"z-image-turbo\",
    \"prompt\": $(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
    \"size\": \"1024x1024\",
    \"num_inference_steps\": 8,
    \"guidance_scale\": 1.0,
    \"negative_prompt\": \"blurry, low quality, distorted, extra fingers\",
    \"seed\": 42,
    \"n\": 1
  }" \
  | python3 -c "import json,sys,base64; d=json.load(sys.stdin); open('${OUT}','wb').write(base64.b64decode(d['data'][0]['b64_json'])); print('wrote ${OUT}')"
