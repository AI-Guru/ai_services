# LibreChat

Local LibreChat deployment configured to use a llama.cpp server as backend.

## Prerequisites

- Docker and Docker Compose
- A running llama.cpp server on port `11437` (OpenAI-compatible API)

## Setup

```bash
cp .env.example .env
# Fill in CREDS_KEY, CREDS_IV, JWT_SECRET, JWT_REFRESH_SECRET
# Generate values at https://www.librechat.ai/toolkit/creds_generator
docker compose up -d
```

Access at http://localhost:3080 or http://<hostname>.local:3080 from LAN.

The first user to register becomes admin.

## Configuration

- `librechat.yaml` — Custom endpoint config pointing to `host.docker.internal:11437`
- `.env.example` — Template for `.env` (copy and fill in secrets)
- `docker-compose.yml` — Services: LibreChat, MongoDB, MeiliSearch

## Notes

- The backend runs in dev mode (`npm run backend:dev`) to disable the `Secure` cookie flag, which is required for sessions to persist over plain HTTP on LAN.
- Built-in providers (OpenAI, Anthropic, Google) are disabled. Add API keys in `.env` to enable them.
- Model discovery is enabled (`fetch: true`), so any models served by llama.cpp will appear automatically.
