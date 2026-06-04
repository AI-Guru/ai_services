# Odysseus — evaluation harness, wired to local model serving

[Odysseus](https://github.com/pewdiepie-archdaemon/odysseus) is PewDiePie's
self-hosted AI *workspace* — a single web UI that bundles chat, autonomous
agents, Deep Research, a model "Cookbook", email/calendar/notes, and image gen.
This directory runs it as a **throwaway evaluation** against the models this repo
already serves, instead of letting its Cookbook download and serve its own.

> ## ⚠️ Maturity: treat this as a 4-day-old toy, not infrastructure
>
> The 48k GitHub stars are a **celebrity-launch signal, not an engineering one.**
> Read this section before you point it at anything you care about.
>
> | Fact | Value (checked 2026-06-04) |
> |---|---|
> | Repo created | **2026-05-31** — i.e. **~4 days old** |
> | Stars | **48,176** (≈ **12,000/day**, ~8 every minute since launch) |
> | Forks | 5,551 |
> | Commits on `main` | ~758 |
> | Merged PRs / **open** PRs | 648 / **514** (a review backlog bigger than all history) |
> | Open issues | 339 |
> | License | MIT |
>
> - **"Vibecoded."** Widely described (and semi-admitted on the project site) as
>   AI-written-by-gut — *"800 merged PRs of LLM-generated code with barely any
>   objective improvement."* Reported smells: ad-hoc SQLite DBs + loose JSON
>   files rather than a coherent data layer. Changes land faster than anyone can
>   review (648 merged in ~95 hours).
> - **Real security exposure.** The agent has **filesystem + shell + web-scrape**
>   tools, so a prompt-injection payload can plausibly reach **arbitrary code
>   execution on the host** — the same failure pattern that dogged OpenClaw.
>   Gizmodo flagged this directly.
> - **Reinvents mature prior art.** Common community objection: *why not Open
>   WebUI / LibreChat?*, which have done this — with auth, RAG, tests, and years
>   of hardening — far longer.
>
> **Mitigations baked into this setup:** every port bound to **loopback only**,
> `AUTH_ENABLED=true`, models served **outside** the container, GPU **not**
> exposed to it, and no real secrets in `.env`. Do not put this on the LAN, do
> not give the agent a `FILESYSTEM_ROOT` you'd miss, and tear it down when done.

## What this directory adds

The upstream image has no published tag (`build: .`), so the compose builds
straight from the upstream repo's **`main` branch** — nothing is vendored into
`ai_services`. Alongside `odysseus` it brings up the sidecars the app hard-
depends on: `chromadb` (vectors), `searxng` (Deep Research web search), `ntfy`
(notifications).

```
odysseus/
├── docker-compose.yaml          # odysseus (git-build) + chromadb + searxng + ntfy, loopback-only
├── .env.example                 # documented config — copy to .env
├── .env                         # working config for this box (gitignored)
├── config/searxng/settings.yml  # vendored searxng template (so it boots without cloning)
└── data/, logs/                 # created on first run (app db, chroma, caches)
```

## Connecting to this box's models

Odysseus discovers OpenAI-compatible servers by scanning, on each configured
host, ports **8000–8020 + 1234 + 11434**, **plus any port embedded in a provider
env URL**. This repo serves the active model on **`11436`** (the port LibreChat
already uses — currently `qwen3.6-27b` via vLLM). Since 11436 isn't in the
default scan, `.env` injects it through `OLLAMA_BASE_URL`:

```env
LLM_HOST=host.docker.internal
OLLAMA_BASE_URL=http://host.docker.internal:11436/v1   # adds :11436 to the scan
RESEARCH_LLM_ENDPOINT=http://host.docker.internal:11436/v1/chat/completions
```

Chat flows over `/v1/chat/completions` regardless of the "ollama" label — it's
just how the port gets into the scan list. **Swap models?** Change `11436` in
`.env` to the new port (and confirm the served-model name in the Odysseus UI).
Embeddings fall back to local fastembed/ONNX unless you set `EMBEDDING_URL` at a
served embedding model (e.g. stand up [`../models/qwen3-embedding`](../models/qwen3-embedding)).

> **GPU note (single shared card).** This stack does **not** request the GPU —
> it consumes the model you're already serving. Per the repo convention, only
> one model is resident at a time, so make sure something is up on `11436`
> (`docker ps`) before expecting models to appear in Odysseus.

## Run

```bash
cd odysseus
cp .env.example .env          # then edit ODYSSEUS_ADMIN_PASSWORD at minimum
docker compose up -d --build  # first build clones + builds upstream — slow

# open http://127.0.0.1:7000  and create the admin user
docker compose logs -f odysseus
docker compose down           # add -v to also wipe chroma/searxng/ntfy volumes
```

Because every compose service uses `restart: unless-stopped`, a failed Odysseus
build/boot will **crash-loop silently** (see the repo `CLAUDE.md`). If the UI
never comes up, check the real cause:

```bash
docker compose logs odysseus | grep -iE 'error|traceback|exception|out of memory' | tail
```

## Updating

The build tracks upstream **`main`** (unpinned), so a fresh build pulls the
latest commit. To refresh:

```bash
docker compose build --no-cache odysseus && docker compose up -d
```

Heads-up: given the ~12k-stars/day churn, `main` moves fast and rewrites history,
so builds are **not reproducible** — what you get depends on when you build. If
you ever need a known-good build, pin a commit SHA in the `build.context` URL in
[docker-compose.yaml](docker-compose.yaml) (`…/odysseus.git#<sha>`).

## Mature alternatives (what to use instead for anything real)

All three below are MIT-licensed, years-mature, actively maintained, and connect
to the exact same OpenAI-compatible endpoints this repo serves. For real use,
prefer these over Odysseus:

| Project | Best at | Why over Odysseus |
|---|---|---|
| **[Open WebUI](https://github.com/open-webui/open-webui)** (~136k★) | Teams / multi-user | Most mature multi-user system: RBAC, OIDC SSO, per-user isolation, usage quotas, global model/preset/document management. The default "serious" self-hosted UI. |
| **[LibreChat](https://www.librechat.ai/)** (already running here on `:3080`) | Multi-provider power users | Multi-provider, token tracking, agents, **MCP**, code interpreter — one interface for local + cloud. Already wired to `host.docker.internal:11436` in [../librechat/](../librechat/). |
| **[AnythingLLM](https://anythingllm.com/)** | "Chat over my documents" | Workspace-centric RAG with a zero-config desktop app — the simplest path for document Q&A. |

**Bottom line:** Odysseus is worth a look as a *reference and a cultural moment*
— and this harness makes that look safe and quick. It is **not** a foundation to
build on. If you want a durable front door to the models in this repo, you
already have one: **LibreChat** on `:3080`.

---

### Sources
- [Odysseus repo](https://github.com/pewdiepie-archdaemon/odysseus) (live API stats) · [SECURITY.md](https://github.com/pewdiepie-archdaemon/odysseus/blob/main/SECURITY.md)
- [Pasquale Pillitteri — Odysseus analysis](https://pasqualepillitteri.it/en/news/4016/odysseus-pewdiepie-local-ai-workspace) · [Gizmodo — privacy & security caveats](https://gizmodo.com/pewdiepie-is-here-to-offer-you-privacy-assurances-in-the-age-of-ai-2000765812)
- [ToolHalla — Open WebUI vs AnythingLLM vs LibreChat (2026)](https://toolhalla.ai/blog/open-webui-vs-anythingllm-vs-librechat-2026) · [Open WebUI](https://github.com/open-webui/open-webui)
