# Cleanup — what this project installs locally

Everything the backend creates on your machine, and how to remove it. Most of
the disk footprint (model weights, DB volumes) lives **outside** this repo.

Nothing here is deleted automatically — run the commands you want when cleaning up.

---

## 1. Docker images, containers, and volumes (if you use `docker compose`)

The Compose project name is pinned to `docintel`, so everything is namespaced.

**Containers + network (keeps volumes/images):**
```bash
docker compose down
```

**Containers + named volumes (deletes DB data, vectors, uploads, model cache):**
```bash
docker compose down -v
```

**Named volumes** (created on first `up`; `-v` above removes them, or by hand):
- `docintel_pgdata`     — Postgres data
- `docintel_qdrantdata` — Qdrant vectors
- `docintel_uploads`    — uploaded documents
- `docintel_models`     — HuggingFace weight cache (`HF_HOME=/models`, several GB)
```bash
docker volume rm docintel_pgdata docintel_qdrantdata docintel_uploads docintel_models
```

**Built app image** (from this repo's Dockerfile):
```bash
docker image rm docintel-api docintel-worker    # names may vary; see: docker image ls | grep docintel
```

**Pulled base images** (only if nothing else uses them):
```bash
docker image rm postgres:16 redis:7-alpine qdrant/qdrant:latest
```

---

## 2. Running the app locally (no Docker)

**Python packages** — always install into a virtualenv so cleanup is one `rm`:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Remove everything (packages + torch, several GB) by deleting the venv:
```bash
rm -rf .venv
```
> If you installed into the system/global Python instead, cleanup is messy —
> prefer the venv above.

**HuggingFace model weights** — downloaded on first run to the default cache
(NOT inside this repo). BGE-M3 + the reranker are ~4–5 GB combined:
```bash
rm -rf ~/.cache/huggingface        # local runs use this (HF_HOME unset)
```
> Inside Docker these live in the `docintel_models` volume instead (see §1).

**GPU torch** (only if you followed the GPU setup): it replaces the CPU torch
inside the same venv, so deleting `.venv` removes it too. No separate step.

---

## 3. Inside the repo

- `data/uploads/` — uploaded documents when running locally (Docker uses the
  `docintel_uploads` volume instead):
  ```bash
  rm -rf data/uploads
  ```
- `.env` — your local config copy (safe to keep; recreate from `.env.example`).

---

## 4. Ollama models (host-side, LLM)

The LLM runs in Ollama **on your host** (the app reaches it via
`host.docker.internal`), so its models are NOT in any Docker volume and NOT in
this repo. The default `qwen2.5:7b-instruct` is ~4.7 GB.

```bash
ollama list                        # see what's pulled
ollama rm qwen2.5:7b-instruct      # remove the model this project uses
```
> Only remove models you don't use elsewhere. Ollama itself is a separate
> host install, not managed by this project.

---

## 5. NOT installed by this project

The NVIDIA driver / CUDA toolkit (from the GPU setup) are **system-level** and
shared across everything on your machine — this project doesn't manage them, so
don't remove them as part of cleaning up the app.

---

## Full local wipe (Docker path), one shot

```bash
docker compose down -v
docker image rm docintel-api docintel-worker postgres:16 redis:7-alpine qdrant/qdrant:latest 2>/dev/null
```
For a local (non-Docker) install, also: `rm -rf .venv ~/.cache/huggingface data/uploads`
