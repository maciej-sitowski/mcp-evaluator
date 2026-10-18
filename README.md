# Docs search evals

Three pieces: a mock MCP server that searches product docs, a one-shot runner that scores it, and a dashboard that reads the saved runs.

- [What a run does](#what-a-run-does)
- [Services](#services)
- [Run](#run)
- [Add a product](#add-a-product)
- [A different MCP server](#a-different-mcp-server)

```
products/     Docs and goldens, one folder per product. Mounted read-only into the server and the runner.
mcp_server/   Long-running. search_docs at http://localhost:8000/mcp
evals/        One shot. Scores the goldens and writes results/runs/<id>.json
dashboard/    Long-running. Reads those files at http://localhost:8080
```

`docker compose up` starts the server and the dashboard. The runner is not a standing service. Start it with `docker compose run`.

No DeepEval account is used. Telemetry is off.

## What a run does

For each golden in `products/<id>/goldens/`:

1. Call `search_docs` with the golden's fixed arguments and score the hits (expected docs, rank, product isolation, response shape, latency).
2. If an agent key is set, hand the customer question to Cursor or Copilot. That model calls the same tool, then a judge grades the answer.
3. Write one JSON file under `results/runs/`. The dashboard reads the folder on refresh.

Without a key, step 2 is skipped and the run says so.

## Services

**mcp-server** indexes `products/*/docs/*.md` on each search. `SEARCH_ENGINE=v1` is exact keywords. `v2` adds stemming, synonyms, and typo tolerance. Restart it after server code changes. Doc edits need no restart.

**evals** waits until the server is healthy, then runs. It mounts `products/` and writes `results/`. The agent and the judge are chosen in the CONFIG block at the top of `evals/run_evals.py`. A flag or an environment variable overrides that for one run. Secrets live in `.env` (see `.env.example`).

**dashboard** serves the page and `/api/runs`. It mounts the same `results/` folder, so a new file shows up on refresh. It can also delete runs.

## Run

```bash
docker compose up -d --build mcp-server dashboard
docker compose run --rm evals
```

Open http://localhost:8080.

```bash
docker compose run --rm evals --no-llm                 # step 1 only
docker compose run --rm evals --strict                 # exit 1 if a case fails
docker compose run --rm evals --product billing --case pro-plan-price
AGENT_PROVIDER=copilot docker compose run --rm evals
```

Cursor needs `CURSOR_API_KEY`. Copilot needs `COPILOT_GITHUB_TOKEN` (a PAT with the Copilot Requests permission).

## Add a product

Copy `products/product-one`. In `product.json`, set `name`, `color`, and `default_arguments.product` to the folder name. Replace `docs/*.md` and `goldens/*.json`.

A golden is one customer question. `expected_docs` is what search should return: `file.md#section`, `file.md` for any section of that file, or `[]` when nothing should be found. Optional: `forbidden_docs`, `reference_answer`, `must_include`. See `products/product-one/goldens/core.json`.

## A different MCP server

```bash
# .env
MCP_URL=http://host.docker.internal:9000/mcp
MCP_AUTH_HEADER=Bearer <token>
```

```bash
docker compose run --rm --no-deps evals
```

The tool name must match `product.json`, and each hit needs an id like `product/file.md#section`.

Chart.js loads from a CDN, so the browser needs network access to draw charts.
