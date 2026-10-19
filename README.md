# Docs search evals

Three pieces: a mock MCP server that searches product docs, a one-shot runner that scores it, and a dashboard that reads the saved runs.

- [What a run does](#what-a-run-does)
- [Services](#services)
- [Run](#run)
- [Add a product](#add-a-product)
- [A different MCP server](#a-different-mcp-server)

```
products/          Docs, one folder per product. Mounted read-only into the server and the runner.
mcp_server/        Long-running. search_docs at http://localhost:8000/mcp
mcp-evaluator/     One shot. Golden sets live in goldens/<id>/. Writes results/runs/<id>.json
mcp-dashboard/     Long-running. Reads those files at http://localhost:8080
```

`docker compose up` starts the server and the dashboard. The runner is not a standing service. Start it with `docker compose run`.

No DeepEval account is used. Telemetry is off.

## What a run does

For each golden in `mcp-evaluator/goldens/<id>/`:

1. Call `search_docs` with the golden's fixed arguments and score the hits (expected docs, rank, product isolation, response shape, latency).
2. If an agent key is set, hand the customer question to Cursor or Copilot. That model calls the same tool, then a judge grades the answer.
3. Write one JSON file under `results/runs/`. The dashboard reads the folder on refresh.

Without a key, step 2 is skipped and the run says so.

## Services

**mcp-server** indexes `products/*/docs/*.md` on each search. `SEARCH_ENGINE=v1` is exact keywords. `v2` adds stemming, synonyms, and typo tolerance. Restart it after server code changes. Doc edits need no restart.

**mcp-evaluator** waits until the server is healthy, then runs. It reads golden sets from `goldens/<id>/` (same folder name as the product), mounts `products/` for `product.json`, and writes `results/`. The agent and the judge are chosen in the CONFIG block at the top of `mcp-evaluator/run_evals.py`, and so are the MCP tools (`TOOLS`). A flag or an environment variable overrides that for one run. `RUN_PARAMS` in that block is stored on the run and shown in the history table, so a score can be matched to the change that produced it. Secrets live in `.env` (see `.env.example`).

**mcp-dashboard** serves the page and `/api/runs`. It mounts the same `results/` folder, so a new file shows up on refresh. It can also delete runs.

## Run

```bash
docker compose up -d --build mcp-server mcp-dashboard
docker compose run --rm mcp-evaluator
```

Open http://localhost:8080.

```bash
docker compose run --rm mcp-evaluator --no-llm                 # step 1 only
docker compose run --rm mcp-evaluator --strict                 # exit 1 if a case fails
docker compose run --rm mcp-evaluator --product billing --case pro-plan-price
docker compose run --rm mcp-evaluator --param change="shorter prompt"
AGENT_PROVIDER=copilot docker compose run --rm mcp-evaluator
```

Cursor needs `CURSOR_API_KEY`. Copilot needs `COPILOT_GITHUB_TOKEN` (a PAT with the Copilot Requests permission).

## Add a product

Copy `products/product-one` and `mcp-evaluator/goldens/product-one`, and give both folders the same new id. In `product.json`, set `name`, `color`, and `default_arguments.product` to that id. Replace `docs/*.md` and the golden JSON.

A golden is one customer question. `expected_docs` is what search should return: `file.md#section`, `file.md` for any section of that file, or `[]` when nothing should be found. Optional: `forbidden_docs`, `reference_answer`, `must_include`. See `mcp-evaluator/goldens/product-one/core.json`.

## A different MCP server

```bash
# .env
MCP_URL=http://host.docker.internal:9000/mcp
MCP_AUTH_HEADER=Bearer <token>
```

```bash
docker compose run --rm --no-deps mcp-evaluator
```

The tool names are `TOOLS` in the CONFIG block of `mcp-evaluator/run_evals.py`. The direct check calls the first name; the agent is given only the list. `--tool name` (repeatable) or `TOOLS=name1,name2` overrides it for one run. Each hit still needs an id like `product/file.md#section`.

Chart.js loads from a CDN, so the browser needs network access to draw charts.
