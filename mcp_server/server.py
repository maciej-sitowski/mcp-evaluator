"""MCP server exposing a single tool: `search_docs`.

It searches the markdown documentation in ``products/<product-id>/docs/``. Transport: streamable HTTP,
endpoint ``/mcp``. A plain ``/health`` route is provided for docker healthchecks.
"""

import os
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from search import ENGINES, get_index, search

ENGINE = os.environ.get("SEARCH_ENGINE", "v1").strip().lower()
if ENGINE not in ENGINES:
    raise SystemExit("SEARCH_ENGINE must be one of %s, got %r" % (ENGINES, ENGINE))

mcp = FastMCP(
    "docs-search",
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", "8000")),
    stateless_http=True,
    json_response=True,
)

MAX_LIMIT = 20


@mcp.tool()
def search_docs(query: str, product: str, limit: int = 5) -> Dict[str, Any]:
    """Search one product's documentation by keywords.

    Args:
        query: Free-text keywords, e.g. "restore deleted note" or "connect slack".
        product: Id of the product whose documentation to search, e.g. "product-one".
            Results never come from another product.
        limit: Maximum number of results to return (1-20, default 5).

    Returns:
        The best matching documentation sections, most relevant first. Each result has an id
        (``<product>/<file>#<section>``), the doc, the section heading, a text snippet and a relevance score.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError("limit must be between 1 and %d" % MAX_LIMIT)
    return search(query, product=product, limit=limit, engine=ENGINE)


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "engine": ENGINE, "products": get_index().products})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
