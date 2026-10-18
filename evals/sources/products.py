"""Loads the products under test: ``products/<id>/{product.json, docs/, goldens/*.json}``.

A product is just a folder. Add a folder and it is picked up by the MCP server (docs) and by the evals
(goldens) with no code changes. See the README for the file formats.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Used when a golden does not spell out `arguments`: the user's question is sent as the search query.
DEFAULT_QUERY_ARGUMENT = "query"


@dataclass
class Product:
    id: str
    name: str
    color: Optional[str] = None
    description: str = ""
    tool: str = "search_docs"
    default_arguments: Dict[str, Any] = field(default_factory=dict)
    assistant_prompt: Optional[str] = None
    goldens: List[Dict[str, Any]] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit("Invalid JSON in %s: %s" % (path, exc))


def _load_goldens(product_dir: Path) -> List[Dict[str, Any]]:
    goldens: List[Dict[str, Any]] = []
    for path in sorted((product_dir / "goldens").glob("*.json")):
        spec = _read_json(path)
        items = spec["goldens"] if isinstance(spec, dict) else spec
        for g in items:
            for key in ("id", "input"):
                if key not in g:
                    raise SystemExit("%s: every golden needs %r (got %s)" % (path, key, sorted(g)))
            g = dict(g)
            g["_file"] = path.name
            goldens.append(g)
    ids = [g["id"] for g in goldens]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit("%s: duplicate golden ids: %s" % (product_dir.name, ", ".join(dupes)))
    return goldens


def load_products(products_dir: Path, only: List[str] = (), only_cases: List[str] = ()) -> List[Product]:
    if not products_dir.is_dir():
        raise SystemExit("Products directory not found: %s" % products_dir)
    products: List[Product] = []
    for product_dir in sorted(p for p in products_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if only and product_dir.name not in only:
            continue
        meta_path = product_dir / "product.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        product = Product(
            id=product_dir.name,
            name=meta.get("name") or product_dir.name.replace("-", " ").title(),
            color=meta.get("color"),
            description=meta.get("description", ""),
            tool=meta.get("tool", "search_docs"),
            default_arguments=meta.get("default_arguments", {}),
            assistant_prompt=meta.get("assistant_prompt"),
            goldens=_load_goldens(product_dir),
        )
        if only_cases:
            product.goldens = [g for g in product.goldens if g["id"] in only_cases]
        if product.goldens:
            products.append(product)
    if not products:
        raise SystemExit(
            "No goldens found in %s (products: %s, cases: %s)" % (products_dir, list(only) or "all", list(only_cases) or "all")
        )
    return products


def normalize_patterns(entries: List[str], product_id: str, all_ids: List[str]) -> List[str]:
    """Doc references in goldens are relative to their product (``pricing.md#plans``).

    A reference that already starts with a product id (``product-two/pricing.md``) is kept as is, which
    lets one product's golden say "this other product's docs must not show up".
    """
    out = []
    for entry in entries:
        out.append(entry if entry.split("/", 1)[0] in all_ids else "%s/%s" % (product_id, entry))
    return out
