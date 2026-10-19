"""Loads the products under test.

Docs and ``product.json`` live in ``products/<id>/`` (what the MCP server searches). Golden sets live
next to this runner, in ``goldens/<id>/*.json``, joined to a product by the folder name. Add a product
folder and a matching golden set and both are picked up with no code changes. See the README.
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


def _load_goldens(golden_dir: Path) -> List[Dict[str, Any]]:
    """One product's golden set: every ``*.json`` in ``goldens/<product-id>/``."""
    if not golden_dir.is_dir():
        return []
    goldens: List[Dict[str, Any]] = []
    for path in sorted(golden_dir.glob("*.json")):
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
        raise SystemExit("%s: duplicate golden ids: %s" % (golden_dir.name, ", ".join(dupes)))
    return goldens


def load_products(
    products_dir: Path, goldens_dir: Path, only: List[str] = (), only_cases: List[str] = ()
) -> List[Product]:
    if not products_dir.is_dir():
        raise SystemExit("Products directory not found: %s" % products_dir)
    if not goldens_dir.is_dir():
        raise SystemExit("Goldens directory not found: %s" % goldens_dir)
    product_dirs = sorted(p for p in products_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    known = {p.name for p in product_dirs}
    extra = sorted(
        p.name
        for p in goldens_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in known and (not only or p.name in only)
    )
    if extra:
        raise SystemExit("Golden sets with no matching product in %s: %s" % (products_dir, ", ".join(extra)))
    products: List[Product] = []
    for product_dir in product_dirs:
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
            goldens=_load_goldens(goldens_dir / product_dir.name),
        )
        if only_cases:
            product.goldens = [g for g in product.goldens if g["id"] in only_cases]
        if product.goldens:
            products.append(product)
    if not products:
        raise SystemExit(
            "No goldens found in %s (products: %s, cases: %s)"
            % (goldens_dir, list(only) or "all", list(only_cases) or "all")
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
