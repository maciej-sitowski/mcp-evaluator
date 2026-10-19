"""Loads the products under test from ``goldens/<id>/*.json``.

The folder name is the product id. A golden file may also carry ``name``, ``color``, ``arguments``
(merged under every question in that file) and ``assistant_prompt``. The MCP server's docs live
elsewhere and are not read here. See the README.
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
    tool: str = "search_docs"
    default_arguments: Dict[str, Any] = field(default_factory=dict)
    assistant_prompt: Optional[str] = None
    goldens: List[Dict[str, Any]] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit("Invalid JSON in %s: %s" % (path, exc))


def _load_goldens(golden_dir: Path) -> "tuple[Dict[str, Any], List[Dict[str, Any]]]":
    """One product: every ``*.json`` in ``goldens/<product-id>/``, plus the file-level fields."""
    meta: Dict[str, Any] = {}
    goldens: List[Dict[str, Any]] = []
    for path in sorted(golden_dir.glob("*.json")):
        spec = _read_json(path)
        if isinstance(spec, dict):
            items = spec.get("goldens")
            if not isinstance(items, list):
                raise SystemExit("%s: expected a 'goldens' list" % path)
            file_args = spec.get("arguments") or {}
            if not isinstance(file_args, dict):
                raise SystemExit("%s: 'arguments' must be an object" % path)
            for key in ("name", "color", "assistant_prompt"):
                if spec.get(key) and key not in meta:
                    meta[key] = spec[key]
            if file_args and "arguments" not in meta:
                meta["arguments"] = file_args
        elif isinstance(spec, list):
            items = spec
            file_args = {}
        else:
            raise SystemExit("%s: expected an object with a 'goldens' list" % path)
        for g in items:
            for key in ("id", "input"):
                if key not in g:
                    raise SystemExit("%s: every golden needs %r (got %s)" % (path, key, sorted(g)))
            g = dict(g)
            g["_file"] = path.name
            g["_shared_arguments"] = dict(file_args)
            goldens.append(g)
    ids = [g["id"] for g in goldens]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit("%s: duplicate golden ids: %s" % (golden_dir.name, ", ".join(dupes)))
    return meta, goldens


def load_products(goldens_dir: Path, only: List[str] = (), only_cases: List[str] = ()) -> List[Product]:
    if not goldens_dir.is_dir():
        raise SystemExit("Goldens directory not found: %s" % goldens_dir)
    products: List[Product] = []
    for golden_dir in sorted(p for p in goldens_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if only and golden_dir.name not in only:
            continue
        meta, goldens = _load_goldens(golden_dir)
        product = Product(
            id=golden_dir.name,
            name=meta.get("name") or golden_dir.name.replace("-", " ").title(),
            color=meta.get("color"),
            default_arguments=meta.get("arguments") or {},
            assistant_prompt=meta.get("assistant_prompt"),
            goldens=goldens,
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
