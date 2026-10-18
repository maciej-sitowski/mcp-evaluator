"""Tiny in-memory search engine over the product documentation chunks (see docs.py).

Two engines are available so you can watch evaluation results change over time:

* ``v1`` - plain keyword matching. Exact tokens only: no plural handling,
  no synonyms, no typo tolerance.
* ``v2`` - same scoring, but adds naive stemming, a small synonym table and
  fuzzy matching of unknown words against the documentation vocabulary.

Pick one with the ``SEARCH_ENGINE`` environment variable (see docker-compose.yml).
A search always stays inside one product: that is what the evals' ProductIsolation check verifies.
"""

import difflib
import re
from typing import Any, Dict, List, Set, Tuple

import docs

ENGINES = ("v1", "v2")

STOPWORDS = {"a", "an", "the", "for", "with", "and", "of", "to", "in", "on", "i", "me", "my", "some",
             "how", "do", "does", "can", "is", "are", "it", "you", "what", "where"}

# How much a query token is worth depending on where it matches in a chunk.
FIELD_WEIGHTS = {"section": 3.0, "doc": 2.0, "text": 1.0}

SNIPPET_CHARS = 400

# v2 only. Keys and values are stemmed tokens.
SYNONYMS: Dict[str, List[str]] = {
    "price": ["pricing", "cost"],
    "cost": ["pricing", "price"],
    "remove": ["delete"],
    "erase": ["delete"],
    "undelete": ["restore"],
    "notify": ["alert"],
    "notification": ["alert"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _stem(token: str) -> str:
    """Very naive plural stripping: 'alerts' -> 'alert'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _field_tokens(chunk: Dict[str, Any], stem: bool) -> Dict[str, Set[str]]:
    fields = {
        "section": chunk["section"],
        "doc": "%s %s" % (chunk["doc_title"], chunk["doc"].replace(".md", "")),
        "text": chunk["text"],
    }
    out: Dict[str, Set[str]] = {}
    for field, text in fields.items():
        toks = _tokenize(text)
        out[field] = {_stem(t) for t in toks} if stem else set(toks)
    return out


class Index:
    """All chunks of all products, indexed once per engine."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.products = sorted({c["product"] for c in chunks})
        self._rows = {
            engine: [(c, _field_tokens(c, stem=(engine == "v2"))) for c in chunks] for engine in ENGINES
        }
        vocab = {tok for _, fields in self._rows["v2"] for toks in fields.values() for tok in toks}
        self.vocab = sorted(vocab)
        self.vocab_set = vocab

    def rows(self, engine: str, product: str) -> List[Tuple[Dict[str, Any], Dict[str, Set[str]]]]:
        return [(c, f) for c, f in self._rows[engine] if c["product"] == product]


_cache: Dict[str, Any] = {"signature": None, "index": None}


def get_index() -> Index:
    """Rebuilt automatically when a markdown file changes."""
    sig = docs.signature()
    if sig != _cache["signature"]:
        _cache.update(signature=sig, index=Index(docs.load_chunks()))
    return _cache["index"]


def _variants(token: str, engine: str, index: Index) -> List[str]:
    """Alternative spellings/words to try for one query token."""
    if engine == "v1":
        return [token]
    variants = [token]
    if token not in index.vocab_set and len(token) >= 4:
        variants += difflib.get_close_matches(token, index.vocab, n=1, cutoff=0.8)
    variants += SYNONYMS.get(token, [])
    return variants


def _token_score(variants: List[str], fields: Dict[str, Set[str]]) -> float:
    best = 0.0
    for variant in variants:
        score = sum(w for field, w in FIELD_WEIGHTS.items() if variant in fields[field])
        best = max(best, score)
    return best


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 1].rstrip() + "…"


def search(query: str, product: str, limit: int = 5, engine: str = "v1") -> Dict[str, Any]:
    if engine not in ENGINES:
        raise ValueError("Unknown search engine %r (expected one of %s)" % (engine, ENGINES))
    index = get_index()
    if product not in index.products:
        raise ValueError("Unknown product %r. Known products: %s" % (product, ", ".join(index.products) or "none"))

    tokens = [t for t in _tokenize(query) if t not in STOPWORDS]
    if engine == "v2":
        tokens = [_stem(t) for t in tokens]
    tokens = list(dict.fromkeys(tokens))  # de-duplicate, keep order

    matches = []
    if tokens:
        variants_per_token = [_variants(t, engine, index) for t in tokens]
        for chunk, fields in index.rows(engine, product):
            scores = [_token_score(v, fields) for v in variants_per_token]
            matched = sum(1 for s in scores if s > 0)
            if matched == 0:
                continue
            # Reward chunks that match more of the query's words.
            matches.append((sum(scores) * matched / len(tokens), chunk))

    matches.sort(key=lambda m: (-m[0], m[1]["id"]))
    results = [
        {
            "id": c["id"],
            "product": c["product"],
            "doc": c["doc"],
            "section": c["section"],
            "text": _snippet(c["text"]),
            "score": round(score, 2),
        }
        for score, c in matches[:limit]
    ]
    return {
        "query": query,
        "product": product,
        "engine": engine,
        "total_matches": len(matches),
        "results": results,
    }
