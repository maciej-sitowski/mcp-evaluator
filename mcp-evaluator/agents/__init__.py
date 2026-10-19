"""Agent providers the runner can use for the agent + LLM-judge metrics.

Switch with ``AGENT_PROVIDER=copilot`` (or ``--agent-provider copilot``); the judge follows the agent
unless ``JUDGE_PROVIDER`` / ``--judge-provider`` says otherwise (handy to keep one fixed judge while you
compare agents). To add a provider, write a module with `DEFAULT_MODEL`, `preflight()`, `run_agent()` and
`judge()` (see `cursor.py`) and register it below.
"""

import importlib
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Tuple

DEFAULT_PROVIDER = "cursor"


@dataclass(frozen=True)
class Provider:
    name: str  # value of AGENT_PROVIDER
    label: str  # for messages
    module: str  # backend module, imported lazily so a missing SDK only matters when it is used
    credentials: Tuple[str, ...]  # env vars that can hold the key / token, first one set wins

    def load(self) -> ModuleType:
        return importlib.import_module(self.module)

    def api_key(self) -> str:
        for var in self.credentials:
            value = (os.environ.get(var) or "").strip()
            if value:
                return value
        return ""

    @property
    def default_model(self) -> str:
        return self.load().DEFAULT_MODEL


PROVIDERS = {
    p.name: p
    for p in (
        Provider("cursor", "Cursor", "agents.cursor", ("CURSOR_API_KEY",)),
        Provider("copilot", "GitHub Copilot", "agents.copilot", ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")),
    )
}


def get_provider(name: str) -> Provider:
    key = (name or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise SystemExit("Unknown agent provider %r (choose from: %s)" % (name, ", ".join(sorted(PROVIDERS))))
    return PROVIDERS[key]
