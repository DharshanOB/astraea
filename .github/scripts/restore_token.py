"""Restore Antigravity / Google Application Default Credentials from AGY_TOKEN_JSON secret."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

CONSUMER_OAUTH_PROJECTS: frozenset[str] = frozenset({"aicode-consumers"})


def _write_secret_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.chmod(path, 0o600)


def restore_token(raw: str, home: Path | None = None) -> str:
    """Restore AGY_TOKEN_JSON to either Antigravity ACP storage or Google Cloud ADC."""
    stripped = raw.strip()
    if not stripped:
        print("No AGY_TOKEN_JSON secret provided; skipping Antigravity token restoration.")
        return "none"

    try:
        parsed = json.loads(stripped)
    except Exception as exc:
        print(f"::warning::Failed to parse AGY_TOKEN_JSON: {exc}", file=sys.stderr)
        return "invalid"

    if not isinstance(parsed, dict):
        print(
            "::warning::Failed to parse AGY_TOKEN_JSON: expected a JSON object.",
            file=sys.stderr,
        )
        return "invalid"

    base_home = home if home is not None else Path.home()
    cred_type = str(parsed.get("type", "")).strip()
    project_id = str(parsed.get("project_id") or parsed.get("quota_project_id") or "").strip()
    is_gcp_adc = cred_type in {"service_account", "external_account"} or (
        bool(project_id) and project_id not in CONSUMER_OAUTH_PROJECTS
    )

    if is_gcp_adc:
        adc_data: dict[str, Any] = dict(parsed)
        if "type" not in adc_data:
            adc_data["type"] = "authorized_user"
        adc_path = base_home / ".config" / "gcloud" / "application_default_credentials.json"
        _write_secret_json(adc_path, adc_data)
        print("Restored Google Cloud ADC credentials successfully.")
        return "gcp_adc"

    acp_path = base_home / ".gemini" / "antigravity-acp" / "acp_token.json"
    _write_secret_json(acp_path, parsed)
    print("Restored Antigravity session token successfully.")
    return "antigravity"


def main() -> None:
    """Entrypoint reading AGY_TOKEN_JSON from the environment."""
    restore_token(os.environ.get("AGY_TOKEN_JSON", ""))


if __name__ == "__main__":
    main()
