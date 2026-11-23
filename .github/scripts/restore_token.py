"""Restore Antigravity / Google Application Default Credentials from AGY_TOKEN_JSON secret."""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    raw = os.environ.get("AGY_TOKEN_JSON", "").strip()
    if not raw:
        print("No AGY_TOKEN_JSON secret provided; skipping Antigravity token restoration.")
        return

    try:
        data = json.loads(raw)
    except Exception as exc:
        print(f"::warning::Failed to parse AGY_TOKEN_JSON: {exc}", file=sys.stderr)
        return

    if "type" not in data:
        data["type"] = "authorized_user"

    paths = [
        os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
        os.path.expanduser("~/.gemini/antigravity-acp/acp_token.json"),
    ]
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.chmod(p, 0o600)
    print("Restored Antigravity session credentials successfully.")


if __name__ == "__main__":
    main()
