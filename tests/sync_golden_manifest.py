#!/usr/bin/env python3
"""Regenerate tests/golden_manifest.yaml from folder layout (no-bet / bet/placing / bet/placed)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from golden_lib import sync_manifest_from_folders  # noqa: E402

REPO = TESTS.parent
GOLDEN = REPO / "tests" / "golden"
MANIFEST = REPO / "tests" / "golden_manifest.yaml"


def main() -> None:
    manifest = sync_manifest_from_folders(GOLDEN)
    MANIFEST.write_text(
        yaml.dump({"images": manifest}, default_flow_style=False, sort_keys=False)
    )
    print(f"Wrote {len(manifest)} entries to {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
