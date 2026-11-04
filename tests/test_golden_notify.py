"""Regression tests: review screenshots must match expected Telegram outcomes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bet_slip_watcher.notify_eval import evaluate_notify_events
from golden_lib import folder_for_expected, iter_golden_images

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
MANIFEST = REPO_ROOT / "tests" / "golden_manifest.yaml"
CONFIG = REPO_ROOT / "config.yaml"


def _load_manifest() -> dict[str, list[str]]:
    data = yaml.safe_load(MANIFEST.read_text())
    images: dict[str, list] = data["images"]
    out: dict[str, list[str]] = {}
    for rel, expected in images.items():
        out[rel.replace("\\", "/")] = list(expected or [])
    return out


def _load_cases() -> list[tuple[str, list[str]]]:
    manifest = _load_manifest()
    cases: list[tuple[str, list[str]]] = []
    for rel, expected in sorted(manifest.items()):
        path = GOLDEN_DIR / rel
        if not path.is_file():
            pytest.fail(f"missing golden image: {path}")
        cases.append((rel, expected))
    return cases


_MANIFEST = _load_manifest()
_CASES = _load_cases()


@pytest.mark.parametrize(
    "rel_path,expected",
    _CASES,
    ids=[name.replace("/", "__")[:56] for name, _ in _CASES],
)
def test_golden_notify_outcome(rel_path: str, expected: list[str]) -> None:
    """Pipeline outcome must match manifest (not inferred from folder)."""
    path = GOLDEN_DIR / rel_path
    actual = evaluate_notify_events(path, CONFIG)
    assert actual == expected, (
        f"{rel_path}: expected notify {expected!r}, got {actual!r}"
    )


@pytest.mark.parametrize(
    "rel_path,expected",
    _CASES,
    ids=[name.replace("/", "__")[:56] for name, _ in _CASES],
)
def test_golden_folder_layout(rel_path: str, expected: list[str]) -> None:
    """Folder layout matches manifest so you can eyeball mis-sorted files."""
    want_prefix = folder_for_expected(expected) + "/"
    assert rel_path.startswith(want_prefix), (
        f"{rel_path}: expected under {want_prefix!r} for outcome {expected!r}"
    )


def test_golden_manifest_covers_all_images() -> None:
    """Every image under tests/golden is listed in the manifest."""
    on_disk = {
        p.relative_to(GOLDEN_DIR).as_posix()
        for p in iter_golden_images(GOLDEN_DIR)
    }
    in_manifest = set(_MANIFEST.keys())
    assert on_disk == in_manifest, (
        f"manifest/disk mismatch: "
        f"only on disk {sorted(on_disk - in_manifest)!r}, "
        f"only in manifest {sorted(in_manifest - on_disk)!r}"
    )
