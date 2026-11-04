"""Shared helpers for golden screenshot tests."""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def folder_for_expected(expected: list[str]) -> str:
    """Review folder name(s) for a manifest expected outcome."""
    if expected == []:
        return "no-bet"
    if expected == ["bet_placing"]:
        return "bet/placing"
    if expected == ["bet_placed"]:
        return "bet/placed"
    raise ValueError(f"unsupported expected notify outcome: {expected!r}")


def expected_from_folder(rel_path: str) -> list[str]:
    """Expected notify events implied by review folder path."""
    rel_path = rel_path.replace("\\", "/")
    if rel_path.startswith("no-bet/"):
        return []
    if rel_path.startswith("bet/placing/"):
        return ["bet_placing"]
    if rel_path.startswith("bet/placed/"):
        return ["bet_placed"]
    raise ValueError(f"image not under no-bet/ or bet/placing|placed/: {rel_path!r}")


def sync_manifest_from_folders(golden_dir: Path) -> dict[str, list[str]]:
    """Build manifest entries from on-disk layout (folder = expected outcome)."""
    manifest: dict[str, list[str]] = {}
    for path in iter_golden_images(golden_dir):
        rel = path.relative_to(golden_dir).as_posix()
        manifest[rel] = expected_from_folder(rel)
    return dict(sorted(manifest.items()))


def iter_golden_images(golden_dir: Path) -> list[Path]:
    """All image files under golden_dir (any subfolder)."""
    out: list[Path] = []
    for path in sorted(golden_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            out.append(path)
    return out
