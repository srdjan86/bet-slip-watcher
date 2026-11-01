from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import pytesseract
from PIL import Image


class OcrEngine:
    def __init__(self, languages: list[str], gpu: bool = False, scale: float = 2.0) -> None:
        lang = "+".join(languages) if languages else "eng"
        self._lang = lang
        self._scale = max(scale, 1.0)

    def read_lines(
        self,
        bgr: np.ndarray,
        *,
        psm: int = 6,
        scale: float | None = None,
    ) -> list[str]:
        if bgr.size == 0:
            return []
        factor = max(scale if scale is not None else self._scale, 1.0)
        if factor != 1.0:
            bgr = cv2.resize(
                bgr,
                None,
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_CUBIC,
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        text = pytesseract.image_to_string(
            pil, lang=self._lang, config=f"--psm {psm}"
        )
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return _dedupe_lines(lines)


def _dedupe_lines(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out
