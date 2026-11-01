from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    x: float
    y: float
    w: float
    h: float

    def crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x0 = int(self.x * width)
        y0 = int(self.y * height)
        x1 = int(min(width, (self.x + self.w) * width))
        y1 = int(min(height, (self.y + self.h) * height))
        if x1 <= x0 or y1 <= y0:
            return frame[0:0, 0:0]
        return frame[y0:y1, x0:x1]


@dataclass
class RegionSet:
    search_bar: Region
    market_row: Region
    bet_slip: Region

    @classmethod
    def from_config(cls, cfg: dict) -> RegionSet:
        r = cfg["regions"]

        def reg(name: str) -> Region:
            block = r[name]
            return Region(
                x=float(block["x"]),
                y=float(block["y"]),
                w=float(block["w"]),
                h=float(block["h"]),
            )

        return cls(
            search_bar=reg("search_bar"),
            market_row=reg("market_row"),
            bet_slip=reg("bet_slip"),
        )
