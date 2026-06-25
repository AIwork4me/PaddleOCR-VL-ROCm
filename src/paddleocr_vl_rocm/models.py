from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


@dataclass
class LightBlock:
    label: str
    content: str
    bbox: list[int]
    score: float
    cls_id: int
    image: Image.Image | None = None
    group_id: int | None = None
    merged: bool = False
    polygon_points: list[list[float]] | None = None
    figure_token_map: dict[str, str] = field(default_factory=dict)


@dataclass
class TableCell:
    text: str
    row_span: int
    col_span: int
    start_row: int
    end_row: int
    start_col: int
    end_col: int
