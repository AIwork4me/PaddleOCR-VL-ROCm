from __future__ import annotations

import base64
import hashlib
import mimetypes
from io import BytesIO
from pathlib import Path

from PIL import Image


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def _data_url_from_bytes(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _encode_png_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/png"
    return _data_url_from_bytes(path.read_bytes(), mime_type)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
