"""Cover the standard-CLI parse path -> ``server.infer_one`` without a server/GPU.

The rocmdoc-v2 migration wired ``standard_cli.cmd_parse`` to a shared VLM core
``server.infer_one``, but that path had no test (and ``infer_one`` itself was
missing until it was implemented). These tests drive it with a fake
``openai.OpenAI`` client so CI asserts the request shape + result handling with
no network and no model.
"""

from __future__ import annotations

from paddleocr_vl_rocm import standard_cli
from paddleocr_vl_rocm.server import infer_one

# Minimal 1x1 PNG; _image_data_url only reads bytes + base64, so contents need
# not be decodable by a real model (the fake client never decodes them).
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, choices: list) -> None:
        self.choices = choices


class _FakeCompletions:
    """Records ``create(**kwargs)`` and returns a canned response."""

    def __init__(self, choices: list, captured: dict) -> None:
        self._choices = choices
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeResponse(self._choices)


class _FakeChat:
    def __init__(self, choices: list, captured: dict) -> None:
        self.completions = _FakeCompletions(choices, captured)


class _FakeClient:
    """Minimal ``openai.OpenAI`` double for ``infer_one`` / ``cmd_parse``."""

    def __init__(self, content: str = "ok", choices: list | None = None) -> None:
        self.captured: dict = {}
        resolved = [_FakeChoice(content)] if choices is None else choices
        self.chat = _FakeChat(resolved, self.captured)


def _write_png(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_BYTES)


def test_infer_one_builds_vision_request_and_returns_text(tmp_path):
    img = tmp_path / "page.png"
    _write_png(img)
    client = _FakeClient(content="# heading")

    out = infer_one(client, str(img), model="PaddleOCR-VL-1.6-GGUF.gguf")

    assert out == "# heading"
    captured = client.captured
    assert captured["model"] == "PaddleOCR-VL-1.6-GGUF.gguf"
    assert captured["temperature"] == 0.0
    content = captured["messages"][0]["content"]
    image_url = next(c["image_url"]["url"] for c in content if c.get("type") == "image_url")
    text = next(c["text"] for c in content if c.get("type") == "text")
    assert image_url.startswith("data:image/")  # same encoder as vlm/client
    assert text == "OCR:"  # codebase default parse prompt


def test_infer_one_returns_empty_string_when_no_choices(tmp_path):
    img = tmp_path / "page.png"
    _write_png(img)

    assert infer_one(_FakeClient(choices=[]), str(img), model="m") == ""


def test_cmd_parse_writes_markdown_per_image(monkeypatch, tmp_path):
    img_dir = tmp_path / "in"
    for name in ("a.png", "b.png"):
        _write_png(img_dir / name)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(standard_cli, "_build_client", lambda _url: _FakeClient(content="# md"))

    rc = standard_cli.cmd_parse(
        img_dir=img_dir,
        out_dir=out_dir,
        platform="windows-hip",
        backend="llama-cpp",
        server_url="http://127.0.0.1:8111/v1",
        model=None,
        limit=None,
    )

    assert rc == standard_cli.EXIT_OK
    assert (out_dir / "a.md").read_text(encoding="utf-8") == "# md"
    assert (out_dir / "b.md").read_text(encoding="utf-8") == "# md"
