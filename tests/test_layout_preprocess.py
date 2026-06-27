from __future__ import annotations

from pathlib import Path

from paddleocr_vl_rocm.layout import preprocess_layout_image


def test_layout_preprocess_uses_expected_onnx_inputs():
    image = Path("examples/input/handwrite_ch_demo.png")
    feeds, image_size, _ = preprocess_layout_image(image)

    assert set(feeds) == {"im_shape", "image", "scale_factor"}
    assert feeds["image"].shape == (1, 3, 800, 800)
    assert feeds["im_shape"].shape == (1, 2)
    assert feeds["scale_factor"].shape == (1, 2)
    assert image_size[0] > 0 and image_size[1] > 0
