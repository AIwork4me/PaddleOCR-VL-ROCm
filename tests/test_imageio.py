from PIL import Image

from paddleocr_vl_rocm.imageio import _crop_margin, _merge_images


def test_merge_images_stacks_vertically():
    a = Image.new("RGB", (10, 4), (1, 1, 1))
    b = Image.new("RGB", (10, 6), (2, 2, 2))
    merged = _merge_images([a, b], ["center"])
    assert merged.size == (10, 10)


def test_crop_margin_passthrough_on_uniform():
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    assert _crop_margin(img).size == (20, 20)
