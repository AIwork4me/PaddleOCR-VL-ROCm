from paddleocr_vl_rocm.preprocess import _construct_img_path


def test_construct_img_path_format():
    path = _construct_img_path("image", [10, 20, 30, 40])
    assert path == "imgs/img_in_image_box_10_20_30_40.jpg"
