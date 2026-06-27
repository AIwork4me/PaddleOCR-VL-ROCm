from paddleocr_vl_rocm.geometry import _area, _overlap_ratio


def test_area_positive_and_zero():
    assert _area([0, 0, 5, 5]) == 25.0
    assert _area([5, 5, 0, 0]) == 0.0


def test_overlap_ratio_small_and_union():
    big = [0, 0, 10, 10]
    small = [2, 2, 8, 8]
    assert _overlap_ratio(big, small, mode="small") == 1.0
    assert abs(_overlap_ratio(big, small, mode="union") - 0.36) < 1e-9
