from paddleocr_vl_rocm.content import _has_cjk, _normalize_vlm_result, _truncate_repetitive_content


def test_has_cjk():
    assert _has_cjk("中文") is True
    assert _has_cjk("english") is False


def test_truncate_repetitive_lines():
    # Need >= min_count (3000) chars for the repetitive-line branch to fire.
    content = "\n".join(["same"] * 610)
    assert _truncate_repetitive_content(content) == "same"


def test_normalize_inline_formula_to_dollars():
    out = _normalize_vlm_result("inline", r"\(x+1\) and \(y\)")
    assert "$" in out
