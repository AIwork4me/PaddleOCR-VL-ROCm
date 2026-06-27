from paddleocr_vl_rocm.markdown import _collapse_soft_newlines, _format_title_text


def test_collapse_soft_newlines():
    assert _collapse_soft_newlines("hyphen-\nated") == "hyphenated"
    assert _collapse_soft_newlines("a\nb") == "a b"


def test_format_title_text_heading_levels():
    assert _format_title_text("1.2.3 Results").startswith("#### ")
    assert _format_title_text("Introduction").startswith("## ")
