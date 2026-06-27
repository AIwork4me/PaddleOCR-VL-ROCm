from paddleocr_vl_rocm.table import _convert_otsl_to_html


def test_otsl_to_html_basic_table():
    otsl = "<fcel>A<ecel><nl><fcel>B<ecel><nl>"
    html = _convert_otsl_to_html(otsl)
    assert html.startswith("<table>") and html.endswith("</table>")
    assert "A" in html and "B" in html
    assert html.count("<tr>") == 2


def test_otsl_to_html_empty_input_returns_empty_string():
    # Behavior-preserving: empty input yields no cells, so the converter
    # returns an empty string (the early-return guard in _convert_otsl_to_html).
    out = _convert_otsl_to_html("")
    assert out == ""
