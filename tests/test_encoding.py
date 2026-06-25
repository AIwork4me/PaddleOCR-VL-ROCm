from paddleocr_vl_rocm.encoding import _data_url_from_bytes, _sha256_hex


def test_data_url_encoding():
    assert _data_url_from_bytes(b"hi", "image/png") == "data:image/png;base64,aGk="


def test_sha256_known_empty():
    assert _sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
