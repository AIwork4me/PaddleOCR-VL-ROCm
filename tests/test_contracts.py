from paddleocr_vl_rocm.contracts import fingerprint, redact


def test_contract_fingerprint_is_order_independent():
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})


def test_redact_removes_credentials_recursively():
    value = {
        "Authorization": "Bearer secret",
        "url": "https://host/path?token=secret",
        "nested": {"api_key": "secret"},
    }
    assert redact(value) == {
        "Authorization": "<redacted>",
        "url": "https://host/path?token=%3Credacted%3E",
        "nested": {"api_key": "<redacted>"},
    }
