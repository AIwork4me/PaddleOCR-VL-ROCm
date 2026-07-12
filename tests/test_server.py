from unittest.mock import Mock

import pytest
import requests

from paddleocr_vl_rocm import server


def test_successful_server_check_reports_count_without_model_identifiers(monkeypatch) -> None:
    response = Mock(status_code=200)
    response.json.return_value = {"data": [{"id": r"C:\Users\private\models\secret.gguf"}]}
    console = Mock()
    monkeypatch.setattr(server.requests, "get", Mock(return_value=response))
    monkeypatch.setattr(server, "get_console", Mock(return_value=console))

    payload = server.check_openai_compatible_server("http://127.0.0.1:8111/v1")

    assert payload == response.json.return_value
    rendered = "\n".join(str(call.args[0]) for call in console.print.call_args_list)
    assert "1 model(s)" in rendered
    assert "private" not in rendered
    assert "secret.gguf" not in rendered


def test_server_check_closes_successful_response(monkeypatch) -> None:
    response = Mock(status_code=200)
    response.json.return_value = {"data": []}
    monkeypatch.setattr(server.requests, "get", Mock(return_value=response))
    monkeypatch.setattr(server, "get_console", Mock(return_value=Mock()))

    server.check_openai_compatible_server("http://127.0.0.1:8111/v1")

    response.close.assert_called_once()


def test_connection_error_redacts_credentials_and_query_secret(monkeypatch) -> None:
    secret_url = "http://user:password@127.0.0.1:8111/v1?api_key=top-secret"
    request = Mock(side_effect=requests.RequestException(f"failed for {secret_url}"))
    monkeypatch.setattr(server.requests, "get", request)

    with pytest.raises(RuntimeError) as caught:
        server.check_openai_compatible_server(secret_url)

    message = str(caught.value)
    assert "password" not in message
    assert "top-secret" not in message
    assert "127.0.0.1:8111" in message
    assert caught.value.__cause__ is None
    assert request.call_args.args[0] == (
        "http://user:password@127.0.0.1:8111/v1/models?api_key=top-secret"
    )


def test_http_error_does_not_echo_response_body(monkeypatch) -> None:
    response = Mock(status_code=500)
    response.text = r"failure at C:\Users\private\models\secret.gguf token=top-secret"
    monkeypatch.setattr(server.requests, "get", Mock(return_value=response))

    with pytest.raises(RuntimeError) as caught:
        server.check_openai_compatible_server("http://127.0.0.1:8111/v1")

    message = str(caught.value)
    assert "HTTP 500" in message
    assert "private" not in message
    assert "top-secret" not in message
    response.close.assert_called_once()


def test_non_json_error_does_not_echo_response_body(monkeypatch) -> None:
    response = Mock(status_code=200)
    response.json.side_effect = ValueError("invalid JSON")
    response.text = r"C:\Users\private\models\secret.gguf"
    monkeypatch.setattr(server.requests, "get", Mock(return_value=response))

    with pytest.raises(RuntimeError) as caught:
        server.check_openai_compatible_server("http://127.0.0.1:8111/v1")

    assert "non-JSON" in str(caught.value)
    assert "private" not in str(caught.value)
    assert caught.value.__cause__ is None
    response.close.assert_called_once()


def test_redacted_url_preserves_ipv6_brackets() -> None:
    assert server._redacted_url("http://user:pass@[::1]:8111/v1?token=secret") == (
        "http://[::1]:8111/v1"
    )
