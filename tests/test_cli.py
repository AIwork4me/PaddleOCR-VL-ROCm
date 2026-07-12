from __future__ import annotations

from types import SimpleNamespace

import pytest

from paddleocr_vl_rocm import cli
from paddleocr_vl_rocm.cli import ExitCode, build_parser, parse_args
from paddleocr_vl_rocm.doctor import CheckResult


def test_cli_parser_accepts_documented_smoke_command():
    args = build_parser().parse_args(
        [
            "--input",
            "examples/input/handwrite_ch_demo.png",
            "--output",
            "outputs/smoke",
            "--layout-model",
            "models/PP-DocLayoutV3-onnx",
            "--server-url",
            "http://127.0.0.1:8000/v1",
            "--api-model-name",
            "PaddleOCR-VL-1.5-0.9B",
            "--vlm-backend",
            "vllm-server",
            "--layout-provider",
            "directml",
        ]
    )

    assert args.input.endswith("handwrite_ch_demo.png")
    assert args.vlm_backend == "vllm-server"
    assert args.api_model_name == "PaddleOCR-VL-1.5-0.9B"
    assert args.layout_provider == "directml"


def test_cli_layout_provider_defaults_to_auto():
    args = build_parser().parse_args(["--input", "input.png"])

    assert args.layout_provider == "auto"


def test_cli_parses_setup_doctor_run_and_legacy_journeys():
    assert parse_args(["setup", "--auto"]).command == "setup"
    assert parse_args(["doctor", "--json"]).command == "doctor"
    assert parse_args(["run", "invoice.png"]).input == "invoice.png"
    assert parse_args(["--input", "invoice.png"]).input == "invoice.png"


def test_run_command_maps_to_legacy_defaults():
    run = parse_args(["run", "invoice.png"])
    legacy = parse_args(["--input", "invoice.png"])

    for name in (
        "input",
        "output",
        "layout_model",
        "layout_provider",
        "server_url",
        "api_model_name",
        "vlm_backend",
        "max_new_tokens",
        "timeout",
        "seed",
        "threshold",
        "vlm_max_workers",
        "skip_server_check",
    ):
        assert getattr(run, name) == getattr(legacy, name)


def test_run_accepts_options_before_positional_input():
    args = parse_args(["run", "--server-url", "http://host.test/v1", "invoice.png"])

    assert args.input == "invoice.png"
    assert args.server_url == "http://host.test/v1"


def test_run_help_exits_successfully(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["run", "--help"])

    assert exc.value.code == 0
    assert "paddleocr-vl-rocm run" in capsys.readouterr().out


def test_setup_auto_installs_starts_and_prints_next_command(tmp_path, monkeypatch, capsys):
    result = SimpleNamespace(
        root=tmp_path,
        config_path=tmp_path / "config.json",
        layout_model_dir=tmp_path / "models" / "layout model",
        llama_server=tmp_path / "runtime" / "llama-server.exe",
        main_gguf=tmp_path / "models" / "main.gguf",
        mmproj=tmp_path / "models" / "mmproj.gguf",
    )
    setup = monkeypatch.setattr(cli, "setup_managed_runtime", lambda options: result)
    started = []
    monkeypatch.setattr(cli, "start_managed_server", lambda value: started.append(value))

    exit_code = cli.main(["setup", "--auto", "--root", str(tmp_path)])

    assert setup is None
    assert started == [result]
    assert exit_code == ExitCode.OK
    output = capsys.readouterr().out
    assert "Next: & 'paddleocr-vl-rocm' 'run' 'C:\\path\\to\\image.png'" in output
    assert "'--server-url' 'http://127.0.0.1:8111/v1'" in output


def test_powershell_command_quotes_spaces_and_apostrophes():
    command = cli._powershell_command([r"C:\A User's App\server.exe", "--port", "8111"])

    assert command == "& 'C:\\A User''s App\\server.exe' '--port' '8111'"


def test_doctor_json_uses_stable_environment_exit_code(monkeypatch, capsys):
    checks = [CheckResult("server", "FAIL", "offline", "Start it.", {})]
    monkeypatch.setattr(cli, "run_doctor", lambda config, server_url=None: checks)

    exit_code = cli.main(["doctor", "--json"])

    assert exit_code == ExitCode.ENVIRONMENT
    assert '"status": "FAIL"' in capsys.readouterr().out


def test_setup_failure_uses_download_exit_code(monkeypatch):
    monkeypatch.setattr(
        cli,
        "setup_managed_runtime",
        lambda _options: (_ for _ in ()).throw(cli.SetupDownloadError("download failed")),
    )

    assert cli.main(["setup", "--no-start"]) == ExitCode.DOWNLOAD


def test_setup_environment_failure_uses_environment_exit_code(monkeypatch):
    monkeypatch.setattr(
        cli,
        "setup_managed_runtime",
        lambda _options: (_ for _ in ()).throw(RuntimeError("runtime validation failed")),
    )

    assert cli.main(["setup", "--no-start"]) == ExitCode.ENVIRONMENT


def test_setup_auto_and_no_start_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        parse_args(["setup", "--auto", "--no-start"])

    assert exc.value.code == ExitCode.USAGE


def test_run_failure_uses_inference_exit_code(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_run_inference",
        lambda _args: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )

    assert cli.main(["run", "input.png"]) == ExitCode.INFERENCE


def test_run_server_error_does_not_leak_url_secrets(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "check_openai_compatible_server",
        lambda _url: (_ for _ in ()).throw(
            RuntimeError("failed http://user:password@host/v1?api_key=top-secret body-secret")
        ),
    )

    exit_code = cli.main(
        ["run", "input.png", "--server-url", "http://user:password@host/v1?api_key=top-secret"]
    )

    assert exit_code == ExitCode.SERVER
    error = capsys.readouterr().err
    assert "password" not in error
    assert "top-secret" not in error
    assert "body-secret" not in error


def test_cli_preflight_disables_pipeline_server_recheck(monkeypatch, tmp_path):
    captured = {}

    class FakeResult:
        def print(self):
            pass

        def save_to_json(self, output):
            return tmp_path / "result.json"

        def save_to_markdown(self, output, pretty=False):
            return tmp_path / "result.md"

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def predict(self, _input):
            return FakeResult()

    monkeypatch.setattr(cli, "PaddleOCRVLROCm", FakePipeline)
    monkeypatch.setattr(cli, "check_openai_compatible_server", lambda _url: None)

    assert cli.main(["run", "input.png", "--output", str(tmp_path)]) == ExitCode.OK
    assert captured["skip_server_check"] is True


def test_exit_codes_are_stable():
    assert {item.name: item.value for item in ExitCode} == {
        "OK": 0,
        "USAGE": 2,
        "ENVIRONMENT": 10,
        "DOWNLOAD": 11,
        "SERVER": 12,
        "INFERENCE": 13,
        "PARTIAL": 14,
    }
