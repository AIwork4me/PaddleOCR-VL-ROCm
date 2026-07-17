import json
from pathlib import Path

from eval.release_contract import KNOWN_V16_OFFICIAL_FAILURE
from eval.symmetric_score_input import prepare_score_input


def _stats(engine: str, failed: list[dict[str, str]]) -> dict[str, object]:
    successful = [
        {"image": f"page-{index:04d}.png", "status": "ok"} for index in range(1651 - len(failed))
    ]
    return {
        "count": 1651,
        "ok": len(successful),
        "fail": len(failed),
        "fallback": 0,
        "limit_pages": None,
        "engine": engine,
        "stats": successful + failed,
    }


def _write_markdown_set(directory: Path, images: list[str]) -> None:
    directory.mkdir()
    for image in images:
        (directory / f"{Path(image).stem}.md").write_text(image, encoding="utf-8")


def test_prepares_immutable_official_score_input_from_path_repair(tmp_path: Path) -> None:
    path_failures = [
        {"image": f"path-{index}.png", "status": "failed: No such file or directory"}
        for index in range(8)
    ]
    peg_failure = {
        "image": KNOWN_V16_OFFICIAL_FAILURE["image"],
        "status": "failed: peg-native",
    }
    source = tmp_path / "official-source"
    original = _stats("official", path_failures + [peg_failure])
    _write_markdown_set(
        source, [item["image"] for item in original["stats"] if item["status"] == "ok"]
    )
    (source / "_run_stats.json").write_text(json.dumps(original), encoding="utf-8")

    repair = _stats("official", [])
    repair["count"] = 8
    repair["ok"] = 8
    repair["stats"] = [{"image": item["image"], "status": "ok"} for item in path_failures]
    repair_path = tmp_path / "repair.json"
    repair_path.write_text(json.dumps(repair), encoding="utf-8")
    for item in path_failures:
        (source / f"{Path(item['image']).stem}.md").write_text(item["image"], encoding="utf-8")

    prepared = prepare_score_input(
        source_dir=source,
        destination_dir=tmp_path / "score-input",
        engine="official",
        repair_stats_path=repair_path,
    )

    effective = json.loads((prepared / "_run_stats.json").read_text(encoding="utf-8"))
    receipt = json.loads((prepared / "score-input-receipt.json").read_text(encoding="utf-8"))
    assert (effective["ok"], effective["fail"], effective["fallback"]) == (1650, 1, 0)
    assert len(list(prepared.glob("*.md"))) == 1650
    assert not (prepared / f"{Path(peg_failure['image']).stem}.md").exists()
    assert receipt["repaired_pages"] == 8
    assert json.loads((source / "_run_stats.json").read_text(encoding="utf-8")) == original
