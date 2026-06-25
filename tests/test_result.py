from __future__ import annotations

import json

from paddleocr_vl_rocm.result import PaddleOCRVLROCmResult


def test_result_saves_paddleocr_vl_style_files(tmp_path):
    result = PaddleOCRVLROCmResult(
        {
            "input_path": "examples/input/handwrite_ch_demo.png",
            "layout_det_res": {"boxes": []},
            "parsing_res_list": [{"block_label": "text", "block_content": "hello"}],
        },
        markdown_text="hello\n",
    )

    json_path = result.save_to_json(tmp_path)
    md_path = result.save_to_markdown(tmp_path, pretty=False)

    assert json_path.name == "handwrite_ch_demo_res.json"
    assert md_path.name == "handwrite_ch_demo.md"
    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["parsing_res_list"][0]["block_content"]
        == "hello"
    )
    assert md_path.read_text(encoding="utf-8") == "hello\n"
