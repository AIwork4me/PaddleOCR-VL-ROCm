from paddleocr_vl_rocm.models import LightBlock
from paddleocr_vl_rocm.serialize import _block_to_json


def test_block_to_json_shape():
    block = LightBlock(label="text", content="hello", bbox=[0, 0, 10, 10], score=0.9, cls_id=23)
    out = _block_to_json(block, idx=0, order=1)
    assert out["block_label"] == "text"
    assert out["block_content"] == "hello"
    assert out["block_bbox"] == [0, 0, 10, 10]
    assert out["block_id"] == 0
    assert out["block_order"] == 1
    assert out["block_polygon_points"][0] == [0.0, 0.0]
