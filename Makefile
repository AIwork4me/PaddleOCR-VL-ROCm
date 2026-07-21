PLATFORM     ?= linux-rocm
VERSION      ?= v16
REVISION     ?= 2b161d0

# ── provisioning ──────────────────────────────────────────────────────────────

setup-linux:
	pip install -e .
	python -c "import paddleocr_vl_rocm; print('paddleocr-vl-rocm installed')"

setup-windows:
	pip install -e .
	python -c "import paddleocr_vl_rocm; print('paddleocr-vl-rocm installed')"

# ── demo / smoke ──────────────────────────────────────────────────────────────

demo:
	python adapter/run_adapter.py \
	  --img-dir examples --out-dir /tmp/out --platform $(PLATFORM) \
	  --backend lightweight

smoke-test:
	python -m pytest

# ── platform evaluation ───────────────────────────────────────────────────────

eval-linux:
	omnidocbench-rocm run \
	  --stage all \
	  --platform linux-rocm \
	  --version v16 \
	  --revision $(REVISION) \
	  --adapter adapter/run_adapter.py \
	  --model-id paddleocr-vl-1.6 \
	  --backend lightweight \
	  --server-url http://127.0.0.1:8111/v1 \
	  --api-model-name PaddleOCR-VL-1.6-GGUF.gguf \
	  --git-commit "$$(git rev-parse HEAD)" \
	  --results-dir results/omnidocbench/v16 \
	  --skip-existing \
	  --cdm

# ── conformance ───────────────────────────────────────────────────────────────

conformance:
	omnidocbench-rocm conformance . && echo CONFORMANT

.PHONY: setup-linux setup-windows demo smoke-test eval-linux conformance
