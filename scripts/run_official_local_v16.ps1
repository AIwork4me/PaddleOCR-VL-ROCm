param(
  [string]$ServerUrl = "http://127.0.0.1:8111/v1",
  [string]$ApiModelName = "PaddleOCR-VL-1.6-GGUF.gguf",
  [int]$SmokePages = 1,
  [int]$SubsetPages = 16,
  [switch]$Full,
  [switch]$Cdm
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
  param([string]$Name, [scriptblock]$Body)
  Write-Host ""
  Write-Host "==> $Name"
  & $Body
}

Invoke-Step "server gate" {
  python scripts/check_server.py --server-url $ServerUrl
}

Invoke-Step "official dependency import gate" {
  python scripts/check_official_paddleocr.py --server-url $ServerUrl --api-model-name $ApiModelName
}

Invoke-Step "official dependency constructor gate" {
  python scripts/check_official_paddleocr.py --construct --server-url $ServerUrl --api-model-name $ApiModelName
}

Invoke-Step "official smoke gate" {
  python eval/run_eval.py --stage infer --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName --limit-pages $SmokePages
}

Invoke-Step "official subset gate" {
  python eval/run_eval.py --stage infer --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName --limit-pages $SubsetPages
}

if ($Full) {
  Invoke-Step "official full non-CDM inference" {
    python eval/run_eval.py --stage infer --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName
  }
  Invoke-Step "official full non-CDM scoring" {
    python eval/run_eval.py --stage eval --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName
  }
}

if ($Cdm) {
  Invoke-Step "official full CDM scoring" {
    python eval/run_eval.py --stage eval --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName --cdm
  }
}
