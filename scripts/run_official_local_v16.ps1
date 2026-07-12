param(
  [string]$ServerUrl = "http://127.0.0.1:8111/v1",
  [string]$ApiModelName = "PaddleOCR-VL-1.6-GGUF.gguf",
  [string]$DatasetDir = "data/omnidocbench/v16",
  [string]$PredictionsDir = "predictions/paddleocr_official_local_llamacpp_gguf_v16",
  [int]$SmokePages = 1,
  [int]$SubsetPages = 16,
  [switch]$Full,
  [switch]$Cdm
)

$ErrorActionPreference = "Stop"

function Invoke-Native {
  param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$FilePath,
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$ArgumentList
  )

  & $FilePath @ArgumentList
  if ($LASTEXITCODE -ne 0) {
    throw "Native command failed with exit code $LASTEXITCODE`: $FilePath $($ArgumentList -join ' ')"
  }
}

function Invoke-Step {
  param([string]$Name, [scriptblock]$Body)
  Write-Host ""
  Write-Host "==> $Name"
  & $Body
}

function Assert-FullPredictionStats {
  $StatsPath = Join-Path $PredictionsDir "_run_stats.json"
  if (-not (Test-Path $StatsPath)) {
    throw "CDM scoring requires full official predictions first. Missing $StatsPath; run with -Full."
  }
  Invoke-Native python eval/release_contract.py --stats $StatsPath --version v16 --engine official
}

Invoke-Step "server gate" {
  Invoke-Native python scripts/check_server.py --server-url $ServerUrl
}

Invoke-Step "official dependency import gate" {
  Invoke-Native python scripts/check_official_paddleocr.py --server-url $ServerUrl --api-model-name $ApiModelName
}

Invoke-Step "official dependency constructor gate" {
  Invoke-Native python scripts/check_official_paddleocr.py --construct --server-url $ServerUrl --api-model-name $ApiModelName
}

if ($Cdm) {
  Invoke-Step "official full prediction stats gate for CDM" {
    Assert-FullPredictionStats
  }
  Invoke-Step "official full CDM scoring" {
    Invoke-Native python eval/run_eval.py --stage eval --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName --dataset-dir $DatasetDir --predictions-dir $PredictionsDir --cdm
  }
  return
}

Invoke-Step "official smoke gate" {
  Invoke-Native python eval/run_eval.py --stage infer --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName --dataset-dir $DatasetDir --predictions-dir $PredictionsDir --limit-pages $SmokePages
}

Invoke-Step "official subset gate" {
  Invoke-Native python eval/run_eval.py --stage infer --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName --dataset-dir $DatasetDir --predictions-dir $PredictionsDir --limit-pages $SubsetPages
}

if ($Full) {
  Invoke-Step "official full non-CDM inference" {
    Invoke-Native python eval/run_eval.py --stage infer --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName --dataset-dir $DatasetDir --predictions-dir $PredictionsDir
  }
  Invoke-Step "official full non-CDM scoring" {
    Invoke-Native python eval/run_eval.py --stage eval --version v16 --engine official --artifact-profile official-local --server-url $ServerUrl --api-model-name $ApiModelName --dataset-dir $DatasetDir --predictions-dir $PredictionsDir
  }
}
