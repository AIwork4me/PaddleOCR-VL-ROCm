param(
  [ValidateSet("Preflight", "Official", "Lightweight", "Decide", "All")]
  [string]$Stage = "Preflight",
  [Parameter(Mandatory = $true)] [string]$EvidenceRoot,
  [string]$ServerUrl = "http://127.0.0.1:8111/v1",
  [string]$ApiModelName = "PaddleOCR-VL-1.6-GGUF.gguf",
  [string]$DatasetDir = "data/omnidocbench/v16",
  [string]$LayoutModel = "models/PP-DocLayoutV3-onnx"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-FullPath([string]$PathValue) {
  if ([IO.Path]::IsPathRooted($PathValue)) { return [IO.Path]::GetFullPath($PathValue) }
  return [IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}
$EvidenceRoot = Resolve-FullPath $EvidenceRoot
$DatasetDir = Resolve-FullPath $DatasetDir
$LayoutModel = Resolve-FullPath $LayoutModel

function Test-IsWithin([string]$Candidate, [string]$Parent) {
  $candidatePath = [IO.Path]::GetFullPath($Candidate).TrimEnd('\', '/')
  $parentPath = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
  return $candidatePath.Equals($parentPath, [StringComparison]::OrdinalIgnoreCase) -or
    $candidatePath.StartsWith($parentPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

$ProtectedPaths = @(
  (Join-Path $RepoRoot "results/omnidocbench/v16"),
  (Join-Path $RepoRoot "predictions/paddleocr_official_local_llamacpp_gguf_v16"),
  (Join-Path $RepoRoot "predictions/paddleocrvl_rocm_cdm")
)
foreach ($protected in $ProtectedPaths) {
  if (Test-IsWithin $EvidenceRoot $protected) {
    throw "EvidenceRoot is inside a protected historical path: $protected"
  }
}

# A release run must begin from a clean tracked checkout. eval/.omnidocbench is
# deliberately ignored because it is an external scorer checkout.
if ($env:RELEASE_EVIDENCE_ALLOW_DIRTY -ne "1") {
  $trackedChanges = @(git status --porcelain --untracked-files=all | Where-Object {
    $_ -notmatch '^\?\? eval/\.omnidocbench/'
  })
  if ($LASTEXITCODE -ne 0) { throw "Unable to inspect git status --porcelain" }
  if ($trackedChanges.Count -ne 0) { throw "Release evidence requires a clean tracked worktree." }
}

$GitCommit = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $GitCommit) { throw "Unable to determine producing commit." }
$ManifestPath = Join-Path $EvidenceRoot "manifest.json"
$LogDir = Join-Path $EvidenceRoot "logs"
$CommandLog = Join-Path $LogDir "commands.jsonl"

function ConvertTo-SafeArgument([string]$Value) {
  if ($Value -match '^(?i)Authorization:') { return "Authorization: REDACTED" }
  if ($Value -match '^(?i)https?://') { return "REDACTED_URL" }
  if ($Value -match '(?i)(api[_-]?key|token|secret)=') { return "REDACTED" }
  return $Value
}

function Invoke-LoggedNative {
  param([string]$FilePath, [string[]]$ArgumentList)
  New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
  $safe = @($ArgumentList | ForEach-Object { ConvertTo-SafeArgument $_ })
  $started = [DateTime]::UtcNow.ToString("o")
  & $FilePath @ArgumentList
  $code = $LASTEXITCODE
  $record = [ordered]@{ timestamp_utc = $started; command = $FilePath; arguments = $safe; exit_code = $code }
  $line = ($record | ConvertTo-Json -Compress -Depth 4) + [Environment]::NewLine
  [IO.File]::AppendAllText($CommandLog, $line, [Text.UTF8Encoding]::new($false))
  if ($code -ne 0) { throw "Native command failed with exit code $code`: $FilePath" }
}

function Get-ImmutableInputs {
  if (-not (Test-Path -LiteralPath $DatasetDir -PathType Container)) { throw "DatasetDir does not exist: $DatasetDir" }
  if (-not (Test-Path -LiteralPath $LayoutModel -PathType Container)) { throw "LayoutModel does not exist: $LayoutModel" }
  $datasetManifest = Get-ChildItem -LiteralPath $DatasetDir -File -Recurse | Where-Object Extension -eq ".json" | Sort-Object FullName | Select-Object -First 1
  $layoutOnnx = Get-ChildItem -LiteralPath $LayoutModel -File -Recurse | Where-Object Extension -eq ".onnx" | Sort-Object FullName | Select-Object -First 1
  if (-not $datasetManifest) { throw "DatasetDir has no JSON manifest." }
  if (-not $layoutOnnx) { throw "LayoutModel has no ONNX file." }
  return [ordered]@{
    dataset = $datasetManifest.FullName
    layout_model = $layoutOnnx.FullName
    runner = $PSCommandPath
    release_contract = (Join-Path $RepoRoot "eval/release_contract.py")
    release_evidence = (Join-Path $RepoRoot "eval/release_evidence.py")
  }
}

function Assert-OrCreateManifest {
  New-Item -ItemType Directory -Force -Path $EvidenceRoot, $LogDir | Out-Null
  $candidate = Join-Path $LogDir "manifest.candidate.json"
  $arguments = @("eval/release_evidence.py", "manifest", "--git-commit", $GitCommit)
  foreach ($entry in (Get-ImmutableInputs).GetEnumerator()) {
    $arguments += @("--input", "$($entry.Key)=$($entry.Value)")
  }
  $arguments += @("--output", $candidate)
  # Contract spelling retained for audit/search: eval/release_evidence.py manifest
  Invoke-LoggedNative "python" $arguments
  if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "Manifest command did not create $candidate" }
  if (Test-Path -LiteralPath $ManifestPath) {
    if ((Get-Content -Raw -LiteralPath $ManifestPath) -cne (Get-Content -Raw -LiteralPath $candidate)) {
      throw "Resume refused: input sha256 values or producing git_commit differ from manifest.json."
    }
    Remove-Item -LiteralPath $candidate
  } else {
    Move-Item -LiteralPath $candidate -Destination $ManifestPath
  }
}

function Invoke-Preflight {
  Assert-OrCreateManifest
  Invoke-LoggedNative "python" @("scripts/check_server.py", "--server-url", $ServerUrl)
  Invoke-LoggedNative "python" @("scripts/check_official_paddleocr.py", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName)
  Invoke-LoggedNative "python" @("scripts/check_official_paddleocr.py", "--construct", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName)
}

function Invoke-Official {
  Assert-OrCreateManifest
  $official = Join-Path $EvidenceRoot "official"
  $results = Join-Path $EvidenceRoot "results/official"
  New-Item -ItemType Directory -Force -Path $official, $results | Out-Null
  Invoke-LoggedNative "python" @("eval/run_eval.py", "--stage", "infer", "--version", "v16", "--engine", "official", "--artifact-profile", "official-local", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName, "--dataset-dir", $DatasetDir, "--predictions-dir", $official)
  $stats = Join-Path $official "_run_stats.json"
  Invoke-LoggedNative "python" @("eval/release_contract.py", "--stats", $stats, "--version", "v16", "--engine", "official")
  Invoke-LoggedNative "python" @("eval/run_eval.py", "--stage", "eval", "--version", "v16", "--engine", "official", "--artifact-profile", "official-local", "--dataset-dir", $DatasetDir, "--predictions-dir", $official, "--copy-report", (Join-Path $results "metric.json"), "--run-summary", (Join-Path $results "run-summary.json"), "--provenance", (Join-Path $results "provenance.json"))
}

function Assert-DirectMlEvidence([string]$StatsPath) {
  $stats = Get-Content -Raw -LiteralPath $StatsPath | ConvertFrom-Json
  $providers = @($stats.layout_providers_active)
  if ($stats.layout_provider_requested -ne "auto" -or
      $providers.Count -lt 2 -or $providers[0] -ne "DmlExecutionProvider" -or
      $providers[1] -ne "CPUExecutionProvider" -or
      $stats.layout_fallback_disabled -ne $true) {
    throw "Lightweight evidence violates DirectML-first/fallback-disabled contract."
  }
}

function Invoke-Lightweight {
  Assert-OrCreateManifest
  $lightweight = Join-Path $EvidenceRoot "lightweight"
  $results = Join-Path $EvidenceRoot "results/lightweight"
  New-Item -ItemType Directory -Force -Path $lightweight, $results | Out-Null
  Invoke-LoggedNative "python" @("eval/run_eval.py", "--stage", "infer", "--version", "v16", "--engine", "lightweight", "--dataset-dir", $DatasetDir, "--predictions-dir", $lightweight, "--layout-model", $LayoutModel)
  $stats = Join-Path $lightweight "_run_stats.json"
  Assert-DirectMlEvidence $stats
  Invoke-LoggedNative "python" @("eval/run_eval.py", "--stage", "eval", "--version", "v16", "--engine", "lightweight", "--dataset-dir", $DatasetDir, "--predictions-dir", $lightweight, "--copy-report", (Join-Path $results "metric.json"), "--run-summary", (Join-Path $results "run-summary.json"), "--provenance", (Join-Path $results "provenance.json"))
}

function Invoke-Decide {
  Assert-OrCreateManifest
  Invoke-LoggedNative "python" @("eval/release_evidence.py", "decide", "--evidence-root", $EvidenceRoot)
}

Push-Location $RepoRoot
try {
  switch ($Stage) {
    "Preflight" { Invoke-Preflight }
    "Official" { Invoke-Official }
    "Lightweight" { Invoke-Lightweight }
    "Decide" { Invoke-Decide }
    "All" { Invoke-Preflight; Invoke-Official; Invoke-Lightweight; Invoke-Decide }
  }
} finally {
  Pop-Location
}
