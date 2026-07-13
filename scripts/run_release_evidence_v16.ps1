param(
  [ValidateSet("Preflight", "Official", "Lightweight", "Decide", "All")]
  [string]$Stage = "Preflight",
  [Parameter(Mandatory = $true)] [string]$EvidenceRoot,
  [string]$ServerUrl = "http://127.0.0.1:8111/v1",
  [string]$ApiModelName = "PaddleOCR-VL-1.6-GGUF.gguf",
  [string]$DatasetDir = "data/omnidocbench/v16",
  [string]$LayoutModel = "models/PP-DocLayoutV3-onnx",
  [string]$RuntimeConfig = "$HOME/.paddleocr-vl-rocm/config.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
function Resolve-FullPath([string]$PathValue) {
  if ([IO.Path]::IsPathRooted($PathValue)) { return [IO.Path]::GetFullPath($PathValue) }
  return [IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}
$EvidenceRoot = Resolve-FullPath $EvidenceRoot
$DatasetDir = Resolve-FullPath $DatasetDir
$LayoutModel = Resolve-FullPath $LayoutModel
$RuntimeConfig = Resolve-FullPath $RuntimeConfig

function Resolve-PhysicalPath([string]$PathValue) {
  $full = [IO.Path]::GetFullPath($PathValue)
  $suffix = @()
  while (-not (Test-Path -LiteralPath $full)) {
    $suffix = @([IO.Path]::GetFileName($full)) + $suffix
    $parent = [IO.Path]::GetDirectoryName($full)
    if (-not $parent -or $parent -eq $full) { break }
    $full = $parent
  }
  if (Test-Path -LiteralPath $full) {
    $item = Get-Item -Force -LiteralPath $full
    if ($item.LinkType -and $item.Target) {
      $target = @($item.Target)[0]
      if (-not [IO.Path]::IsPathRooted($target)) { $target = Join-Path $item.Parent.FullName $target }
      $full = [IO.Path]::GetFullPath($target)
    } else { $full = $item.FullName }
  }
  foreach ($part in $suffix) { $full = Join-Path $full $part }
  return [IO.Path]::GetFullPath($full)
}

function Test-IsWithin([string]$Candidate, [string]$Parent) {
  $candidatePath = (Resolve-PhysicalPath $Candidate).TrimEnd('\', '/')
  $parentPath = (Resolve-PhysicalPath $Parent).TrimEnd('\', '/')
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
$trackedChanges = @(git status --porcelain --untracked-files=all | Where-Object {
  $_ -notmatch '^\?\? eval/\.omnidocbench/'
})
if ($LASTEXITCODE -ne 0) { throw "Unable to inspect git status --porcelain" }
if ($trackedChanges.Count -ne 0) { throw "Release evidence requires a clean worktree." }

$GitCommit = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $GitCommit) { throw "Unable to determine producing commit." }
$ManifestPath = Join-Path $EvidenceRoot "manifest.json"
$LogDir = Join-Path $EvidenceRoot "logs"
$CommandLog = Join-Path $LogDir "commands.jsonl"

function ConvertTo-SafeArgument([string]$Value) {
  if ($Value -match '^(?i)Authorization:') { return "Authorization: REDACTED" }
  if ($Value -match '^(?i)https?://') { return "<redacted-url>" }
  if ($Value -match '(?i)(api[_-]?key|token|secret)=') { return "REDACTED" }
  if ($Value -match '^([^=]+)=([A-Za-z]:[\\/]|/)') { return "$($Matches[1])=<redacted-path>" }
  if ([IO.Path]::IsPathRooted($Value)) { return "<redacted-path>" }
  return $Value
}

function Invoke-LoggedNative {
  param([string]$StageName, [string]$CommandName, [string]$FilePath, [string[]]$ArgumentList)
  New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
  $secretNext = $false
  $safe = @($ArgumentList | ForEach-Object {
    if ($secretNext) { $secretNext = $false; return "<redacted-secret>" }
    if ($_ -in @("--token", "--api-key", "--authorization")) { $secretNext = $true; return $_ }
    ConvertTo-SafeArgument $_
  })
  $started = [DateTime]::UtcNow.ToString("o")
  $timer = [Diagnostics.Stopwatch]::StartNew()
  $nativeOutput = @(& $FilePath @ArgumentList)
  $code = $LASTEXITCODE
  $timer.Stop()
  $record = [ordered]@{ timestamp_utc = $started; stage = $StageName; command_name = $CommandName; arguments = $safe; exit_code = $code; duration_ms = $timer.ElapsedMilliseconds }
  $line = ($record | ConvertTo-Json -Compress -Depth 4) + [Environment]::NewLine
  [IO.File]::AppendAllText($CommandLog, $line, [Text.UTF8Encoding]::new($false))
  if ($code -ne 0) { throw "Native command failed with exit code $code`: $FilePath" }
  return $nativeOutput
}

function Get-ImmutableInputs {
  if (-not (Test-Path -LiteralPath $DatasetDir -PathType Container)) { throw "DatasetDir does not exist: $DatasetDir" }
  if (-not (Test-Path -LiteralPath $LayoutModel -PathType Container)) { throw "LayoutModel does not exist: $LayoutModel" }
  $datasetManifest = Join-Path $DatasetDir "OmniDocBench.json"
  $layoutOnnx = Join-Path $LayoutModel "model.onnx"
  $layoutConfig = Join-Path $LayoutModel "inference.yml"
  $runtimeManifest = Join-Path $RepoRoot "src/paddleocr_vl_rocm/assets/runtime-manifest.json"
  $scoringConfig = Join-Path $RepoRoot "eval/configs/omnidocbench_v16.yaml"
  $benchmarkContract = Join-Path $RepoRoot "eval/benchmark_contract.py"
  $scorerCheckout = Join-Path $RepoRoot "eval/.omnidocbench"
  $scorerFiles = [ordered]@{
    scorer_notebook = (Join-Path $scorerCheckout "tools/generate_result_tables.ipynb")
    scorer_core_metrics = (Join-Path $scorerCheckout "src/core/metrics.py")
    scorer_cal_metric = (Join-Path $scorerCheckout "src/metrics/cal_metric.py")
    scorer_table_metric = (Join-Path $scorerCheckout "src/metrics/table_metric.py")
    scorer_cdm_metric = (Join-Path $scorerCheckout "src/metrics/cdm_metric.py")
    scorer_dataset = (Join-Path $scorerCheckout "src/dataset/end2end_dataset.py")
    scorer_windows_patch = (Join-Path $RepoRoot "eval/patches/omnidocbench-v16-windows-cdm.patch")
  }
  if (-not (Test-Path $RuntimeConfig -PathType Leaf)) { throw "Active RuntimeConfig is missing: $RuntimeConfig" }
  $config = Get-Content -Raw $RuntimeConfig | ConvertFrom-Json
  $mainGguf = [string]$config.main_gguf
  $mmproj = [string]$config.mmproj
  $runtime = Get-Content -Raw $runtimeManifest | ConvertFrom-Json
  $mainRecords = @($runtime.resources | Where-Object name -eq "paddleocr-vl-main-gguf")
  $mmprojRecords = @($runtime.resources | Where-Object name -eq "paddleocr-vl-mmproj")
  if ($mainRecords.Count -ne 1 -or $mmprojRecords.Count -ne 1) { throw "Runtime manifest model anchors are absent or ambiguous." }
  if ((Split-Path -Leaf $mainGguf) -ne (Split-Path -Leaf $mainRecords[0].destination) -or
      (Split-Path -Leaf $mmproj) -ne (Split-Path -Leaf $mmprojRecords[0].destination)) {
    throw "Active config model paths do not match the pinned runtime manifest."
  }
  foreach ($required in @($datasetManifest, $layoutOnnx, $layoutConfig, $runtimeManifest, $scoringConfig, $benchmarkContract, $mainGguf, $mmproj) + @($scorerFiles.Values)) {
    if ([string]::IsNullOrWhiteSpace($required) -or -not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required immutable input is absent or ambiguous: $required" }
  }
  $inputs = [ordered]@{
    dataset = $datasetManifest
    layout_model = $layoutOnnx
    layout_config = $layoutConfig
    main_gguf = $mainGguf
    mmproj = $mmproj
    runtime_config = $RuntimeConfig
    runtime_manifest = $runtimeManifest
    scoring_config = $scoringConfig
    benchmark_contract = $benchmarkContract
    runner = $PSCommandPath
    release_contract = (Join-Path $RepoRoot "eval/release_contract.py")
    release_evidence = (Join-Path $RepoRoot "eval/release_evidence.py")
  }
  foreach ($entry in $scorerFiles.GetEnumerator()) { $inputs[$entry.Key] = $entry.Value }
  return $inputs
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
  Invoke-LoggedNative $Stage "manifest" "python" $arguments
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
  Invoke-LoggedNative "Preflight" "scorer-contract" "python" @("eval/benchmark_contract.py", "--checkout", (Join-Path $RepoRoot "eval/.omnidocbench"))
  Invoke-LoggedNative "Preflight" "server-gate" "python" @("scripts/check_server.py", "--server-url", $ServerUrl)
  Invoke-LoggedNative "Preflight" "official-import" "python" @("scripts/check_official_paddleocr.py", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName)
  Invoke-LoggedNative "Preflight" "official-constructor" "python" @("scripts/check_official_paddleocr.py", "--construct", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName)
}

function Invoke-Official {
  Assert-OrCreateManifest
  $official = Join-Path $EvidenceRoot "official"
  $results = Join-Path $EvidenceRoot "results/official"
  New-Item -ItemType Directory -Force -Path $official, $results | Out-Null
  Invoke-LoggedNative "Official" "official-infer" "python" @("eval/run_eval.py", "--stage", "infer", "--version", "v16", "--engine", "official", "--artifact-profile", "official-local", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName, "--dataset-dir", $DatasetDir, "--predictions-dir", $official)
  $stats = Join-Path $official "_run_stats.json"
  Invoke-LoggedNative "Official" "official-contract" "python" @("eval/release_contract.py", "--stats", $stats, "--version", "v16", "--engine", "official")
  Invoke-LoggedNative "Official" "official-score" "python" @("eval/run_eval.py", "--stage", "eval", "--version", "v16", "--engine", "official", "--artifact-profile", "official-local", "--dataset-dir", $DatasetDir, "--predictions-dir", $official, "--copy-report", (Join-Path $results "metric.json"), "--run-summary", (Join-Path $results "run-summary.json"), "--provenance", (Join-Path $results "provenance.json"))
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
  Invoke-LoggedNative "Lightweight" "directml-preflight" "python" @("-m", "paddleocr_vl_rocm", "doctor", "--json", "--config", $RuntimeConfig)
  Invoke-LoggedNative "Lightweight" "lightweight-infer" "python" @("eval/run_eval.py", "--stage", "infer", "--version", "v16", "--engine", "lightweight", "--dataset-dir", $DatasetDir, "--predictions-dir", $lightweight, "--layout-model", $LayoutModel)
  $stats = Join-Path $lightweight "_run_stats.json"
  Assert-DirectMlEvidence $stats
  Invoke-LoggedNative "Lightweight" "lightweight-score" "python" @("eval/run_eval.py", "--stage", "eval", "--version", "v16", "--engine", "lightweight", "--dataset-dir", $DatasetDir, "--predictions-dir", $lightweight, "--copy-report", (Join-Path $results "metric.json"), "--run-summary", (Join-Path $results "run-summary.json"), "--provenance", (Join-Path $results "provenance.json"))
}

function Invoke-Decide {
  Assert-OrCreateManifest
  $decision = @(Invoke-LoggedNative "Decide" "release-decision" "python" @("eval/release_evidence.py", "decide", "--evidence-root", $EvidenceRoot)) -join [Environment]::NewLine
  $temporary = Join-Path $EvidenceRoot "decision.json.tmp"
  [IO.File]::WriteAllText($temporary, $decision + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
  Move-Item -Force $temporary (Join-Path $EvidenceRoot "decision.json")
}

function Get-Sha256([string]$Path) {
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Write-AtomicJson([string]$Path, [object]$Value) {
  $temporary = "$Path.tmp"
  [IO.File]::WriteAllText($temporary, (($Value | ConvertTo-Json -Depth 8) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
  Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Get-StageOutputs([string]$StageName) {
  $roots = switch ($StageName) {
    "Preflight" { @($ManifestPath) }
    "Official" { @((Join-Path $EvidenceRoot "official"), (Join-Path $EvidenceRoot "results/official")) }
    "Lightweight" { @((Join-Path $EvidenceRoot "lightweight"), (Join-Path $EvidenceRoot "results/lightweight")) }
    "Decide" { @((Join-Path $EvidenceRoot "decision.json")) }
  }
  $files = foreach ($root in $roots) {
    if (Test-Path -LiteralPath $root -PathType Leaf) { Get-Item -LiteralPath $root }
    elseif (Test-Path -LiteralPath $root -PathType Container) { Get-ChildItem -LiteralPath $root -File -Recurse | Sort-Object FullName }
  }
  $records = [ordered]@{}
  foreach ($file in $files) {
    $relative = $file.FullName.Substring($EvidenceRoot.Length).TrimStart('\', '/') -replace '\\', '/'
    $records[$relative] = Get-Sha256 $file.FullName
  }
  return $records
}

function Clear-StageOutputs([string]$StageName) {
  $paths = switch ($StageName) {
    "Preflight" { @() }
    "Official" { @((Join-Path $EvidenceRoot "official"), (Join-Path $EvidenceRoot "results/official")) }
    "Lightweight" { @((Join-Path $EvidenceRoot "lightweight"), (Join-Path $EvidenceRoot "results/lightweight")) }
    "Decide" { @((Join-Path $EvidenceRoot "decision.json"), (Join-Path $EvidenceRoot "decision.json.tmp")) }
  }
  foreach ($path in $paths) { if (Test-Path -LiteralPath $path) { Remove-Item -Force -Recurse -LiteralPath $path } }
}

function Test-OutputHashes([object]$Expected) {
  foreach ($property in $Expected.PSObject.Properties) {
    $path = Join-Path $EvidenceRoot ($property.Name -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Sha256 $path) -ne $property.Value) { return $false }
  }
  return $Expected.PSObject.Properties.Count -gt 0
}

function Invoke-DurableStage([string]$StageName, [scriptblock]$Body) {
  Assert-OrCreateManifest
  $manifestSha = Get-Sha256 $ManifestPath
  $stateDir = Join-Path $LogDir "stages"
  New-Item -ItemType Directory -Force $stateDir | Out-Null
  $statePath = Join-Path $stateDir "$($StageName.ToLowerInvariant()).json"
  if (Test-Path -LiteralPath $statePath) {
    $prior = Get-Content -Raw $statePath | ConvertFrom-Json
    if ($prior.producing_commit -ne $GitCommit -or $prior.input_manifest_sha256 -ne $manifestSha) { throw "Stage state commit/input mismatch: $StageName" }
    if ($prior.status -eq "completed") {
      if (-not (Test-OutputHashes $prior.output_sha256)) { throw "Completed stage output hash mismatch: $StageName" }
      Write-Host "Skipping completed stage $StageName"
      return
    }
    Clear-StageOutputs $StageName
  } else {
    Clear-StageOutputs $StageName
  }
  $started = [ordered]@{ stage = $StageName; status = "started"; producing_commit = $GitCommit; input_manifest_sha256 = $manifestSha; command_sha256 = $null; output_sha256 = @{} }
  Write-AtomicJson $statePath $started
  try {
    & $Body
    $outputs = Get-StageOutputs $StageName
    if ($outputs.Count -eq 0) { throw "Stage produced no durable artifacts: $StageName" }
    $completed = [ordered]@{ stage = $StageName; status = "completed"; producing_commit = $GitCommit; input_manifest_sha256 = $manifestSha; command_sha256 = (Get-Sha256 $CommandLog); output_sha256 = $outputs }
    Write-AtomicJson $statePath $completed
  } catch {
    $failed = [ordered]@{ stage = $StageName; status = "failed"; producing_commit = $GitCommit; input_manifest_sha256 = $manifestSha; command_sha256 = $(if (Test-Path $CommandLog) { Get-Sha256 $CommandLog } else { $null }); output_sha256 = (Get-StageOutputs $StageName); error = $_.Exception.Message }
    Write-AtomicJson $statePath $failed
    throw
  }
}

try {
  switch ($Stage) {
    "Preflight" { Invoke-DurableStage "Preflight" { Invoke-Preflight } }
    "Official" { Invoke-DurableStage "Official" { Invoke-Official } }
    "Lightweight" { Invoke-DurableStage "Lightweight" { Invoke-Lightweight } }
    "Decide" { Invoke-DurableStage "Decide" { Invoke-Decide } }
    "All" {
      Invoke-DurableStage "Preflight" { Invoke-Preflight }
      Invoke-DurableStage "Official" { Invoke-Official }
      Invoke-DurableStage "Lightweight" { Invoke-Lightweight }
      Invoke-DurableStage "Decide" { Invoke-Decide }
      if (-not (Test-Path (Join-Path $EvidenceRoot "decision.json") -PathType Leaf)) { throw "All requires durable decision.json" }
    }
  }
} finally {
  Pop-Location
}
