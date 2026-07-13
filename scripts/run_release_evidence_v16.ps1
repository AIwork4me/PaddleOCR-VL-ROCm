param(
  [ValidateSet("Preflight", "Official", "OfficialScore", "Lightweight", "Decide", "All")]
  [string]$Stage = "Preflight",
  [Parameter(Mandatory = $true)] [string]$EvidenceRoot,
  [string]$ServerUrl = "http://127.0.0.1:8111/v1",
  [string]$ApiModelName = "PaddleOCR-VL-1.6-GGUF.gguf",
  [string]$DatasetDir = "data/omnidocbench/v16",
  [string]$LayoutModel = "models/PP-DocLayoutV3-onnx",
  [string]$RuntimeConfig = "$HOME/.paddleocr-vl-rocm/config.json",
  [string]$PythonExe,
  [string]$ScorerPythonExe,
  [string]$RecoverySourceRoot
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
if (-not [string]::IsNullOrWhiteSpace($RecoverySourceRoot)) {
  $RecoverySourceRoot = Resolve-FullPath $RecoverySourceRoot
}

function Resolve-PhysicalPath([string]$PathValue) {
  $full = [IO.Path]::GetFullPath($PathValue)
  $root = [IO.Path]::GetPathRoot($full)
  $current = $root
  $relative = $full.Substring($root.Length)
  foreach ($part in ($relative -split '[\\/]' | Where-Object { $_ })) {
    $candidate = Join-Path $current $part
    if (Test-Path -LiteralPath $candidate) {
      $item = Get-Item -Force -LiteralPath $candidate
      if ($item.LinkType -and $item.Target) {
        $target = @($item.Target)[0]
        if (-not [IO.Path]::IsPathRooted($target)) { $target = Join-Path $item.Parent.FullName $target }
        $current = [IO.Path]::GetFullPath($target)
      } else { $current = $item.FullName }
    } else { $current = $candidate }
  }
  return [IO.Path]::GetFullPath($current)
}

function Test-IsWithin([string]$Candidate, [string]$Parent) {
  $candidatePath = (Resolve-PhysicalPath $Candidate).TrimEnd('\', '/')
  $parentPath = (Resolve-PhysicalPath $Parent).TrimEnd('\', '/')
  return $candidatePath.Equals($parentPath, [StringComparison]::OrdinalIgnoreCase) -or
    $candidatePath.StartsWith($parentPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  $PythonExe = if ($IsWindows -or $env:OS -eq "Windows_NT") { Join-Path $RepoRoot ".venv/Scripts/python.exe" } else { Join-Path $RepoRoot ".venv/bin/python" }
}
if (-not [IO.Path]::IsPathRooted($PythonExe)) { throw "PythonExe must be an absolute path." }
$PythonExe = Resolve-PhysicalPath $PythonExe
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "PythonExe must be an existing regular executable file." }
if ([string]::IsNullOrWhiteSpace($ScorerPythonExe)) {
  $ScorerPythonExe = if ($IsWindows -or $env:OS -eq "Windows_NT") { Join-Path $RepoRoot ".scorer-venv/Scripts/python.exe" } else { Join-Path $RepoRoot ".scorer-venv/bin/python" }
}
if (-not [IO.Path]::IsPathRooted($ScorerPythonExe)) { throw "ScorerPythonExe must be an absolute path." }
$ScorerPythonExe = Resolve-PhysicalPath $ScorerPythonExe

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
if (-not [string]::IsNullOrWhiteSpace($RecoverySourceRoot) -and
    ((Test-IsWithin $EvidenceRoot $RecoverySourceRoot) -or
     (Test-IsWithin $RecoverySourceRoot $EvidenceRoot))) {
  throw "EvidenceRoot and RecoverySourceRoot must not overlap."
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
  if ($Value -match '^([A-Za-z]:[\\/]|/)') { return "<redacted-path>" }
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
  $stageCommandLog = Join-Path (Join-Path $LogDir "stages") "$($StageName.ToLowerInvariant()).commands.jsonl"
  New-Item -ItemType Directory -Force (Split-Path -Parent $stageCommandLog) | Out-Null
  [IO.File]::AppendAllText($stageCommandLog, $line, [Text.UTF8Encoding]::new($false))
  if ($code -ne 0) { throw "Native command failed with exit code $code`: $FilePath" }
  return $nativeOutput
}

function Get-PythonProvenance {
  if ($script:PythonProvenance) { return $script:PythonProvenance }
  $probe = 'import hashlib,importlib.metadata as metadata,json,sys; from pathlib import Path; import eval.release_evidence as release_evidence; import paddleocr,paddleocr_vl_rocm as package,paddlex,paddle; modules=dict(paddleocr=paddleocr,paddlex=paddlex,paddlepaddle=paddle); distributions={name:metadata.distribution(name) for name in modules}; records={name:next(Path(dist.locate_file(path)).resolve() for path in dist.files if Path(path).name==chr(82)+chr(69)+chr(67)+chr(79)+chr(82)+chr(68)) for name,dist in distributions.items()}; installed=sorted((dist.metadata.get(chr(78)+chr(97)+chr(109)+chr(101)) or dist.name).lower()+chr(61)+chr(61)+dist.version for dist in metadata.distributions()); print(json.dumps(dict(version=sys.version,executable=sys.executable,eval_origin=str(Path(release_evidence.__file__).resolve()),package_origin=str(Path(package.__file__).resolve()),core_versions={name:dist.version for name,dist in distributions.items()},core_origins={name:str(Path(module.__file__).resolve()) for name,module in modules.items()},distribution_origins={name:str(Path(dist.locate_file(chr(46))).resolve()) for name,dist in distributions.items()},record_paths={name:str(path) for name,path in records.items()},record_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in records.items()},dependency_environment_sha256=hashlib.sha256(chr(10).join(installed).encode()).hexdigest())))'
  $rendered = @(Invoke-LoggedNative "Manifest" "python-origin-probe" $PythonExe @("-c", $probe)) -join [Environment]::NewLine
  try { $value = $rendered | ConvertFrom-Json } catch { throw "Python module-origin probe returned invalid JSON." }
  if ((Resolve-PhysicalPath ([string]$value.executable)) -ne $PythonExe) { throw "Python origin probe executable does not match PythonExe." }
  if (-not (Test-IsWithin ([string]$value.eval_origin) (Join-Path $RepoRoot "eval")) -or
      -not (Test-IsWithin ([string]$value.package_origin) (Join-Path $RepoRoot "src"))) {
    throw "Python module origins are outside this worktree."
  }
  $venvRoot = Split-Path -Parent (Split-Path -Parent $PythonExe)
  $expectedVersions = [ordered]@{ paddleocr = "3.7.0"; paddlex = "3.7.2"; paddlepaddle = "3.2.1" }
  foreach ($entry in $expectedVersions.GetEnumerator()) {
    if ([string]$value.core_versions.($entry.Key) -ne $entry.Value) { throw "Required dependency version mismatch: $($entry.Key)==$($entry.Value)" }
    if (-not (Test-IsWithin ([string]$value.core_origins.($entry.Key)) $venvRoot) -or
        -not (Test-IsWithin ([string]$value.distribution_origins.($entry.Key)) $venvRoot) -or
        -not (Test-IsWithin ([string]$value.record_paths.($entry.Key)) $venvRoot)) {
      throw "Dependency origin is outside the pinned worktree venv: $($entry.Key)"
    }
    if ([string]$value.record_sha256.($entry.Key) -notmatch '^[0-9a-f]{64}$') { throw "Dependency RECORD hash is invalid: $($entry.Key)" }
  }
  if ([string]$value.dependency_environment_sha256 -notmatch '^[0-9a-f]{64}$') { throw "Dependency environment hash is invalid." }
  $script:PythonProvenance = [ordered]@{
    path_sha256 = Get-StringSha256 $PythonExe
    file_sha256 = Get-Sha256 $PythonExe
    version_sha256 = Get-StringSha256 ([string]$value.version)
    executable_sha256 = Get-StringSha256 (Resolve-PhysicalPath ([string]$value.executable))
    eval_origin_sha256 = Get-StringSha256 (Resolve-PhysicalPath ([string]$value.eval_origin))
    package_origin_sha256 = Get-StringSha256 (Resolve-PhysicalPath ([string]$value.package_origin))
    dependency_environment_sha256 = [string]$value.dependency_environment_sha256
    paddleocr_record_sha256 = [string]$value.record_sha256.paddleocr
    paddlex_record_sha256 = [string]$value.record_sha256.paddlex
    paddlepaddle_record_sha256 = [string]$value.record_sha256.paddlepaddle
    record_paths = $value.record_paths
  }
  return $script:PythonProvenance
}

function Get-ScorerProvenance {
  if ($script:ScorerProvenance) { return $script:ScorerProvenance }
  if (-not (Test-Path -LiteralPath $ScorerPythonExe -PathType Leaf)) { throw "ScorerPythonExe must be an existing regular executable file." }
  New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
  $attestation = Join-Path $LogDir "scorer-environment.json"
  $candidate = Join-Path $LogDir "scorer-environment.candidate.json"
  $arguments = @(
    "scripts/check_omnidocbench_scorer.py",
    "--checkout", (Join-Path $RepoRoot "eval/.omnidocbench"),
    "--direct-lock", (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16.txt"),
    "--transitive-lock", (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16-transitive.txt"),
    "--attest-only", "--output", $candidate
  )
  Invoke-LoggedNative "Manifest" "scorer-environment" $ScorerPythonExe $arguments | Out-Null
  if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "Scorer attestation did not create $candidate" }
  if (Test-Path -LiteralPath $attestation -PathType Leaf) {
    if ((Get-Content -Raw -LiteralPath $attestation) -cne (Get-Content -Raw -LiteralPath $candidate)) {
      throw "Resume refused: scorer interpreter, origin, RECORD, or package content differs."
    }
    Remove-Item -LiteralPath $candidate
  } else {
    Move-Item -LiteralPath $candidate -Destination $attestation
  }
  try { $value = Get-Content -Raw -LiteralPath $attestation | ConvertFrom-Json }
  catch { throw "Scorer attestation returned invalid JSON." }
  foreach ($key in @("python_executable_sha256", "python_version_sha256", "dependency_environment_sha256")) {
    if ([string]$value.$key -notmatch '^[0-9a-f]{64}$') { throw "Scorer attestation hash is invalid: $key" }
  }
  $script:ScorerProvenance = [ordered]@{
    path_sha256 = Get-StringSha256 $ScorerPythonExe
    file_sha256 = Get-Sha256 $ScorerPythonExe
    environment_sha256 = [string]$value.dependency_environment_sha256
    attestation = $attestation
  }
  return $script:ScorerProvenance
}

function Get-ImmutableInputs {
  $pythonProvenance = Get-PythonProvenance
  $scorerProvenance = Get-ScorerProvenance
  if (-not (Test-Path -LiteralPath $DatasetDir -PathType Container)) { throw "DatasetDir does not exist: $DatasetDir" }
  if (-not (Test-Path -LiteralPath $LayoutModel -PathType Container)) { throw "LayoutModel does not exist: $LayoutModel" }
  $datasetManifest = Join-Path $DatasetDir "OmniDocBench.json"
  $layoutOnnx = Join-Path $LayoutModel "inference.onnx"
  $layoutConfig = Join-Path $LayoutModel "inference.yml"
  $runtimeManifest = Join-Path $RepoRoot "src/paddleocr_vl_rocm/assets/runtime-manifest.json"
  $scoringConfig = Join-Path $RepoRoot "eval/configs/omnidocbench_v16.yaml"
  $benchmarkContract = Join-Path $RepoRoot "eval/benchmark_contract.py"
  $scorerRequirements = Join-Path $RepoRoot "eval/requirements-omnidocbench-v16.txt"
  $scorerTransitiveRequirements = Join-Path $RepoRoot "eval/requirements-omnidocbench-v16-transitive.txt"
  $scorerPreflight = Join-Path $RepoRoot "scripts/check_omnidocbench_scorer.py"
  $scoreRecovery = Join-Path $RepoRoot "eval/score_recovery.py"
  $scorerCheckout = Join-Path $RepoRoot "eval/.omnidocbench"
  $scorerFiles = [ordered]@{
    scorer_notebook = (Join-Path $scorerCheckout "tools/generate_result_tables.ipynb")
    scorer_pyproject = (Join-Path $scorerCheckout "pyproject.toml")
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
  $configuredLayout = [string]$config.layout_model_dir
  if ([string]::IsNullOrWhiteSpace($configuredLayout) -or
      (Resolve-PhysicalPath $configuredLayout) -ne (Resolve-PhysicalPath $LayoutModel)) {
    throw "Active RuntimeConfig layout_model_dir does not match LayoutModel."
  }
  $runtime = Get-Content -Raw $runtimeManifest | ConvertFrom-Json
  $mainRecords = @($runtime.resources | Where-Object name -eq "paddleocr-vl-main-gguf")
  $mmprojRecords = @($runtime.resources | Where-Object name -eq "paddleocr-vl-mmproj")
  if ($mainRecords.Count -ne 1 -or $mmprojRecords.Count -ne 1) { throw "Runtime manifest model anchors are absent or ambiguous." }
  if ((Split-Path -Leaf $mainGguf) -ne (Split-Path -Leaf $mainRecords[0].destination) -or
      (Split-Path -Leaf $mmproj) -ne (Split-Path -Leaf $mmprojRecords[0].destination)) {
    throw "Active config model paths do not match the pinned runtime manifest."
  }
  $requiredInputs = @($datasetManifest, $scoringConfig, $benchmarkContract, $scorerRequirements, $scorerTransitiveRequirements, $scorerPreflight, $scoreRecovery, $scorerProvenance.attestation) + @($scorerFiles.Values)
  $requiredInputs += @($layoutOnnx, $layoutConfig, $runtimeManifest, $mainGguf, $mmproj, $RuntimeConfig)
  foreach ($required in $requiredInputs) {
    if ([string]::IsNullOrWhiteSpace($required) -or -not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required immutable input is absent or ambiguous: $required" }
  }
  $inputs = [ordered]@{
    dataset = $datasetManifest
    scoring_config = $scoringConfig
    benchmark_contract = $benchmarkContract
    scorer_requirements = $scorerRequirements
    scorer_transitive_requirements = $scorerTransitiveRequirements
    scorer_environment = $scorerProvenance.attestation
    scorer_preflight = $scorerPreflight
    score_recovery = $scoreRecovery
    python_executable = $PythonExe
    scorer_python_executable = $ScorerPythonExe
    paddleocr_record = [string]$pythonProvenance.record_paths.paddleocr
    paddlex_record = [string]$pythonProvenance.record_paths.paddlex
    paddlepaddle_record = [string]$pythonProvenance.record_paths.paddlepaddle
    runner = $PSCommandPath
    release_contract = (Join-Path $RepoRoot "eval/release_contract.py")
    release_evidence = (Join-Path $RepoRoot "eval/release_evidence.py")
  }
  $inputs["layout_model"] = $layoutOnnx
  $inputs["layout_config"] = $layoutConfig
  $inputs["main_gguf"] = $mainGguf
  $inputs["mmproj"] = $mmproj
  $inputs["runtime_config"] = $RuntimeConfig
  $inputs["runtime_manifest"] = $runtimeManifest
  foreach ($entry in $scorerFiles.GetEnumerator()) { $inputs[$entry.Key] = $entry.Value }
  if (-not [string]::IsNullOrWhiteSpace($RecoverySourceRoot)) {
    $sourceInputs = [ordered]@{
      recovery_source_manifest = (Join-Path $RecoverySourceRoot "manifest.json")
      recovery_source_state = (Join-Path $RecoverySourceRoot "logs/stages/official.json")
      recovery_source_commands = (Join-Path $RecoverySourceRoot "logs/stages/official.commands.jsonl")
      recovery_source_stats = (Join-Path $RecoverySourceRoot "official/_run_stats.json")
      recovery_source_errors = (Join-Path $RecoverySourceRoot "official/_errors.log")
    }
    foreach ($entry in $sourceInputs.GetEnumerator()) {
      if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) { throw "Required recovery source input is missing: $($entry.Key)" }
      $inputs[$entry.Key] = $entry.Value
    }
  }
  return $inputs
}

function Assert-OrCreateManifest {
  New-Item -ItemType Directory -Force -Path $EvidenceRoot, $LogDir | Out-Null
  $candidate = Join-Path $LogDir "manifest.candidate.json"
  $arguments = @("-m", "eval.release_evidence", "manifest", "--git-commit", $GitCommit)
  foreach ($entry in (Get-ImmutableInputs).GetEnumerator()) {
    $arguments += @("--input", "$($entry.Key)=$($entry.Value)")
  }
  $arguments += @("--output", $candidate)
  # Package-module execution keeps repository imports stable without PYTHONPATH.
  Invoke-LoggedNative "Manifest" "manifest" $PythonExe $arguments
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
  Invoke-LoggedNative "Preflight" "scorer-contract" $PythonExe @("-m", "eval.benchmark_contract", "--checkout", (Join-Path $RepoRoot "eval/.omnidocbench"))
  Invoke-LoggedNative "Preflight" "scorer-preflight" $ScorerPythonExe @("scripts/check_omnidocbench_scorer.py", "--checkout", (Join-Path $RepoRoot "eval/.omnidocbench"), "--direct-lock", (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16.txt"), "--transitive-lock", (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16-transitive.txt"), "--require-cdm-tools")
  if (-not [string]::IsNullOrWhiteSpace($RecoverySourceRoot)) { return }
  Invoke-LoggedNative "Preflight" "server-gate" $PythonExe @("scripts/check_server.py", "--server-url", $ServerUrl)
  Invoke-LoggedNative "Preflight" "official-import" $PythonExe @("scripts/check_official_paddleocr.py", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName)
  Invoke-LoggedNative "Preflight" "official-constructor" $PythonExe @("scripts/check_official_paddleocr.py", "--construct", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName)
}

function Invoke-OfficialInference {
  $official = Join-Path $EvidenceRoot "official"
  New-Item -ItemType Directory -Force -Path $official | Out-Null
  Invoke-LoggedNative "OfficialInference" "official-infer" $PythonExe @("-m", "eval.run_eval", "--stage", "infer", "--version", "v16", "--engine", "official", "--artifact-profile", "official-local", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName, "--dataset-dir", $DatasetDir, "--predictions-dir", $official)
  $stats = Join-Path $official "_run_stats.json"
  Invoke-LoggedNative "OfficialInference" "official-contract" $PythonExe @("-m", "eval.release_contract", "--stats", $stats, "--version", "v16", "--engine", "official")
}

function Invoke-RecoveryAuthentication([string]$StageName) {
  $recovery = Join-Path $EvidenceRoot "recovery"
  New-Item -ItemType Directory -Force -Path $recovery | Out-Null
  Invoke-LoggedNative $StageName "score-recovery" $PythonExe @("-m", "eval.score_recovery", "--source-root", $RecoverySourceRoot, "--output", (Join-Path $recovery "source.json"))
}

function Invoke-OfficialScore {
  $official = if ([string]::IsNullOrWhiteSpace($RecoverySourceRoot)) { Join-Path $EvidenceRoot "official" } else { Join-Path $RecoverySourceRoot "official" }
  $results = Join-Path $EvidenceRoot "results/official"
  New-Item -ItemType Directory -Force -Path $results | Out-Null
  if (-not [string]::IsNullOrWhiteSpace($RecoverySourceRoot)) { Invoke-RecoveryAuthentication "Official" }
  Invoke-LoggedNative "Official" "official-score" $PythonExe @("-m", "eval.run_eval", "--stage", "eval", "--version", "v16", "--engine", "official", "--artifact-profile", "official-local", "--dataset-dir", $DatasetDir, "--predictions-dir", $official, "--scorer-python", $ScorerPythonExe, "--copy-report", (Join-Path $results "metric.json"), "--run-summary", (Join-Path $results "run-summary.json"), "--provenance", (Join-Path $results "provenance.json"))
  Invoke-LoggedNative "Official" "official-score-cdm" $PythonExe @("-m", "eval.run_eval", "--stage", "eval", "--version", "v16", "--engine", "official", "--artifact-profile", "official-local", "--dataset-dir", $DatasetDir, "--predictions-dir", $official, "--scorer-python", $ScorerPythonExe, "--cdm", "--copy-report", (Join-Path $results "metric-cdm.json"), "--run-summary", (Join-Path $results "run-summary-cdm.json"), "--provenance", (Join-Path $results "provenance-cdm.json"))
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
  $lightweight = Join-Path $EvidenceRoot "lightweight"
  $results = Join-Path $EvidenceRoot "results/lightweight"
  New-Item -ItemType Directory -Force -Path $lightweight, $results | Out-Null
  Invoke-LoggedNative "Lightweight" "directml-preflight" $PythonExe @("-m", "paddleocr_vl_rocm", "doctor", "--json", "--config", $RuntimeConfig)
  Invoke-LoggedNative "Lightweight" "lightweight-infer" $PythonExe @("-m", "eval.run_eval", "--stage", "infer", "--version", "v16", "--engine", "lightweight", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName, "--dataset-dir", $DatasetDir, "--predictions-dir", $lightweight, "--layout-model", $LayoutModel)
  $stats = Join-Path $lightweight "_run_stats.json"
  Assert-DirectMlEvidence $stats
  Invoke-LoggedNative "Lightweight" "lightweight-score" $PythonExe @("-m", "eval.run_eval", "--stage", "eval", "--version", "v16", "--engine", "lightweight", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName, "--dataset-dir", $DatasetDir, "--predictions-dir", $lightweight, "--layout-model", $LayoutModel, "--scorer-python", $ScorerPythonExe, "--copy-report", (Join-Path $results "metric.json"), "--run-summary", (Join-Path $results "run-summary.json"), "--provenance", (Join-Path $results "provenance.json"))
}

function Invoke-Decide {
  $decision = @(Invoke-LoggedNative "Decide" "release-decision" $PythonExe @("-m", "eval.release_evidence", "decide", "--evidence-root", $EvidenceRoot)) -join [Environment]::NewLine
  $temporary = Join-Path $EvidenceRoot "decision.json.tmp"
  [IO.File]::WriteAllText($temporary, $decision + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
  Move-Item -Force $temporary (Join-Path $EvidenceRoot "decision.json")
}

function Get-Sha256([string]$Path) {
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-StringSha256([string]$Value) {
  $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
  $sha = [Security.Cryptography.SHA256]::Create()
  try { return ([BitConverter]::ToString($sha.ComputeHash($bytes)) -replace '-', '').ToLowerInvariant() }
  finally { $sha.Dispose() }
}

function Get-CDMToolEnvironmentSha256 {
  $records = @()
  foreach ($name in @("pdflatex", "kpsewhich", "magick")) {
    $commands = @(Get-Command $name -CommandType Application -ErrorAction SilentlyContinue)
    if ($commands.Count -ne 1 -or [string]::IsNullOrWhiteSpace([string]$commands[0].Source)) {
      $records += "$name|missing-or-ambiguous"
      continue
    }
    $path = Resolve-PhysicalPath ([string]$commands[0].Source)
    $records += "$name|$(Get-StringSha256 $path)|$(Get-Sha256 $path)"
  }
  $kpsewhich = @(Get-Command "kpsewhich" -CommandType Application -ErrorAction SilentlyContinue)
  foreach ($resource in @("CJK.sty", "c70gkai.fd")) {
    if ($kpsewhich.Count -ne 1) { $records += "$resource|missing-kpsewhich"; continue }
    $resourcePaths = @(& $kpsewhich[0].Source $resource)
    if ($LASTEXITCODE -ne 0 -or $resourcePaths.Count -ne 1 -or
        [string]::IsNullOrWhiteSpace([string]$resourcePaths[0]) -or
        -not (Test-Path -LiteralPath $resourcePaths[0] -PathType Leaf)) {
      $records += "$resource|missing"
      continue
    }
    $resolvedResource = Resolve-PhysicalPath ([string]$resourcePaths[0])
    $records += "$resource|$(Get-StringSha256 $resolvedResource)|$(Get-Sha256 $resolvedResource)"
  }
  $magick = @(Get-Command "magick" -CommandType Application -ErrorAction SilentlyContinue)
  if ($magick.Count -eq 1) {
    $policy = @(& $magick[0].Source "-list" "policy") -join [Environment]::NewLine
    $records += "magick-policy|$(Get-StringSha256 $policy)"
  } else { $records += "magick-policy|missing" }
  return Get-StringSha256 ($records -join "`n")
}

function Get-InvocationFingerprint([string]$StageName, [string]$ManifestSha) {
  $pythonProvenance = Get-PythonProvenance
  $scorerProvenance = Get-ScorerProvenance
  $values = [ordered]@{
    stage = $StageName
    producing_commit = $GitCommit
    input_manifest_sha256 = $ManifestSha
    server_url_sha256 = Get-StringSha256 $ServerUrl
    api_model_name_sha256 = Get-StringSha256 $ApiModelName
    evidence_path_sha256 = Get-StringSha256 (Resolve-PhysicalPath $EvidenceRoot)
    dataset_path_sha256 = Get-StringSha256 (Resolve-PhysicalPath $DatasetDir)
    layout_path_sha256 = Get-StringSha256 (Resolve-PhysicalPath $LayoutModel)
    runtime_config_path_sha256 = Get-StringSha256 (Resolve-PhysicalPath $RuntimeConfig)
    recovery_source_path_sha256 = $(if ([string]::IsNullOrWhiteSpace($RecoverySourceRoot)) { Get-StringSha256 "none" } else { Get-StringSha256 (Resolve-PhysicalPath $RecoverySourceRoot) })
    engine = $(if ($StageName -in @("Official", "OfficialInference")) { "official" } elseif ($StageName -eq "Lightweight") { "lightweight" } else { "none" })
    output_contract_sha256 = Get-StringSha256 "$StageName|official|lightweight|results|decision.json|copy-report|run-summary|provenance|directml-preflight"
    python_path_sha256 = $pythonProvenance.path_sha256
    python_file_sha256 = $pythonProvenance.file_sha256
    python_version_sha256 = $pythonProvenance.version_sha256
    python_executable_sha256 = $pythonProvenance.executable_sha256
    eval_origin_sha256 = $pythonProvenance.eval_origin_sha256
    package_origin_sha256 = $pythonProvenance.package_origin_sha256
    dependency_environment_sha256 = $pythonProvenance.dependency_environment_sha256
    scorer_python_path_sha256 = $scorerProvenance.path_sha256
    scorer_python_file_sha256 = $scorerProvenance.file_sha256
    scorer_environment_sha256 = $scorerProvenance.environment_sha256
    cdm_tool_environment_sha256 = Get-CDMToolEnvironmentSha256
    paddleocr_record_sha256 = $pythonProvenance.paddleocr_record_sha256
    paddlex_record_sha256 = $pythonProvenance.paddlex_record_sha256
    paddlepaddle_record_sha256 = $pythonProvenance.paddlepaddle_record_sha256
  }
  return Get-StringSha256 ($values | ConvertTo-Json -Compress)
}

function Write-AtomicJson([string]$Path, [object]$Value) {
  $temporary = "$Path.tmp"
  [IO.File]::WriteAllText($temporary, (($Value | ConvertTo-Json -Depth 8) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
  Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Get-StageOutputs([string]$StageName) {
  $roots = switch ($StageName) {
    "Preflight" { @($ManifestPath) }
    "OfficialInference" { if ([string]::IsNullOrWhiteSpace($RecoverySourceRoot)) { @((Join-Path $EvidenceRoot "official")) } else { @((Join-Path $EvidenceRoot "recovery/source.json")) } }
    "Official" { @((Join-Path $EvidenceRoot "results/official")) }
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
    "OfficialInference" { if ([string]::IsNullOrWhiteSpace($RecoverySourceRoot)) { @((Join-Path $EvidenceRoot "official")) } else { @((Join-Path $EvidenceRoot "recovery")) } }
    "Official" { @((Join-Path $EvidenceRoot "results/official")) }
    "Lightweight" { @((Join-Path $EvidenceRoot "lightweight"), (Join-Path $EvidenceRoot "results/lightweight")) }
    "Decide" { @((Join-Path $EvidenceRoot "decision.json"), (Join-Path $EvidenceRoot "decision.json.tmp")) }
  }
  foreach ($path in $paths) { if (Test-Path -LiteralPath $path) { Remove-Item -Force -Recurse -LiteralPath $path } }
}

function Test-OutputHashes([string]$StageName, [object]$Expected) {
  $current = Get-StageOutputs $StageName
  $expectedNames = @($Expected.PSObject.Properties.Name | Sort-Object)
  $currentNames = @($current.Keys | Sort-Object)
  if ((Compare-Object $expectedNames $currentNames).Count -ne 0) { return $false }
  foreach ($property in $Expected.PSObject.Properties) {
    $path = Join-Path $EvidenceRoot ($property.Name -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Sha256 $path) -ne $property.Value) { return $false }
  }
  return $Expected.PSObject.Properties.Count -gt 0
}

function Assert-CompletedStageIntegrity([string]$StageName) {
  if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw "Missing completed predecessor: $StageName" }
  $manifestSha = Get-Sha256 $ManifestPath
  $stateDir = Join-Path $LogDir "stages"
  $statePath = Join-Path $stateDir "$($StageName.ToLowerInvariant()).json"
  $stageCommandLog = Join-Path $stateDir "$($StageName.ToLowerInvariant()).commands.jsonl"
  if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) { throw "Missing completed predecessor: $StageName" }
  $prior = Get-Content -Raw $statePath | ConvertFrom-Json
  if ($prior.status -ne "completed" -or $prior.producing_commit -ne $GitCommit -or
      $prior.input_manifest_sha256 -ne $manifestSha -or
      $prior.invocation_fingerprint -ne (Get-InvocationFingerprint $StageName $manifestSha)) {
    throw "Predecessor state integrity mismatch: $StageName"
  }
  if (-not (Test-Path -LiteralPath $stageCommandLog -PathType Leaf) -or
      (Get-Sha256 $stageCommandLog) -ne $prior.command_sha256) { throw "Predecessor command log hash mismatch: $StageName" }
  if (-not (Test-OutputHashes $StageName $prior.output_sha256)) { throw "Predecessor output hash/set mismatch: $StageName" }
}

function Invoke-DurableStage([string]$StageName, [scriptblock]$Body) {
  Assert-OrCreateManifest
  $manifestSha = Get-Sha256 $ManifestPath
  $fingerprint = Get-InvocationFingerprint $StageName $manifestSha
  $stateDir = Join-Path $LogDir "stages"
  New-Item -ItemType Directory -Force $stateDir | Out-Null
  $statePath = Join-Path $stateDir "$($StageName.ToLowerInvariant()).json"
  $stageCommandLog = Join-Path $stateDir "$($StageName.ToLowerInvariant()).commands.jsonl"
  if (Test-Path -LiteralPath $statePath) {
    $prior = Get-Content -Raw $statePath | ConvertFrom-Json
    if ($prior.producing_commit -ne $GitCommit -or $prior.input_manifest_sha256 -ne $manifestSha) { throw "Stage state commit/input mismatch: $StageName" }
    if ($prior.invocation_fingerprint -ne $fingerprint) { throw "Stage invocation fingerprint mismatch: $StageName" }
    if ($prior.status -eq "completed") {
      if (-not (Test-Path -LiteralPath $stageCommandLog -PathType Leaf) -or (Get-Sha256 $stageCommandLog) -ne $prior.command_sha256) { throw "Completed stage command log hash mismatch: $StageName" }
      if (-not (Test-OutputHashes $StageName $prior.output_sha256)) { throw "Completed stage output hash/set mismatch: $StageName" }
      Write-Host "Skipping completed stage $StageName"
      return
    }
    Clear-StageOutputs $StageName
  } else {
    Clear-StageOutputs $StageName
  }
  if (Test-Path -LiteralPath $stageCommandLog) { Remove-Item -Force -LiteralPath $stageCommandLog }
  $started = [ordered]@{ stage = $StageName; status = "started"; producing_commit = $GitCommit; input_manifest_sha256 = $manifestSha; invocation_fingerprint = $fingerprint; command_sha256 = $null; output_sha256 = @{} }
  Write-AtomicJson $statePath $started
  try {
    & $Body
    $outputs = Get-StageOutputs $StageName
    if ($outputs.Count -eq 0) { throw "Stage produced no durable artifacts: $StageName" }
    $completed = [ordered]@{ stage = $StageName; status = "completed"; producing_commit = $GitCommit; input_manifest_sha256 = $manifestSha; invocation_fingerprint = $fingerprint; command_sha256 = (Get-Sha256 $stageCommandLog); output_sha256 = $outputs }
    Write-AtomicJson $statePath $completed
  } catch {
    $failed = [ordered]@{ stage = $StageName; status = "failed"; producing_commit = $GitCommit; input_manifest_sha256 = $manifestSha; invocation_fingerprint = $fingerprint; command_sha256 = $(if (Test-Path $stageCommandLog) { Get-Sha256 $stageCommandLog } else { $null }); output_sha256 = (Get-StageOutputs $StageName); error_sha256 = (Get-StringSha256 $_.Exception.Message) }
    Write-AtomicJson $statePath $failed
    throw
  }
}

try {
  switch ($Stage) {
    "Preflight" { Invoke-DurableStage "Preflight" { Invoke-Preflight } }
    "Official" {
      Assert-CompletedStageIntegrity "Preflight"
      Invoke-DurableStage "OfficialInference" { Invoke-OfficialInference }
      Assert-CompletedStageIntegrity "OfficialInference"
      Invoke-DurableStage "Official" { Invoke-OfficialScore }
    }
    "OfficialScore" {
      if ([string]::IsNullOrWhiteSpace($RecoverySourceRoot)) { throw "OfficialScore requires RecoverySourceRoot." }
      Invoke-DurableStage "Preflight" { Invoke-Preflight }
      Assert-CompletedStageIntegrity "Preflight"
      Invoke-DurableStage "OfficialInference" { Invoke-RecoveryAuthentication "OfficialInference" }
      Assert-CompletedStageIntegrity "OfficialInference"
      Invoke-DurableStage "Official" { Invoke-OfficialScore }
    }
    "Lightweight" { Assert-CompletedStageIntegrity "Preflight"; Assert-CompletedStageIntegrity "OfficialInference"; Assert-CompletedStageIntegrity "Official"; Invoke-DurableStage "Lightweight" { Invoke-Lightweight } }
    "Decide" { Assert-CompletedStageIntegrity "Preflight"; Assert-CompletedStageIntegrity "OfficialInference"; Assert-CompletedStageIntegrity "Official"; Assert-CompletedStageIntegrity "Lightweight"; Invoke-DurableStage "Decide" { Invoke-Decide } }
    "All" {
      if (-not [string]::IsNullOrWhiteSpace($RecoverySourceRoot)) { throw "All does not accept RecoverySourceRoot; use OfficialScore." }
      Invoke-DurableStage "Preflight" { Invoke-Preflight }
      Assert-CompletedStageIntegrity "Preflight"
      Invoke-DurableStage "OfficialInference" { Invoke-OfficialInference }
      Assert-CompletedStageIntegrity "OfficialInference"
      Invoke-DurableStage "Official" { Invoke-OfficialScore }
      Assert-CompletedStageIntegrity "Official"
      Invoke-DurableStage "Lightweight" { Invoke-Lightweight }
      Assert-CompletedStageIntegrity "Lightweight"
      Invoke-DurableStage "Decide" { Invoke-Decide }
      if (-not (Test-Path (Join-Path $EvidenceRoot "decision.json") -PathType Leaf)) { throw "All requires durable decision.json" }
    }
  }
} finally {
  Pop-Location
}
