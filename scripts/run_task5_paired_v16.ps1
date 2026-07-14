[CmdletBinding()]
param(
  [ValidateSet("Preflight", "Official", "Lightweight", "Score", "Compare", "Decide", "All")]
  [string]$Stage = "Preflight",
  [Parameter(Mandatory=$true)][string]$R7Root,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-z0-9][a-z0-9-]{0,63}$')][string]$AttemptId,
  [string]$PythonExe = ".\.venv\Scripts\python.exe",
  [string]$ScorerPythonExe = "C:\Users\rocm\Desktop\PaddleOCR-VL-ROCm-scorer-v16-py310\Scripts\python.exe",
  [string]$ServerUrl = "http://127.0.0.1:8111/v1",
  [string]$ApiModelName = "PaddleOCR-VL-1.6-GGUF.gguf",
  [string]$DatasetDir = "data/omnidocbench/v16",
  [string]$LayoutModel = "models/PP-DocLayoutV3-onnx",
  [string]$RuntimeConfig = "$HOME/.paddleocr-vl-rocm/config.json",
  [string]$G0Receipt = "",
  [ValidateRange(1, 2147483)][int]$CommandTimeoutSeconds = 86400,
  [ValidateRange(0, 300)][int]$TerminationGraceSeconds = 10
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
  function Resolve-FullPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
  }

  $R7Root = (Resolve-Path -LiteralPath (Resolve-FullPath $R7Root)).Path
  $Task5Root = Join-Path $R7Root "task5"
  $AttemptRoot = Join-Path (Join-Path $Task5Root "attempts") $AttemptId
  $WorkRoot = Join-Path $AttemptRoot "work"
  $CommandRoot = Join-Path $AttemptRoot "commands"
  $StageStatePath = Join-Path $AttemptRoot "stage-state.json"
  $ManifestPath = Join-Path $Task5Root "manifest.json"
  $DatasetDir = Resolve-FullPath $DatasetDir
  $LayoutModel = Resolve-FullPath $LayoutModel
  $RuntimeConfig = Resolve-FullPath $RuntimeConfig
  $PythonExe = Resolve-FullPath $PythonExe
  $ScorerPythonExe = Resolve-FullPath $ScorerPythonExe
  if ([string]::IsNullOrWhiteSpace($G0Receipt)) { $G0Receipt = Join-Path $R7Root "receipt.sha256.json" }
  $G0Receipt = Resolve-FullPath $G0Receipt

  $ApprovedExcludedStem = "newspaper_The Times UK_0801@magazinesclubnew_page_031" # peg-native issue #18248
  $Benchmark = "OmniDocBench-v1.6"
  $ExpectedPages = 1651
  $ExpectedPairs = 1650
  $ApprovedProviders = @("DmlExecutionProvider", "CPUExecutionProvider")

  function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  }

  function Get-StringSha256([string]$Value) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
      $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
      return ([BitConverter]::ToString($sha.ComputeHash($bytes)) -replace '-', '').ToLowerInvariant()
    } finally { $sha.Dispose() }
  }

  function Write-AtomicText([string]$Path, [string]$Text) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = Join-Path $parent (".{0}.{1}.tmp" -f (Split-Path -Leaf $Path), [Guid]::NewGuid().ToString("N"))
    [IO.File]::WriteAllText($temporary, $Text, [Text.UTF8Encoding]::new($false))
    Move-Item -Force -LiteralPath $temporary -Destination $Path
  }

  function Write-AtomicJson([string]$Path, [object]$Value) {
    Write-AtomicText $Path (($Value | ConvertTo-Json -Depth 20) + [Environment]::NewLine)
  }

  function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing JSON evidence: $Path" }
    return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
  }

  function Read-StrictJson([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing JSON evidence: $Path" }
    $code = 'import json,sys; p=sys.argv[1]; seen=lambda pairs: (_ for _ in ()).throw(ValueError("duplicate JSON key")) if len(pairs)!=len(dict(pairs)) else dict(pairs); v=json.loads(open(p,"r",encoding="utf-8").read(),object_pairs_hook=seen,parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON"))); print(json.dumps(v,separators=(",",":"),sort_keys=True))'
    $canonical = Invoke-DirectPython $code @($Path)
    return ($canonical | ConvertFrom-Json)
  }

  function Invoke-DirectPython([string]$Code, [string[]]$Arguments) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $PythonExe; $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
    $wrapper = "import base64;exec(base64.b64decode('$encoded'))"
    $nativeArguments = @("-c", $wrapper) + @($Arguments)
    $info.Arguments = (($nativeArguments | ForEach-Object { ConvertTo-NativeArgument ([string]$_) }) -join ' ')
    $process = [Diagnostics.Process]::Start($info)
    $stdoutTask = $process.StandardOutput.ReadToEndAsync(); $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit(30000)) {
      & taskkill.exe /PID $process.Id /T /F 2>&1 | Out-Null
      $process.Dispose()
      throw "strict helper timeout"
    }
    $stdout = $stdoutTask.GetAwaiter().GetResult(); $stderr = $stderrTask.GetAwaiter().GetResult()
    $exitCode = $process.ExitCode; $process.Dispose()
    if ($exitCode -ne 0) { throw "strict helper failed: $(Get-StringSha256 $stderr)" }
    return $stdout.Trim()
  }

  function ConvertTo-CanonicalJson([object]$Value) {
    $code = 'import json,sys; v=json.loads(sys.argv[1],parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON"))); print(json.dumps(v,ensure_ascii=False,separators=(",",":"),sort_keys=True))'
    return Invoke-DirectPython $code @(($Value | ConvertTo-Json -Compress -Depth 20))
  }

  function Assert-NoReparsePoint([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $item = Get-Item -Force -LiteralPath $Path
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
      throw "$Label cannot be a symlink or reparse point: $Path"
    }
  }

  function Assert-TrackedWorktreeClean {
    $rows = @(git status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0) { throw "Unable to authenticate tracked worktree state" }
    $unexpected = @($rows | Where-Object { $_ -notmatch '^\?\? eval/\.omnidocbench/' })
    if ($unexpected.Count -ne 0) { throw "Tracked worktree is not clean: $($unexpected -join '; ')" }
  }

  function Get-GitCommit {
    $commit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[0-9a-f]{40}$') { throw "Unable to authenticate producing commit" }
    return $commit
  }

  function Add-CommandRecord([string]$StageName, [object]$Record) {
    New-Item -ItemType Directory -Force -Path $CommandRoot | Out-Null
    $path = Join-Path $CommandRoot "$($StageName.ToLowerInvariant()).jsonl"
    [IO.File]::AppendAllText($path, (($Record | ConvertTo-Json -Compress -Depth 8) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
  }

  function Add-InternalCommandRecord([string]$StageName, [string]$Name, [string]$Description) {
    $logPath = Join-Path $CommandRoot "$($StageName.ToLowerInvariant())-$Name.log"
    Write-AtomicText $logPath ((Protect-LoggedText $Description) + [Environment]::NewLine)
    $relativeLog = $logPath.Substring($AttemptRoot.Length).TrimStart('\', '/') -replace '\\', '/'
    $now = [DateTimeOffset]::UtcNow.ToString("o")
    Add-CommandRecord $StageName ([ordered]@{
      name=$Name; executable_sha256=Get-StringSha256 "internal"; arguments_sha256=Get-StringSha256 $Description
      started_at_utc=$now; ended_at_utc=$now; exit_code=0; timed_out=$false; descendant_pids=@()
      termination_result="not-required"; orphan_audit="PASS"; log_path=$relativeLog; log_sha256=Get-Sha256 $logPath
    })
  }

  function ConvertTo-NativeArgument([string]$Value) {
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $builder = [Text.StringBuilder]::new('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
      if ($character -eq '\') { $slashes += 1; continue }
      if ($character -eq '"') {
        [void]$builder.Append(('\' * (2 * $slashes + 1)))
        [void]$builder.Append('"')
      } else {
        [void]$builder.Append(('\' * $slashes))
        [void]$builder.Append($character)
      }
      $slashes = 0
    }
    [void]$builder.Append(('\' * (2 * $slashes)))
    [void]$builder.Append('"')
    return $builder.ToString()
  }

  function Protect-LoggedText([string]$Value) {
    if ($null -eq $Value) { return "" }
    $redacted = $Value
    $redacted = [regex]::Replace($redacted, '(?im)(Authorization\s*:\s*)[^\r\n]+', '${1}<redacted>')
    $redacted = [regex]::Replace($redacted, '(?i)Bearer\s+[^\s,;"''\]]+', 'Bearer <redacted>')
    $redacted = [regex]::Replace($redacted, '(?i)\b(api[_-]?key|token|signature)\s*[=:]\s*[^\s,;"''\]]+', '${1}=<redacted>')
    $redacted = [regex]::Replace($redacted, '(?i)\bprompt\s*[=:]\s*[^\r\n]+', 'prompt=<prompt-redacted>')
    $redacted = [regex]::Replace($redacted, '(?i)\bpayload\s*[=:]\s*[^\r\n]+', 'payload=<payload-redacted>')
    $redacted = [regex]::Replace($redacted, '(?i)\braw[_-]?result\s*[=:]\s*[^\r\n]+', 'raw_result=<raw-result-redacted>')
    $redacted = [regex]::Replace($redacted, '(?i)(?<![A-Za-z0-9_])[A-Z]:\\[^\r\n\t"'']+', '<absolute-model-path>')
    return $redacted
  }

  function Get-ProcessTreeSnapshot([int]$RootPid, [DateTimeOffset]$RootStartedAt) {
    $all = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $children = @{}
    foreach ($item in $all) {
      $parent = [int]$item.ParentProcessId
      if (-not $children.ContainsKey($parent)) { $children[$parent] = [Collections.ArrayList]::new() }
      [void]$children[$parent].Add($item)
    }
    $records = [Collections.ArrayList]::new()
    $queue = [Collections.Queue]::new()
    $queue.Enqueue([pscustomobject]@{ pid=$RootPid; depth=0 })
    while ($queue.Count -gt 0) {
      $node = $queue.Dequeue()
      if (-not $children.ContainsKey([int]$node.pid)) { continue }
      foreach ($child in $children[[int]$node.pid]) {
        $createdUtc = ([DateTimeOffset]$child.CreationDate).ToUniversalTime()
        # A process older than the authenticated root cannot be its descendant; this is PID-reuse defense.
        if ($createdUtc -lt $RootStartedAt.AddSeconds(-2)) { continue }
        $record = [pscustomobject]@{
          pid = [int]$child.ProcessId
          parent_pid = [int]$child.ParentProcessId
          depth = [int]$node.depth + 1
          creation_utc = $createdUtc.ToString("o")
          creation_identity = [string]$child.CreationDate
        }
        [void]$records.Add($record)
        $queue.Enqueue([pscustomobject]@{ pid=$record.pid; depth=$record.depth })
      }
    }
    return @($records)
  }

  function Stop-AuthenticatedProcessTree([int]$RootPid, [DateTimeOffset]$RootStartedAt, [object[]]$Snapshot) {
    $targets = @($Snapshot | Sort-Object depth -Descending)
    $currentByPid = @{}
    foreach ($item in @(Get-CimInstance Win32_Process -ErrorAction Stop)) { $currentByPid[[int]$item.ProcessId] = $item }
    foreach ($target in $targets) {
      if (-not $currentByPid.ContainsKey([int]$target.pid)) { continue }
      $current = $currentByPid[[int]$target.pid]
      if ([int]$current.ParentProcessId -ne [int]$target.parent_pid -or [string]$current.CreationDate -cne [string]$target.creation_identity) { continue }
      Stop-Process -Id ([int]$target.pid) -Force -ErrorAction SilentlyContinue
    }
    $root = $(if ($currentByPid.ContainsKey($RootPid)) { $currentByPid[$RootPid] } else { $null })
    if ($null -ne $root) {
      $created = ([DateTimeOffset]$root.CreationDate).ToUniversalTime()
      if ($created -ge $RootStartedAt.AddSeconds(-2)) { Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue }
    }
    Start-Sleep -Milliseconds 100
    return @((Get-ProcessTreeSnapshot $RootPid $RootStartedAt) | ForEach-Object { [int]$_.pid })
  }

  function Invoke-LoggedNative([string]$StageName, [string]$Name, [string]$Executable, [string[]]$Arguments, [switch]$Capture) {
    $startedOffset = [DateTimeOffset]::UtcNow
    $started = $startedOffset.ToString("o")
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.Arguments = (($Arguments | ForEach-Object { ConvertTo-NativeArgument ([string]$_) }) -join ' ')
    $info.WorkingDirectory = $RepoRoot
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = [Text.Encoding]::UTF8
    $info.StandardErrorEncoding = [Text.Encoding]::UTF8
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    if (-not $process.Start()) { throw "Unable to launch command: $Name" }
    $childId = $process.Id
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $observedDuringRun = @{}
    $commandDeadline = [DateTimeOffset]::UtcNow.AddSeconds($CommandTimeoutSeconds)
    $completed = $false
    do {
      $completed = $process.WaitForExit(100)
      foreach ($item in @(Get-ProcessTreeSnapshot $childId $startedOffset)) {
        $observedDuringRun["$($item.pid)|$($item.creation_identity)"] = $item
      }
    } while (-not $completed -and [DateTimeOffset]::UtcNow -lt $commandDeadline)
    $timedOut = -not $completed
    $descendants = @($observedDuringRun.Values)
    $hadSurvivors = $descendants.Count -gt 0
    $remaining = @()
    $terminationResult = "not-required"
    if ($timedOut -or $hadSurvivors) {
      $observedByIdentity = @{}
      foreach ($item in $descendants) { $observedByIdentity["$($item.pid)|$($item.creation_identity)"] = $item }
      $terminationDeadline = [DateTimeOffset]::UtcNow.AddSeconds($TerminationGraceSeconds)
      do {
        $wave = @(Get-ProcessTreeSnapshot $childId $startedOffset)
        foreach ($item in $wave) { $observedByIdentity["$($item.pid)|$($item.creation_identity)"] = $item }
        $killSet = @($observedByIdentity.Values)
        if ($killSet.Count -eq 0) { break }
        [void](Stop-AuthenticatedProcessTree $childId $startedOffset $killSet)
        Start-Sleep -Milliseconds 100
      } while ([DateTimeOffset]::UtcNow -lt $terminationDeadline)
      $descendants = @($observedByIdentity.Values)
      $remaining = @((Get-ProcessTreeSnapshot $childId $startedOffset) | ForEach-Object { [int]$_.pid })
    }
    if (-not $process.HasExited) { [void]$process.WaitForExit([int]([Math]::Max(1, $TerminationGraceSeconds) * 1000)) }
    if (-not $process.HasExited) { $remaining += $childId }
    if ($timedOut -or $hadSurvivors) { $terminationResult = $(if ($remaining.Count -eq 0) { "terminated" } else { "survivors" }) }
    $streamWaitMs = [int]([Math]::Max(1, $TerminationGraceSeconds) * 1000)
    $stdout = $(if ($stdoutTask.Wait($streamWaitMs)) { $stdoutTask.GetAwaiter().GetResult() } else { "<raw-result-redacted: stream remained open>" })
    $stderr = $(if ($stderrTask.Wait($streamWaitMs)) { $stderrTask.GetAwaiter().GetResult() } else { "<raw-result-redacted: stream remained open>" })
    $exitCode = $(if ($process.HasExited -and -not $timedOut) { $process.ExitCode } else { -1 })
    $process.Dispose()
    $rendered = Protect-LoggedText "[stdout]`n$stdout`n[stderr]`n$stderr"
    $logPath = Join-Path $CommandRoot "$($StageName.ToLowerInvariant())-$Name.log"
    Write-AtomicText $logPath ($rendered + [Environment]::NewLine)
    $relativeLog = $logPath.Substring($AttemptRoot.Length).TrimStart('\', '/') -replace '\\', '/'
    Add-CommandRecord $StageName ([ordered]@{
      name = $Name
      executable_sha256 = Get-StringSha256 $Executable
      arguments_sha256 = Get-StringSha256 ($Arguments -join "`0")
      started_at_utc = $started
      ended_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
      exit_code = $exitCode
      timed_out = $timedOut
      descendant_pids = @($descendants | ForEach-Object { [int]$_.pid } | Sort-Object -Unique)
      termination_result = $terminationResult
      orphan_audit = $(if ($remaining.Count -eq 0) { "PASS" } else { "FAIL" })
      log_path = $relativeLog
      log_sha256 = Get-Sha256 $logPath
    })
    if ($remaining.Count -ne 0) { throw "orphan process audit failed for command: $Name" }
    if ($hadSurvivors -and -not $timedOut) { throw "orphan process observed after normal command exit: $Name" }
    if ($timedOut) { throw "Command timeout ($Name) after $CommandTimeoutSeconds seconds" }
    if ($exitCode -ne 0) { throw "Command failed ($Name), exit=$exitCode. See $logPath" }
    if ($Capture) { return $stdout.Trim() }
  }

  function Get-ImmutableInputs {
    $datasetManifest = Join-Path $DatasetDir "OmniDocBench.json"
    $layoutOnnx = Join-Path $LayoutModel "inference.onnx"
    $layoutConfig = Join-Path $LayoutModel "inference.yml"
    $runtimeManifest = Join-Path $RepoRoot "src/paddleocr_vl_rocm/assets/runtime-manifest.json"
    $scoringConfig = Join-Path $RepoRoot "eval/configs/omnidocbench_v16.yaml"
    $required = [ordered]@{
      dataset = $datasetManifest
      layout_model = $layoutOnnx
      layout_config = $layoutConfig
      runtime_config = $RuntimeConfig
      runtime_manifest = $runtimeManifest
      scoring_config = $scoringConfig
      scorer_requirements = (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16.txt")
      scorer_transitive_requirements = (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16-transitive.txt")
      benchmark_contract = (Join-Path $RepoRoot "eval/benchmark_contract.py")
      task5_runner = $PSCommandPath
      python_executable = $PythonExe
      scorer_python_executable = $ScorerPythonExe
    }
    if (-not (Test-Path -LiteralPath $RuntimeConfig -PathType Leaf)) { throw "Active RuntimeConfig is missing" }
    $config = Read-Json $RuntimeConfig
    $required.main_gguf = [string]$config.main_gguf
    $required.mmproj = [string]$config.mmproj
    if ([string]$config.layout_model_dir -and (Resolve-Path $config.layout_model_dir).Path -ne (Resolve-Path $LayoutModel).Path) {
      throw "RuntimeConfig layout_model_dir does not match LayoutModel"
    }
    foreach ($entry in $required.GetEnumerator()) {
      if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) { throw "Required immutable input is missing: $($entry.Key)" }
    }
    return $required
  }

  function Get-EnvironmentContract {
    function Get-PythonRuntime([string]$Executable) {
      $code = 'import json,sys; print(json.dumps({"version":sys.version.split()[0],"executable_sha256":__import__("hashlib").sha256(open(sys.executable,"rb").read()).hexdigest()},sort_keys=True,separators=(",",":")))'
      if ($Executable -cne $PythonExe) {
        $originalPython = $script:PythonExe; $script:PythonExe = $Executable
        try { $value = Invoke-DirectPython $code @() } finally { $script:PythonExe = $originalPython }
      } else { $value = Invoke-DirectPython $code @() }
      return ($value | ConvertFrom-Json)
    }
    function Get-PackageRuntime([string]$Package, [switch]$Providers) {
      $code = $(if ($Providers) { 'import json,onnxruntime as m; print(json.dumps({"version":m.__version__,"available_providers":m.get_available_providers()},sort_keys=True,separators=(",",":")))' } else { 'import importlib.metadata as m,json,sys; p=sys.argv[1]; print(json.dumps({"version":m.version(p)},sort_keys=True,separators=(",",":")))' })
      try { $value = Invoke-DirectPython $code @($Package) } catch { throw "Package environment identity failed: $Package" }
      return ($value | ConvertFrom-Json)
    }
    $ort = Get-PackageRuntime "onnxruntime-directml" -Providers
    $serverCode = 'import hashlib,json,sys,urllib.request; u=sys.argv[1].rstrip("/")+"/models"; v=json.load(urllib.request.urlopen(u,timeout=10)); c=json.dumps(v,ensure_ascii=False,separators=(",",":"),sort_keys=True).encode(); print(json.dumps({"models_sha256":hashlib.sha256(c).hexdigest(),"requested_model":sys.argv[2]},sort_keys=True,separators=(",",":")))'
    try { $server = Invoke-DirectPython $serverCode @($ServerUrl, $ApiModelName) } catch { throw "Server model runtime identity failed" }
    $gpus = @(
      Get-CimInstance Win32_VideoController -ErrorAction Stop |
        ForEach-Object { [ordered]@{ Name=[string]$_.Name; PNPDeviceID=[string]$_.PNPDeviceID; DriverVersion=[string]$_.DriverVersion } } |
        Sort-Object PNPDeviceID
    )
    return [ordered]@{
      benchmark = $Benchmark
      os = [Environment]::OSVersion.VersionString
      machine = [Environment]::MachineName
      gpu_devices = $gpus
      python = Get-PythonRuntime $PythonExe
      scorer_python = Get-PythonRuntime $ScorerPythonExe
      onnxruntime = [ordered]@{ version=[string]$ort.version }
      available_providers = @($ort.available_providers)
      paddleocr = Get-PackageRuntime "paddleocr"
      official_adapter = [ordered]@{
        image_to_markdown = Get-Sha256 (Join-Path $RepoRoot "eval/PaddleOCRVLROCm_img2md.py")
        evaluation = Get-Sha256 (Join-Path $RepoRoot "eval/run_eval.py")
      }
      lightweight_adapter = [ordered]@{
        layout = Get-Sha256 (Join-Path $RepoRoot "src/paddleocr_vl_rocm/layout.py")
        pipeline = Get-Sha256 (Join-Path $RepoRoot "src/paddleocr_vl_rocm/pipeline.py")
      }
      server_model_runtime = ($server | ConvertFrom-Json)
    }
  }

  function Get-InferenceContract {
    return [ordered]@{
      benchmark = $Benchmark
      pages = $ExpectedPages
      paired_pages = $ExpectedPairs
      approved_exclusion = $ApprovedExcludedStem
      page_retries = 1
      cross_engine_fallback = $false
      formula = "display_formula.page.CDM.ALL"
      table = "table.page.TEDS.ALL"
    }
  }

  function Assert-OrCreateManifest([string]$StageName) {
    $inputs = Get-ImmutableInputs
    if (Test-Path -LiteralPath $ManifestPath -PathType Leaf) {
      Invoke-LoggedNative $StageName "manifest-validate" $PythonExe @("-m", "eval.task5_manifest", "validate", "--manifest", $ManifestPath, "--task5-root", $Task5Root)
      return
    }
    $args = @("-m", "eval.task5_manifest", "create", "--r7-root", $R7Root, "--receipt", $G0Receipt, "--git-commit", (Get-GitCommit))
    foreach ($entry in $inputs.GetEnumerator()) { $args += @("--input", "$($entry.Key)=$($entry.Value)") }
    $args += @("--environment-json", ((Get-EnvironmentContract) | ConvertTo-Json -Compress), "--contracts-json", ((Get-InferenceContract) | ConvertTo-Json -Compress), "--output", $ManifestPath)
    Invoke-LoggedNative $StageName "manifest-create" $PythonExe $args
  }

  function Get-OutputMap([string[]]$Roots) {
    $map = [ordered]@{}
    foreach ($root in $Roots) {
      if (Test-Path -LiteralPath $root -PathType Leaf) { $files = @(Get-Item -LiteralPath $root) }
      elseif (Test-Path -LiteralPath $root -PathType Container) { $files = @(Get-ChildItem -LiteralPath $root -File -Recurse | Sort-Object FullName) }
      else { throw "Stage output is missing: $root" }
      foreach ($file in $files) {
        $relative = $file.FullName.Substring($Task5Root.Length).TrimStart('\', '/') -replace '\\', '/'
        $map[$relative] = Get-Sha256 $file.FullName
      }
    }
    return $map
  }

  function Get-OutputMapSha([object]$Map) {
    return Get-StringSha256 ($Map | ConvertTo-Json -Compress -Depth 10)
  }

  function Read-State {
    $state = Read-Json $StageStatePath
    if ($state.attempt_id -ne $AttemptId) { throw "old-attempt file reuse detected" }
    if ($state.status -eq "invalid") { throw "Attempt is invalid; use a new AttemptId" }
    return $state
  }

  function Assert-RecordedStagesIntegrity([object]$State) {
    if ($State.producing_commit -ne (Get-GitCommit)) { throw "producing commit integrity mismatch" }
    if ($State.manifest_sha256 -ne (Get-Sha256 $ManifestPath)) { throw "manifest integrity mismatch" }
    foreach ($property in $State.stages.PSObject.Properties) {
      $record = $property.Value
      $commandPath = Join-Path $CommandRoot "$($property.Name.ToLowerInvariant()).jsonl"
      if (-not (Test-Path -LiteralPath $commandPath -PathType Leaf) -or (Get-Sha256 $commandPath) -ne $record.command_log_sha256) {
        throw "command log integrity mismatch: $($property.Name)"
      }
      Assert-CommandLogIntegrity $commandPath
      $map = Get-OutputMap @($record.output_roots)
      if ((Get-OutputMapSha $map) -ne $record.output_map_sha256) { throw "output integrity mismatch: $($property.Name)" }
    }
  }

  function Assert-CommandLogIntegrity([string]$JsonlPath) {
    # Authenticate the durable orphan audit and its backing process-lifecycle fields.
    $code = @'
import json,re,sys
expected={"name","executable_sha256","arguments_sha256","started_at_utc","ended_at_utc","exit_code","timed_out","descendant_pids","termination_result","orphan_audit","log_path","log_sha256"}
def pairs(v):
    if len(v)!=len(dict(v)): raise ValueError("duplicate JSON key")
    return dict(v)
rows=[]
for line in open(sys.argv[1],encoding="utf-8"):
    if not line.strip(): raise ValueError("blank JSONL line")
    row=json.loads(line,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    if set(row)!=expected: raise ValueError("command record schema mismatch")
    if not isinstance(row["name"],str) or not row["name"]: raise ValueError("invalid command name")
    if any(not isinstance(row[k],str) or not re.fullmatch(r"[0-9a-f]{64}",row[k]) for k in ("executable_sha256","arguments_sha256","log_sha256")): raise ValueError("invalid command digest")
    if any(not isinstance(row[k],str) or not row[k] for k in ("started_at_utc","ended_at_utc","termination_result","orphan_audit","log_path")): raise ValueError("invalid command scalar")
    if type(row["exit_code"]) is not int or type(row["timed_out"]) is not bool: raise ValueError("invalid command status type")
    if not isinstance(row["descendant_pids"],list) or any(type(v) is not int or v<=0 for v in row["descendant_pids"]): raise ValueError("invalid descendant PID list")
    if row["termination_result"] not in {"not-required","terminated","survivors"}: raise ValueError("invalid termination result")
    if row["orphan_audit"] not in {"PASS","FAIL"}: raise ValueError("invalid orphan audit")
    rows.append(row)
if not rows or len({r["name"] for r in rows})!=len(rows): raise ValueError("duplicate command name")
print(json.dumps(rows,separators=(",",":"),sort_keys=True))
'@
    try { $parsed = Invoke-DirectPython $code @($JsonlPath) }
    catch { throw "command log strict JSON integrity mismatch: $JsonlPath ($($_.Exception.Message))" }
    $records = @(($parsed | ConvertFrom-Json))
    foreach ($entry in $records) {
      if ($entry.exit_code -ne 0 -or $entry.timed_out -ne $false -or $entry.orphan_audit -cne "PASS") { throw "command log execution integrity mismatch" }
      if ([string]$entry.log_path -match '(^[\\/]|^[A-Za-z]:|\.\.)') { throw "command log path integrity mismatch" }
      $logPath = [IO.Path]::GetFullPath((Join-Path $AttemptRoot ([string]$entry.log_path -replace '/', '\')))
      if (-not $logPath.StartsWith($AttemptRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
          -not (Test-Path -LiteralPath $logPath -PathType Leaf) -or (Get-Sha256 $logPath) -cne [string]$entry.log_sha256) {
        throw "command log digest mismatch"
      }
    }
  }

  function Assert-StageStartIntegrity([object]$State, [string]$StageName, [string]$CommandName = "manifest-revalidate") {
    $head = Get-GitCommit
    $manifest = Read-StrictJson $ManifestPath
    if ([string]$manifest.git_commit -cne $head -or [string]$State.producing_commit -cne [string]$manifest.git_commit) {
      throw "producing commit integrity mismatch"
    }
    if ([string]$State.manifest_sha256 -cne (Get-Sha256 $ManifestPath)) { throw "manifest integrity mismatch" }
    $environment = Get-EnvironmentContract
    if ((ConvertTo-CanonicalJson $manifest.environment) -cne (ConvertTo-CanonicalJson $environment)) {
      throw "manifest environment integrity mismatch"
    }
    # This is intentionally after all direct prechecks so its own log cannot recursively authenticate itself.
    Invoke-LoggedNative $StageName $CommandName $PythonExe @("-m", "eval.task5_manifest", "validate", "--manifest", $ManifestPath, "--task5-root", $Task5Root)
  }

  function Start-Attempt {
    Assert-NoReparsePoint $Task5Root "task5 root"
    if (Test-Path -LiteralPath $AttemptRoot) { throw "Attempt already exists; use a new AttemptId" }
    $attemptsRoot = Join-Path $Task5Root "attempts"
    if (-not (Test-Path -LiteralPath $attemptsRoot)) { New-Item -ItemType Directory -Path $attemptsRoot | Out-Null }
    Assert-NoReparsePoint $attemptsRoot "attempt parent"
    New-Item -ItemType Directory -Path $AttemptRoot, $WorkRoot, $CommandRoot | Out-Null
    Assert-NoReparsePoint $AttemptRoot "attempt root"
    $state = [ordered]@{
      schema = 1
      attempt_id = $AttemptId
      status = "active"
      producing_commit = Get-GitCommit
      manifest_sha256 = $null
      stages = [ordered]@{}
      started_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
    }
    Write-AtomicJson $StageStatePath $state
  }

  function Invoke-DurableStage([string]$StageName, [scriptblock]$Body, [string[]]$OutputRoots) {
    $state = Read-State
    Assert-RecordedStagesIntegrity $state
    Assert-StageStartIntegrity $state $StageName
    if ($state.stages.PSObject.Properties.Name -contains $StageName) { throw "Stage already completed: $StageName" }
    $started = [DateTimeOffset]::UtcNow.ToString("o")
    try {
      & $Body
      $map = Get-OutputMap $OutputRoots
      $commandPath = Join-Path $CommandRoot "$($StageName.ToLowerInvariant()).jsonl"
      if (-not (Test-Path -LiteralPath $commandPath -PathType Leaf)) { throw "Stage command log is missing" }
      $record = [ordered]@{
        started_at_utc = $started
        ended_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        exit_code = 0
        command_log_sha256 = Get-Sha256 $commandPath
        output_map_sha256 = Get-OutputMapSha $map
        output_roots = $OutputRoots
        orphan_audit = "PASS"
      }
      $state.stages | Add-Member -NotePropertyName $StageName -NotePropertyValue $record
      Write-AtomicJson $StageStatePath $state
    } catch {
      $state.status = "invalid"
      $state | Add-Member -Force -NotePropertyName failure -NotePropertyValue ([ordered]@{ stage=$StageName; ended_at_utc=[DateTimeOffset]::UtcNow.ToString("o"); error_sha256=Get-StringSha256 $_.Exception.Message })
      Write-AtomicJson $StageStatePath $state
      throw
    }
  }

  function Invoke-Preflight {
    Assert-TrackedWorktreeClean
    Assert-NoReparsePoint $Task5Root "task5 root"
    if (-not (Test-Path -LiteralPath $G0Receipt -PathType Leaf)) { throw "G0 receipt is missing" }
    Assert-NoReparsePoint $G0Receipt "G0 receipt"
    if ((Resolve-Path -LiteralPath $G0Receipt).Path -cne [IO.Path]::GetFullPath($G0Receipt)) { throw "G0 receipt path must be canonical and cannot traverse a symlink" }
    Start-Attempt
    Assert-OrCreateManifest "Preflight"
    $state = Read-State
    $state.manifest_sha256 = Get-Sha256 $ManifestPath
    Write-AtomicJson $StageStatePath $state
    $snapshot = Invoke-LoggedNative "Preflight" "g0-snapshot-before" $PythonExe @("-m", "eval.task5_manifest", "snapshot", "--r7-root", $R7Root, "--receipt", $G0Receipt) -Capture
    Write-AtomicText (Join-Path $AttemptRoot "snapshot-before.json") ($snapshot + [Environment]::NewLine)
    Invoke-LoggedNative "Preflight" "python-auth" $PythonExe @("--version")
    Invoke-LoggedNative "Preflight" "scorer-python-auth" $ScorerPythonExe @("--version")
    Invoke-LoggedNative "Preflight" "scorer-contract" $PythonExe @("-m", "eval.benchmark_contract", "--checkout", (Join-Path $RepoRoot "eval/.omnidocbench"))
    Invoke-LoggedNative "Preflight" "scorer-preflight" $ScorerPythonExe @("scripts/check_omnidocbench_scorer.py", "--checkout", (Join-Path $RepoRoot "eval/.omnidocbench"), "--direct-lock", (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16.txt"), "--transitive-lock", (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16-transitive.txt"), "--require-cdm-tools")
    Invoke-LoggedNative "Preflight" "server" $PythonExe @("scripts/check_server.py", "--server-url", $ServerUrl)
    Invoke-LoggedNative "Preflight" "official-constructor" $PythonExe @("scripts/check_official_paddleocr.py", "--construct", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName)
  }

  function Assert-OfficialCoverage([object]$Stats) {
    if ($Stats.count -ne 1651 -or $Stats.ok -ne 1650 -or $Stats.fail -ne 1 -or $Stats.fallback -ne 0 -or $null -ne $Stats.limit_pages) {
      throw "Official coverage integrity mismatch: approved peg-native contract requires 1651/1650/1/0/null; Official fallback is forbidden"
    }
  }

  function Assert-LightweightCoverage([object]$Stats) {
    if ($Stats.count -ne 1651 -or $Stats.ok -ne 1651 -or $Stats.fail -ne 0 -or $Stats.fallback -ne 0 -or $null -ne $Stats.limit_pages) {
      throw "Lightweight coverage integrity mismatch: partial coverage or fallback"
    }
  }

  function Invoke-Official {
    $predictions = Join-Path $WorkRoot "paired-official"
    $traces = Join-Path $WorkRoot "traces/official"
    Invoke-LoggedNative "Official" "official-infer" $PythonExe @("-m", "eval.run_eval", "--stage", "infer", "--version", "v16", "--engine", "official", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName, "--dataset-dir", $DatasetDir, "--predictions-dir", $predictions, "--trace-dir", $traces, "--page-retries", "1")
    $stats = Join-Path $predictions "_run_stats.json"
    Invoke-LoggedNative "Official" "official-contract" $PythonExe @("-m", "eval.release_contract", "--stats", $stats, "--version", "v16", "--engine", "official")
    Assert-OfficialCoverage (Read-Json $stats)
  }

  function Invoke-Lightweight {
    $predictions = Join-Path $WorkRoot "lightweight"
    $traces = Join-Path $WorkRoot "traces/lightweight"
    $profileRoot = Join-Path $WorkRoot "profiles"
    New-Item -ItemType Directory -Path $profileRoot | Out-Null
    $profilePrefix = Join-Path $profileRoot "layout-profile"
    Invoke-LoggedNative "Lightweight" "lightweight-infer" $PythonExe @("-m", "eval.run_eval", "--stage", "infer", "--version", "v16", "--engine", "lightweight", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName, "--dataset-dir", $DatasetDir, "--predictions-dir", $predictions, "--layout-model", $LayoutModel, "--trace-dir", $traces, "--layout-profile-prefix", $profilePrefix)
    $stats = Join-Path $predictions "_run_stats.json"
    Invoke-LoggedNative "Lightweight" "lightweight-contract" $PythonExe @("-m", "eval.release_contract", "--stats", $stats, "--version", "v16", "--engine", "lightweight")
    Assert-LightweightCoverage (Read-Json $stats)
    $profiles = @(Get-ChildItem -LiteralPath $profileRoot -Filter "layout-profile*.json" -File)
    if ($profiles.Count -ne 1) { throw "missing profile or ambiguous DirectML profile" }
  }

  function Invoke-EngineScores([string]$Engine, [string]$StageName) {
    $predictions = Join-Path $WorkRoot $(if ($Engine -eq "official") { "paired-official" } else { "lightweight" })
    $results = Join-Path $WorkRoot "results/$Engine"
    New-Item -ItemType Directory -Force -Path $results | Out-Null
    $common = @("-m", "eval.run_eval", "--stage", "eval", "--version", "v16", "--engine", $Engine, "--dataset-dir", $DatasetDir, "--predictions-dir", $predictions, "--scorer-python", $ScorerPythonExe)
    Invoke-LoggedNative $StageName "$Engine-score" $PythonExe ($common + @("--copy-report", (Join-Path $results "metric.json"), "--run-summary", (Join-Path $results "run-summary.json"), "--provenance", (Join-Path $results "provenance.json")))
    $cdmFlag = if ($Engine -eq "official") { "--cdm" } else { "--cdm" }
    Invoke-LoggedNative $StageName "$Engine-score-cdm" $PythonExe ($common + @($cdmFlag, "--copy-report", (Join-Path $results "metric-cdm.json"), "--run-summary", (Join-Path $results "run-summary-cdm.json"), "--provenance", (Join-Path $results "provenance-cdm.json")))
    # Quality is fail-closed later in task5_decision: CDM timeout and TEDS error cannot yield AMD PASS.
    foreach ($name in @("metric.json", "metric-cdm.json", "run-summary.json", "run-summary-cdm.json", "provenance.json", "provenance-cdm.json")) {
      if (-not (Test-Path -LiteralPath (Join-Path $results $name) -PathType Leaf)) { throw "stale score or missing fresh score artifact: $Engine/$name" }
    }
  }

  function Invoke-Score {
    Invoke-EngineScores "official" "Score"
    Invoke-EngineScores "lightweight" "Score"
  }

  function Invoke-Compare {
    $comparison = Join-Path $WorkRoot "comparison"
    New-Item -ItemType Directory -Force -Path $comparison | Out-Null
    $inline = 'import json,sys; from pathlib import Path; from eval.task5_comparison import compare_prediction_dirs; p=compare_prediction_dirs(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]); Path(sys.argv[4]).write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")'
    Invoke-LoggedNative "Compare" "normalized-output" $PythonExe @("-c", $inline, (Join-Path $WorkRoot "paired-official"), (Join-Path $WorkRoot "lightweight"), $ApprovedExcludedStem, (Join-Path $comparison "normalized-output.json"))
    Invoke-LoggedNative "Compare" "trace-diff" $PythonExe @("scripts/compare_inference_traces.py", (Join-Path $WorkRoot "traces/official"), (Join-Path $WorkRoot "traces/lightweight"), "--output", (Join-Path $comparison "trace-diff.json"))
    $contract = [ordered]@{ benchmark=$Benchmark; pages=$ExpectedPages; paired_pages=$ExpectedPairs; approved_exclusion=$ApprovedExcludedStem; formula="CDM"; table="TEDS" }
    Write-AtomicJson (Join-Path $comparison "input-contract.json") $contract
    Add-InternalCommandRecord "Compare" "input-contract" ($contract | ConvertTo-Json -Compress)
  }

  function Copy-CompactEvidence {
    function Publish-FileAtomic([string]$Source, [string]$Destination) {
      if (Test-Path -LiteralPath $Destination) { throw "Published evidence already exists" }
      $temporary = "$Destination.$AttemptId.tmp"
      if (Test-Path -LiteralPath $temporary) { throw "Stale publication temporary exists" }
      Copy-Item -LiteralPath $Source -Destination $temporary
      if ((Get-Sha256 $Source) -ne (Get-Sha256 $temporary)) { throw "Published evidence staging hash mismatch" }
      Move-Item -LiteralPath $temporary -Destination $Destination
      if ((Get-Sha256 $Source) -ne (Get-Sha256 $Destination)) { throw "Published evidence final hash mismatch" }
    }
    foreach ($engine in @("official", "lightweight")) {
      $destination = Join-Path $Task5Root "results/$engine"
      if (Test-Path -LiteralPath $destination) { throw "Published compact result already exists; old-attempt file reuse is forbidden" }
      New-Item -ItemType Directory -Path $destination | Out-Null
      foreach ($file in Get-ChildItem -LiteralPath (Join-Path $WorkRoot "results/$engine") -File) {
        Publish-FileAtomic $file.FullName (Join-Path $destination $file.Name)
      }
    }
    $destination = Join-Path $Task5Root "comparison"
    if (Test-Path -LiteralPath $destination) { throw "Published comparison already exists; old-attempt file reuse is forbidden" }
    New-Item -ItemType Directory -Path $destination | Out-Null
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $WorkRoot "comparison") -File) {
      Publish-FileAtomic $file.FullName (Join-Path $destination $file.Name)
    }
  }

  function Assert-ProviderMajority([object]$Report) {
    if ($Report.verdict -ne "PASS") { throw "DirectML attestation FAIL" }
    if ($Report.dml_node_events -le 0 -or $Report.dml_node_share -le 0.5) { throw "DirectML node share must be strictly greater than 0.5" }
    if ($Report.missing_provider_node_events -ne 0 -or $Report.other_provider_node_events -ne 0) { throw "missing/other provider node events are forbidden" }
    $stats = Read-Json (Join-Path $WorkRoot "lightweight/_run_stats.json")
    if ($stats.layout_provider_requested -ne "auto" -or @($stats.layout_providers_active).Count -ne 2 -or $stats.layout_providers_active[0] -ne $ApprovedProviders[0] -or $stats.layout_providers_active[1] -ne $ApprovedProviders[1]) { throw "CPU-first provider order or provider contract mismatch" }
    if ($stats.layout_fallback_disabled -ne $true) { throw "Runtime fallback must be disabled" }
  }

  function Invoke-Decide {
    $comparison = Join-Path $WorkRoot "comparison"
    $profiles = @(Get-ChildItem -LiteralPath (Join-Path $WorkRoot "profiles") -Filter "layout-profile*.json" -File)
    if ($profiles.Count -ne 1) { throw "missing profile" }
    $attestation = Invoke-LoggedNative "Decide" "directml-attestation" $PythonExe @("-m", "eval.directml_attestation", "--profile", $profiles[0].FullName, "--stats", (Join-Path $WorkRoot "lightweight/_run_stats.json"), "--allow-fail-verdict") -Capture
    $strictJson = 'import json,sys; seen=lambda pairs: (_ for _ in ()).throw(ValueError("duplicate JSON key")) if len(pairs)!=len(dict(pairs)) else dict(pairs); value=json.loads(sys.argv[1],parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON")),object_pairs_hook=seen); assert isinstance(value,dict)'
    Invoke-LoggedNative "Decide" "directml-strict-json" $PythonExe @("-c", $strictJson, $attestation)
    Write-AtomicText (Join-Path $comparison "directml-attestation.json") ($attestation + [Environment]::NewLine)
    $r = Join-Path $WorkRoot "results"
    Invoke-LoggedNative "Decide" "decision" $PythonExe @("-m", "eval.task5_decision", "decide", "--official-non-cdm", (Join-Path $r "official/metric.json"), "--official-cdm", (Join-Path $r "official/metric-cdm.json"), "--lightweight-non-cdm", (Join-Path $r "lightweight/metric.json"), "--lightweight-cdm", (Join-Path $r "lightweight/metric-cdm.json"), "--output-report", (Join-Path $comparison "normalized-output.json"), "--trace-report", (Join-Path $comparison "trace-diff.json"), "--provider-attestation", (Join-Path $comparison "directml-attestation.json"), "--lightweight-stats", (Join-Path $WorkRoot "lightweight/_run_stats.json"), "--public-contracts-pass", "--output", (Join-Path $comparison "decision.json"))
    $measuredDecision = Read-Json (Join-Path $comparison "decision.json")
    if ($measuredDecision.amd_adaptation.verdict -eq "PASS") { Assert-ProviderMajority (Read-Json (Join-Path $comparison "directml-attestation.json")) }
    $after = Invoke-LoggedNative "Decide" "g0-snapshot-after" $PythonExe @("-m", "eval.task5_manifest", "snapshot", "--r7-root", $R7Root, "--receipt", $G0Receipt) -Capture
    Write-AtomicText (Join-Path $AttemptRoot "snapshot-after.json") ($after + [Environment]::NewLine)
    $beforeObject = Read-Json (Join-Path $AttemptRoot "snapshot-before.json")
    $afterObject = Read-Json (Join-Path $AttemptRoot "snapshot-after.json")
    if (($beforeObject | ConvertTo-Json -Compress -Depth 20) -cne ($afterObject | ConvertTo-Json -Compress -Depth 20)) { throw "G0 integrity mismatch" }
    Assert-StageStartIntegrity (Read-State) "Decide" "manifest-revalidate-seal"
    Copy-CompactEvidence
    $decision = Read-Json (Join-Path $comparison "decision.json")
    $selected = [ordered]@{schema=1;attempt_id=$AttemptId;manifest_sha256=Get-Sha256 $ManifestPath;strict_equivalence=$decision.strict_equivalence.verdict;amd_adaptation=$decision.amd_adaptation.verdict;effective_only_with_valid_receipt=$true;g0_closure="PASS";selected_at_utc=[DateTimeOffset]::UtcNow.ToString("o")}
    Write-AtomicJson (Join-Path $Task5Root "selected-attempt.json") $selected
  }

  function Complete-Receipt {
    $state = Read-State
    Assert-RecordedStagesIntegrity $state
    Assert-StageStartIntegrity $state "Receipt" "manifest-revalidate-receipt"
    $receipt = Join-Path $Task5Root "receipt.sha256.json"
    if (Test-Path -LiteralPath $receipt) { throw "receipt mutation or replacement is forbidden" }
    $paths = @(
      "manifest.json", "selected-attempt.json",
      "attempts/$AttemptId/stage-state.json", "attempts/$AttemptId/snapshot-before.json", "attempts/$AttemptId/snapshot-after.json",
      "results/official/metric.json", "results/official/metric-cdm.json", "results/official/run-summary.json", "results/official/run-summary-cdm.json", "results/official/provenance.json", "results/official/provenance-cdm.json",
      "results/lightweight/metric.json", "results/lightweight/metric-cdm.json", "results/lightweight/run-summary.json", "results/lightweight/run-summary-cdm.json", "results/lightweight/provenance.json", "results/lightweight/provenance-cdm.json",
      "comparison/input-contract.json", "comparison/normalized-output.json", "comparison/trace-diff.json", "comparison/directml-attestation.json", "comparison/decision.json"
    )
    $args = @("-m", "eval.task5_decision", "receipt", "--task5-root", $Task5Root)
    foreach ($path in $paths) { $args += @("--path", $path) }
    $args += @("--output", $receipt)
    try {
      Invoke-LoggedNative "Receipt" "receipt" $PythonExe $args
      Invoke-LoggedNative "Receipt" "receipt-validate" $PythonExe @("-m", "eval.task5_decision", "validate-receipt", "--task5-root", $Task5Root, "--receipt", $receipt)
    } catch {
      $state = Read-Json $StageStatePath
      $state.status = "invalid"
      $state | Add-Member -Force -NotePropertyName failure -NotePropertyValue ([ordered]@{stage="Receipt";ended_at_utc=[DateTimeOffset]::UtcNow.ToString("o");error_sha256=Get-StringSha256 $_.Exception.Message;selected_marker_effective=$false})
      Write-AtomicJson $StageStatePath $state
      throw
    }
    $decision = Read-Json (Join-Path $Task5Root "comparison/decision.json")
    $strictVerdict = $decision.strict_equivalence.verdict
    $amdVerdict = $decision.amd_adaptation.verdict
    Write-Host "strict_equivalence=$strictVerdict"
    Write-Host "amd_adaptation=$amdVerdict"
  }

  function Require-Stages([string[]]$Names) {
    $state = Read-State
    Assert-RecordedStagesIntegrity $state
    foreach ($name in $Names) {
      if ($state.stages.PSObject.Properties.Name -notcontains $name) { throw "Missing completed predecessor: $name" }
    }
  }

  switch ($Stage) {
    "Preflight" {
      Invoke-Preflight
      Invoke-DurableStage "Preflight" { Add-InternalCommandRecord "Preflight" "preflight-complete" $Benchmark } @($ManifestPath, (Join-Path $AttemptRoot "snapshot-before.json"))
    }
    "Official" { Require-Stages @("Preflight"); Invoke-DurableStage "Official" { Invoke-Official } @((Join-Path $WorkRoot "paired-official"), (Join-Path $WorkRoot "traces/official")) }
    "Lightweight" { Require-Stages @("Preflight", "Official"); Invoke-DurableStage "Lightweight" { Invoke-Lightweight } @((Join-Path $WorkRoot "lightweight"), (Join-Path $WorkRoot "traces/lightweight"), (Join-Path $WorkRoot "profiles")) }
    "Score" { Require-Stages @("Preflight", "Official", "Lightweight"); Invoke-DurableStage "Score" { Invoke-Score } @((Join-Path $WorkRoot "results")) }
    "Compare" { Require-Stages @("Preflight", "Official", "Lightweight", "Score"); Invoke-DurableStage "Compare" { Invoke-Compare } @((Join-Path $WorkRoot "comparison")) }
    "Decide" { Require-Stages @("Preflight", "Official", "Lightweight", "Score", "Compare"); Invoke-DurableStage "Decide" { Invoke-Decide } @((Join-Path $Task5Root "results"), (Join-Path $Task5Root "comparison"), (Join-Path $Task5Root "selected-attempt.json"), (Join-Path $AttemptRoot "snapshot-after.json")); Complete-Receipt }
    "All" {
      Invoke-Preflight
      Invoke-DurableStage "Preflight" { Add-InternalCommandRecord "Preflight" "preflight-complete" $Benchmark } @($ManifestPath, (Join-Path $AttemptRoot "snapshot-before.json"))
      Invoke-DurableStage "Official" { Invoke-Official } @((Join-Path $WorkRoot "paired-official"), (Join-Path $WorkRoot "traces/official"))
      Invoke-DurableStage "Lightweight" { Invoke-Lightweight } @((Join-Path $WorkRoot "lightweight"), (Join-Path $WorkRoot "traces/lightweight"), (Join-Path $WorkRoot "profiles"))
      Invoke-DurableStage "Score" { Invoke-Score } @((Join-Path $WorkRoot "results"))
      Invoke-DurableStage "Compare" { Invoke-Compare } @((Join-Path $WorkRoot "comparison"))
      Invoke-DurableStage "Decide" { Invoke-Decide } @((Join-Path $Task5Root "results"), (Join-Path $Task5Root "comparison"), (Join-Path $Task5Root "selected-attempt.json"), (Join-Path $AttemptRoot "snapshot-after.json"))
      Complete-Receipt
    }
  }
} catch {
  if ($null -ne $StageStatePath -and (Test-Path -LiteralPath $StageStatePath -PathType Leaf)) {
    try {
      $failedState = Read-Json $StageStatePath
      if ($failedState.status -ne "invalid") {
        $failedState.status = "invalid"
        $failedState | Add-Member -Force -NotePropertyName failure -NotePropertyValue ([ordered]@{stage=$Stage;ended_at_utc=[DateTimeOffset]::UtcNow.ToString("o");error_sha256=Get-StringSha256 $_.Exception.Message;selected_marker_effective=$false})
        Write-AtomicJson $StageStatePath $failedState
      }
    } catch { }
  }
  throw
} finally {
  Pop-Location
}
