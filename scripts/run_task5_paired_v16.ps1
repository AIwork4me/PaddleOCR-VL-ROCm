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
  $CompactRoot = Join-Path $AttemptRoot "compact"
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
    Initialize-NativeJobRunner
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
    $wrapper = "import base64;exec(base64.b64decode('$encoded'))"
    $nativeArguments = @("-c", $wrapper) + @($Arguments)
    $rendered = (($nativeArguments | ForEach-Object { ConvertTo-NativeArgument ([string]$_) }) -join ' ')
    $job = [Task5NativeJob]::Run($PythonExe, $rendered, $RepoRoot, 30000, 5000)
    if ($job.ExitCode -ne 0 -or $job.TimedOut -or $job.NormalExitOrphan -or $job.FinalActiveCount -ne 0) { throw "strict helper failed: $(Get-StringSha256 ([string]$job.Stderr))" }
    return ([string]$job.Stdout).Trim()
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
      survivor_pids=@(); job_active_count=0; termination_result="not-required"; orphan_audit="PASS"; log_path=$relativeLog; log_sha256=Get-Sha256 $logPath
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
    if ([string]::IsNullOrWhiteSpace($Value)) { return [string]$Value }
    $sensitiveValues = [Collections.Generic.List[string]]::new()
    function Add-SensitiveLeaves([object]$InputValue) {
      if ($null -eq $InputValue) { return }
      if ($InputValue -is [string]) {
        if ($InputValue.Length -gt 0) { $sensitiveValues.Add([string]$InputValue) }
        return
      }
      if ($InputValue -is [Collections.IDictionary]) {
        foreach ($entry in $InputValue.GetEnumerator()) { Add-SensitiveLeaves $entry.Value }
        return
      }
      if ($InputValue -is [pscustomobject]) {
        foreach ($property in $InputValue.PSObject.Properties) { Add-SensitiveLeaves $property.Value }
        return
      }
      if ($InputValue -is [Collections.IEnumerable]) { foreach ($item in $InputValue) { Add-SensitiveLeaves $item } }
    }
    function Protect-StructuredValue([object]$InputValue, [string]$Key = "") {
      $normalized = $Key.ToLowerInvariant() -replace '[^a-z0-9]', ''
      $isSensitive = $normalized -match '(credential|authorization|bearer|apikey|token|signature|signedurl|prompt|payload|rawresult)'
      if ($isSensitive) {
        Add-SensitiveLeaves $InputValue
        if ($normalized -match 'prompt') { return "<prompt-redacted>" }
        if ($normalized -match 'payload') { return "<payload-redacted>" }
        if ($normalized -match 'rawresult') { return "<raw-result-redacted>" }
        return "<redacted>"
      }
      if ($InputValue -is [Collections.IDictionary]) {
        $result = [ordered]@{}
        foreach ($entry in $InputValue.GetEnumerator()) { $result[[string]$entry.Key] = Protect-StructuredValue $entry.Value ([string]$entry.Key) }
        return $result
      }
      if ($InputValue -is [pscustomobject]) {
        $result = [ordered]@{}
        foreach ($property in $InputValue.PSObject.Properties) { $result[$property.Name] = Protect-StructuredValue $property.Value $property.Name }
        return $result
      }
      if ($InputValue -is [Collections.IEnumerable] -and $InputValue -isnot [string]) {
        return @($InputValue | ForEach-Object { Protect-StructuredValue $_ "" })
      }
      if ($InputValue -is [string]) { return Protect-FreeText ([string]$InputValue) }
      return $InputValue
    }
    function Protect-FreeText([string]$Text) {
      $redacted = $Text
      $redacted = [regex]::Replace($redacted, '(?i)(["'']?prompt["'']?\s*[:=]\s*)(?:"[^"\r\n]*"|''[^''\r\n]*''|[^\s,;&]+)', '${1}"<prompt-redacted>"')
      $redacted = [regex]::Replace($redacted, '(?i)(["'']?payload["'']?\s*[:=]\s*)(?:"[^"\r\n]*"|''[^''\r\n]*''|[^\s,;&]+)', '${1}"<payload-redacted>"')
      $redacted = [regex]::Replace($redacted, '(?i)(["'']?raw[ _-]?result["'']?\s*[:=]\s*)(?:"[^"\r\n]*"|''[^''\r\n]*''|[^\s,;&]+)', '${1}"<raw-result-redacted>"')
      $redacted = [regex]::Replace($redacted, '(?i)(["'']?(?:api[ _-]?key|token|[^\s=:"'']*signature|credential|authorization|bearer)["'']?\s*[:=]\s*)(?:"[^"\r\n]*"|''[^''\r\n]*''|[^\s,;&]+)', '${1}"<redacted>"')
      $redacted = [regex]::Replace($redacted, '(?im)(Authorization\s*:\s*)[^\r\n]+', '${1}<redacted>')
      $redacted = [regex]::Replace($redacted, '(?i)Bearer\s+[^\s,;"''\]&]+', 'Bearer <redacted>')
      $redacted = [regex]::Replace($redacted, '(?i)\b(api[ _-]?key|token|[^\s=:]*signature|credential|authorization)\s*[=:]\s*[^\s,;"''\]&]+', '${1}=<redacted>')
      $redacted = [regex]::Replace($redacted, '(?i)([?&][^=]*(?:signature|token|api[ _-]?key|credential|authorization)[^=]*=)[^&\s]+', '${1}<redacted>')
      $redacted = [regex]::Replace($redacted, '(?i)\bprompt\s*[=:]\s*[^\r\n]+', 'prompt=<prompt-redacted>')
      $redacted = [regex]::Replace($redacted, '(?i)\bpayload\s*[=:]\s*[^\r\n]+', 'payload=<payload-redacted>')
      $redacted = [regex]::Replace($redacted, '(?i)\braw[ _-]?result\s*[=:]\s*[^\r\n]+', 'raw_result=<raw-result-redacted>')
      $redacted = [regex]::Replace($redacted, '(?i)(?<![A-Za-z0-9_])[A-Z]:\\[^\r\n\t"'']+', '<absolute-model-path>')
      return $redacted
    }
    function Protect-JsonFragments([string]$Text) {
      try {
        $structured = $Text | ConvertFrom-Json -ErrorAction Stop
        if ($null -eq $structured) { return "null" }
        return (Protect-StructuredValue $structured | ConvertTo-Json -Compress -Depth 30)
      } catch { }
      $output = [Text.StringBuilder]::new(); $cursor = 0
      while ($cursor -lt $Text.Length) {
        $start = -1
        for ($i = $cursor; $i -lt $Text.Length; $i++) { if ($Text[$i] -eq '{' -or $Text[$i] -eq '[') { $start = $i; break } }
        if ($start -lt 0) { [void]$output.Append($Text.Substring($cursor)); break }
        [void]$output.Append($Text.Substring($cursor, $start - $cursor))
        $stack = [Collections.Generic.Stack[char]]::new(); $inString = $false; $escaped = $false; $end = -1
        for ($i = $start; $i -lt $Text.Length; $i++) {
          $character = $Text[$i]
          if ($inString) {
            if ($escaped) { $escaped = $false }
            elseif ($character -eq '\') { $escaped = $true }
            elseif ($character -eq '"') { $inString = $false }
            continue
          }
          if ($character -eq '"') { $inString = $true; continue }
          if ($character -eq '{') { $stack.Push('}') }
          elseif ($character -eq '[') { $stack.Push(']') }
          elseif ($character -eq '}' -or $character -eq ']') {
            if ($stack.Count -eq 0 -or $stack.Pop() -ne $character) { break }
            if ($stack.Count -eq 0) { $end = $i; break }
          }
        }
        if ($end -lt 0) { [void]$output.Append($Text[$start]); $cursor = $start + 1; continue }
        $candidate = $Text.Substring($start, $end - $start + 1)
        try {
          $structured = $candidate | ConvertFrom-Json -ErrorAction Stop
          [void]$output.Append((Protect-StructuredValue $structured | ConvertTo-Json -Compress -Depth 30))
        } catch { [void]$output.Append($candidate) }
        $cursor = $end + 1
      }
      return Protect-FreeText $output.ToString()
    }
    $protected = Protect-JsonFragments $Value
    foreach ($secret in @($sensitiveValues | Sort-Object Length -Descending -Unique)) {
      if ($secret.Length -gt 0) { $protected = $protected.Replace($secret, '<redacted>') }
    }
    $protected = $protected.Replace('\u003c', '<').Replace('\u003e', '>')
    return Protect-FreeText $protected
  }

  function Initialize-NativeJobRunner {
    if ("Task5NativeJob" -as [type]) { return }
    Add-Type -Language CSharp -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

public sealed class Task5JobResult {
  public int RootPid { get; set; }
  public string Stdout { get; set; }
  public string Stderr { get; set; }
  public int ExitCode { get; set; }
  public bool TimedOut { get; set; }
  public bool NormalExitOrphan { get; set; }
  public int[] DescendantPids { get; set; }
  public int[] SurvivorPids { get; set; }
  public int FinalActiveCount { get; set; }
  public bool TerminationAttempted { get; set; }
}

public static class Task5NativeJob {
  const uint CREATE_SUSPENDED=0x4, CREATE_NO_WINDOW=0x08000000, STARTF_USESTDHANDLES=0x100;
  const uint HANDLE_FLAG_INHERIT=1, WAIT_OBJECT_0=0, WAIT_TIMEOUT=258;
  const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE=0x2000;
  const int JobObjectBasicProcessIdList=3, JobObjectExtendedLimitInformation=9;
  const uint TH32CS_SNAPPROCESS=2, PROCESS_TERMINATE=1, SYNCHRONIZE=0x00100000;

  [StructLayout(LayoutKind.Sequential)] struct SECURITY_ATTRIBUTES { public int nLength; public IntPtr lpSecurityDescriptor; public int bInheritHandle; }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] struct STARTUPINFO { public int cb; public string lpReserved,lpDesktop,lpTitle; public int dwX,dwY,dwXSize,dwYSize,dwXCountChars,dwYCountChars,dwFillAttribute; public uint dwFlags; public short wShowWindow,cbReserved2; public IntPtr lpReserved2,hStdInput,hStdOutput,hStdError; }
  [StructLayout(LayoutKind.Sequential)] struct PROCESS_INFORMATION { public IntPtr hProcess,hThread; public int dwProcessId,dwThreadId; }
  [StructLayout(LayoutKind.Sequential)] struct IO_COUNTERS { public ulong ReadOperationCount,WriteOperationCount,OtherOperationCount,ReadTransferCount,WriteTransferCount,OtherTransferCount; }
  [StructLayout(LayoutKind.Sequential)] struct BASIC_LIMIT { public long PerProcessUserTimeLimit,PerJobUserTimeLimit; public uint LimitFlags; public UIntPtr MinimumWorkingSetSize,MaximumWorkingSetSize; public uint ActiveProcessLimit; public UIntPtr Affinity; public uint PriorityClass,SchedulingClass; }
  [StructLayout(LayoutKind.Sequential)] struct EXTENDED_LIMIT { public BASIC_LIMIT BasicLimitInformation; public IO_COUNTERS IoInfo; public UIntPtr ProcessMemoryLimit,JobMemoryLimit,PeakProcessMemoryUsed,PeakJobMemoryUsed; }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Auto)] struct PROCESSENTRY32 { public uint dwSize,cntUsage,th32ProcessID; public IntPtr th32DefaultHeapID; public uint th32ModuleID,cntThreads,th32ParentProcessID; public int pcPriClassBase; public uint dwFlags; [MarshalAs(UnmanagedType.ByValTStr,SizeConst=260)] public string szExeFile; }

  [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern bool CreateProcessW(string app,StringBuilder cmd,IntPtr pa,IntPtr ta,bool inherit,uint flags,IntPtr env,string cwd,ref STARTUPINFO si,out PROCESS_INFORMATION pi);
  [DllImport("kernel32.dll",SetLastError=true)] static extern bool CreatePipe(out IntPtr read,out IntPtr write,ref SECURITY_ATTRIBUTES sa,int size);
  [DllImport("kernel32.dll",SetLastError=true)] static extern bool SetHandleInformation(IntPtr h,uint mask,uint flags);
  [DllImport("kernel32.dll",SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attrs,string name);
  [DllImport("kernel32.dll",SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job,int cls,IntPtr info,uint len);
  [DllImport("kernel32.dll",SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job,IntPtr process);
  [DllImport("kernel32.dll",SetLastError=true)] static extern bool QueryInformationJobObject(IntPtr job,int cls,IntPtr info,uint len,out uint ret);
  [DllImport("kernel32.dll",SetLastError=true)] static extern uint ResumeThread(IntPtr thread);
  [DllImport("kernel32.dll",SetLastError=true)] static extern uint WaitForSingleObject(IntPtr h,uint ms);
  [DllImport("kernel32.dll",SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr h,out uint code);
  [DllImport("kernel32.dll",SetLastError=true)] static extern bool TerminateJobObject(IntPtr job,uint code);
  [DllImport("kernel32.dll",SetLastError=true)] static extern IntPtr OpenProcess(uint access,bool inherit,int pid);
  [DllImport("kernel32.dll",SetLastError=true)] static extern bool TerminateProcess(IntPtr process,uint code);
  [DllImport("kernel32.dll",SetLastError=true)] static extern bool CloseHandle(IntPtr h);
  [DllImport("kernel32.dll",SetLastError=true)] static extern IntPtr CreateToolhelp32Snapshot(uint flags,uint pid);
  [DllImport("kernel32.dll",CharSet=CharSet.Auto)] static extern bool Process32First(IntPtr snap,ref PROCESSENTRY32 entry);
  [DllImport("kernel32.dll",CharSet=CharSet.Auto)] static extern bool Process32Next(IntPtr snap,ref PROCESSENTRY32 entry);

  static Exception Win32(string op) { return new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(),op); }
  static int[] Members(IntPtr job) {
    int cap=4096, header=8; IntPtr mem=Marshal.AllocHGlobal(header+cap*IntPtr.Size);
    try { uint ret; if(!QueryInformationJobObject(job,JobObjectBasicProcessIdList,mem,(uint)(header+cap*IntPtr.Size),out ret)) throw Win32("QueryInformationJobObject"); int count=Marshal.ReadInt32(mem,4); if(count<0||count>cap) throw new InvalidOperationException("Job member audit capacity exceeded"); var ids=new List<int>(); for(int i=0;i<count;i++){ long v=Marshal.ReadIntPtr(mem,header+i*IntPtr.Size).ToInt64(); if(v>0 && v<=int.MaxValue) ids.Add((int)v); } return ids.Distinct().ToArray(); }
    finally { Marshal.FreeHGlobal(mem); }
  }
  static int[] LiveMembers(IntPtr job) { var live=new List<int>(); foreach(int pid in Members(job)){ IntPtr h=OpenProcess(SYNCHRONIZE,false,pid); if(h==IntPtr.Zero){ if(Marshal.GetLastWin32Error()!=87) live.Add(pid); continue; } try { if(WaitForSingleObject(h,0)==WAIT_TIMEOUT) live.Add(pid); } finally { CloseHandle(h); } } return live.ToArray(); }
  static Dictionary<int,int> Parents() {
    var map=new Dictionary<int,int>(); IntPtr snap=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0); if(snap==new IntPtr(-1)) return map;
    try { var e=new PROCESSENTRY32(); e.dwSize=(uint)Marshal.SizeOf(typeof(PROCESSENTRY32)); if(Process32First(snap,ref e)){ do { map[(int)e.th32ProcessID]=(int)e.th32ParentProcessID; } while(Process32Next(snap,ref e)); } return map; } finally { CloseHandle(snap); }
  }
  static int Depth(int pid,HashSet<int> members,Dictionary<int,int> parents) { int d=0,cur=pid; var seen=new HashSet<int>(); while(seen.Add(cur) && parents.ContainsKey(cur) && members.Contains(parents[cur])){ d++; cur=parents[cur]; } return d; }
  static void DeepestFirstTerminate(IntPtr job,int root) { var ids=LiveMembers(job); var set=new HashSet<int>(ids); var parents=Parents(); foreach(int pid in ids.Where(x=>x!=root).OrderByDescending(x=>Depth(x,set,parents))){ IntPtr h=OpenProcess(PROCESS_TERMINATE,false,pid); if(h!=IntPtr.Zero){ try { TerminateProcess(h,239); } finally { CloseHandle(h); } } } }
  static Task<string> ReadAsync(IntPtr handle) { var safe=new SafeFileHandle(handle,true); var stream=new FileStream(safe,FileAccess.Read,4096,false); var reader=new StreamReader(stream,new UTF8Encoding(false),true,4096); return Task.Run(()=>{ using(reader) return reader.ReadToEnd(); }); }

  public static Task5JobResult Run(string executable,string arguments,string cwd,long timeoutMs,int graceMs) {
    return Run(executable,arguments,cwd,timeoutMs,graceMs,false);
  }

  public static Task5JobResult Run(string executable,string arguments,string cwd,long timeoutMs,int graceMs,bool forceAssignFailure) {
    IntPtr job=IntPtr.Zero,outRead=IntPtr.Zero,outWrite=IntPtr.Zero,errRead=IntPtr.Zero,errWrite=IntPtr.Zero; PROCESS_INFORMATION pi=new PROCESS_INFORMATION(); bool created=false;
    var result=new Task5JobResult { Stdout="",Stderr="",ExitCode=-1,DescendantPids=new int[0],SurvivorPids=new int[0] };
    try {
      job=CreateJobObject(IntPtr.Zero,null); if(job==IntPtr.Zero) throw Win32("CreateJobObject");
      var limit=new EXTENDED_LIMIT(); limit.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; int size=Marshal.SizeOf(typeof(EXTENDED_LIMIT)); IntPtr lim=Marshal.AllocHGlobal(size); try { Marshal.StructureToPtr(limit,lim,false); if(!SetInformationJobObject(job,JobObjectExtendedLimitInformation,lim,(uint)size)) throw Win32("SetInformationJobObject"); } finally { Marshal.FreeHGlobal(lim); }
      var sa=new SECURITY_ATTRIBUTES{nLength=Marshal.SizeOf(typeof(SECURITY_ATTRIBUTES)),bInheritHandle=1}; if(!CreatePipe(out outRead,out outWrite,ref sa,0)||!CreatePipe(out errRead,out errWrite,ref sa,0)) throw Win32("CreatePipe"); if(!SetHandleInformation(outRead,HANDLE_FLAG_INHERIT,0)||!SetHandleInformation(errRead,HANDLE_FLAG_INHERIT,0)) throw Win32("SetHandleInformation");
      var si=new STARTUPINFO(); si.cb=Marshal.SizeOf(typeof(STARTUPINFO)); si.dwFlags=STARTF_USESTDHANDLES; si.hStdOutput=outWrite; si.hStdError=errWrite; si.hStdInput=IntPtr.Zero;
      var cmd=new StringBuilder("\""+executable+"\""+(String.IsNullOrEmpty(arguments)?"":" "+arguments)); if(!CreateProcessW(executable,cmd,IntPtr.Zero,IntPtr.Zero,true,CREATE_SUSPENDED|CREATE_NO_WINDOW,IntPtr.Zero,cwd,ref si,out pi)) throw Win32("CreateProcessW"); created=true; result.RootPid=pi.dwProcessId;
      bool assigned=AssignProcessToJobObject(forceAssignFailure?IntPtr.Zero:job,pi.hProcess); if(!assigned){ int assignError=Marshal.GetLastWin32Error(); bool terminated=TerminateProcess(pi.hProcess,239); int terminateError=Marshal.GetLastWin32Error(); uint waited=WaitForSingleObject(pi.hProcess,(uint)Math.Max(1000,graceMs)); if(!terminated) throw new System.ComponentModel.Win32Exception(terminateError,"TerminateProcess after AssignProcessToJobObject failure"); if(waited!=WAIT_OBJECT_0) throw new InvalidOperationException("Wait after AssignProcessToJobObject failure did not observe process termination"); throw new System.ComponentModel.Win32Exception(assignError,"AssignProcessToJobObject"); } var stdout=ReadAsync(outRead); outRead=IntPtr.Zero; var stderr=ReadAsync(errRead); errRead=IntPtr.Zero; CloseHandle(outWrite); outWrite=IntPtr.Zero; CloseHandle(errWrite); errWrite=IntPtr.Zero; if(ResumeThread(pi.hThread)==UInt32.MaxValue) throw Win32("ResumeThread"); CloseHandle(pi.hThread); pi.hThread=IntPtr.Zero;
      var observed=new HashSet<int>(); var sw=Stopwatch.StartNew(); bool rootExited=false; while(sw.ElapsedMilliseconds<timeoutMs){ foreach(var id in Members(job)) observed.Add(id); if(WaitForSingleObject(pi.hProcess,100)==WAIT_OBJECT_0){rootExited=true;break;} }
      foreach(var id in Members(job)) observed.Add(id); int effectiveRoot=result.RootPid;
      foreach(var id in Members(job)) observed.Add(id); result.TimedOut=!rootExited;
      if(rootExited){ var settle=Stopwatch.StartNew(); while(settle.ElapsedMilliseconds<500 && LiveMembers(job).Any(id=>id!=result.RootPid&&id!=effectiveRoot)) System.Threading.Thread.Sleep(25); }
      var active=LiveMembers(job); result.NormalExitOrphan=rootExited && active.Any(id=>id!=result.RootPid&&id!=effectiveRoot); result.TerminationAttempted=result.TimedOut||result.NormalExitOrphan;
      if(result.TerminationAttempted){ DeepestFirstTerminate(job,effectiveRoot); TerminateJobObject(job,239); var stop=Stopwatch.StartNew(); while(stop.ElapsedMilliseconds<graceMs && LiveMembers(job).Length>0) System.Threading.Thread.Sleep(50); }
      active=LiveMembers(job); result.FinalActiveCount=active.Length; result.SurvivorPids=active.Where(id=>id!=result.RootPid).Distinct().OrderBy(x=>x).ToArray(); result.DescendantPids=observed.Where(id=>id!=result.RootPid).Distinct().OrderBy(x=>x).ToArray(); uint exit; if(GetExitCodeProcess(pi.hProcess,out exit)) result.ExitCode=unchecked((int)exit);
      if(!Task.WaitAll(new Task[]{stdout,stderr},Math.Max(1000,graceMs))) { result.Stdout="<raw-result-redacted: stream remained open>"; result.Stderr="<raw-result-redacted: stream remained open>"; } else { result.Stdout=stdout.Result; result.Stderr=stderr.Result; }
      return result;
    } finally { if(created && pi.hThread!=IntPtr.Zero) CloseHandle(pi.hThread); if(created && pi.hProcess!=IntPtr.Zero) CloseHandle(pi.hProcess); if(outRead!=IntPtr.Zero) CloseHandle(outRead); if(outWrite!=IntPtr.Zero) CloseHandle(outWrite); if(errRead!=IntPtr.Zero) CloseHandle(errRead); if(errWrite!=IntPtr.Zero) CloseHandle(errWrite); if(job!=IntPtr.Zero) CloseHandle(job); }
  }
}
'@
  }

  function Invoke-LoggedNative([string]$StageName, [string]$Name, [string]$Executable, [string[]]$Arguments, [switch]$Capture) {
    $startedOffset = [DateTimeOffset]::UtcNow
    $started = $startedOffset.ToString("o")
    Initialize-NativeJobRunner
    $nativeArguments = (($Arguments | ForEach-Object { ConvertTo-NativeArgument ([string]$_) }) -join ' ')
    $job = [Task5NativeJob]::Run($Executable, $nativeArguments, $RepoRoot, ([long]$CommandTimeoutSeconds * 1000), ([int]$TerminationGraceSeconds * 1000))
    $stdout = [string]$job.Stdout
    $stderr = [string]$job.Stderr
    $exitCode = [int]$job.ExitCode
    $timedOut = [bool]$job.TimedOut
    if ($timedOut) { $exitCode = -1 }
    $descendantPids = @($job.DescendantPids | ForEach-Object { [int]$_ } | Sort-Object -Unique)
    $survivorPids = @($job.SurvivorPids | ForEach-Object { [int]$_ } | Sort-Object -Unique)
    $terminationResult = $(if (-not $job.TerminationAttempted) { $(if ($descendantPids.Count -eq 0) { "not-required" } else { "completed" }) } elseif ($job.FinalActiveCount -ne 0) { "survivors" } elseif ($descendantPids.Count -gt 0) { "terminated" } else { "root-terminated" })
    $rendered = "[stdout]`n$(Protect-LoggedText $stdout)`n[stderr]`n$(Protect-LoggedText $stderr)"
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
      descendant_pids = $descendantPids
      survivor_pids = $survivorPids
      job_active_count = [int]$job.FinalActiveCount
      termination_result = $terminationResult
      orphan_audit = $(if ($job.FinalActiveCount -eq 0 -and $survivorPids.Count -eq 0) { "PASS" } else { "FAIL" })
      log_path = $relativeLog
      log_sha256 = Get-Sha256 $logPath
    })
    if ($job.FinalActiveCount -ne 0 -or $survivorPids.Count -ne 0) { throw "orphan process audit failed for command: $Name" }
    if ($job.NormalExitOrphan) { throw "orphan process observed after normal command exit: $Name" }
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
    $serverCode = 'import hashlib,json,sys,urllib.request; u=sys.argv[1].rstrip("/")+"/models"; v=json.load(urllib.request.urlopen(u,timeout=10),object_pairs_hook=lambda p:dict(p) if len(p)==len(dict(p)) else (_ for _ in ()).throw(ValueError("duplicate JSON key")),parse_constant=lambda x:(_ for _ in ()).throw(ValueError("non-finite JSON constant"))); c=json.dumps(v,ensure_ascii=False,separators=(",",":"),sort_keys=True,allow_nan=False).encode(); print(json.dumps({"models_sha256":hashlib.sha256(c).hexdigest()},sort_keys=True,separators=(",",":")))'
    try { $server = Invoke-DirectPython $serverCode @($ServerUrl) } catch { throw "Server model runtime identity failed" }
    $gpus = @(
      Get-CimInstance Win32_VideoController -ErrorAction Stop |
        ForEach-Object { [ordered]@{ Name=[string]$_.Name; PNPDeviceID=[string]$_.PNPDeviceID; DriverVersion=[string]$_.DriverVersion } } |
        Sort-Object PNPDeviceID
    )
    if ($gpus.Count -eq 0) { throw "GPU environment identity is empty" }
    foreach ($gpu in $gpus) {
      if ([string]::IsNullOrWhiteSpace($gpu.Name) -or [string]::IsNullOrWhiteSpace($gpu.PNPDeviceID) -or [string]::IsNullOrWhiteSpace($gpu.DriverVersion)) {
        throw "GPU environment identity contains an empty field"
      }
    }
    $serverIdentity = [ordered]@{
      models_sha256 = [string]($server | ConvertFrom-Json).models_sha256
      requested_model = "<redacted>"
      requested_model_sha256 = Get-StringSha256 ($ApiModelName.Trim())
    }
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
      server_model_runtime = $serverIdentity
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
expected={"name","executable_sha256","arguments_sha256","started_at_utc","ended_at_utc","exit_code","timed_out","descendant_pids","survivor_pids","job_active_count","termination_result","orphan_audit","log_path","log_sha256"}
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
    for key in ("descendant_pids","survivor_pids"):
        if not isinstance(row[key],list) or any(type(v) is not int or v<=0 for v in row[key]) or len(set(row[key]))!=len(row[key]): raise ValueError("invalid PID list")
    if set(row["descendant_pids"]) & set(row["survivor_pids"]): raise ValueError("duplicate PID across lists")
    if type(row["job_active_count"]) is not int or row["job_active_count"]<0: raise ValueError("invalid job active count")
    if row["termination_result"] not in {"not-required","completed","terminated","root-terminated","survivors"}: raise ValueError("invalid termination result")
    if row["orphan_audit"] not in {"PASS","FAIL"}: raise ValueError("invalid orphan audit")
    descendants=bool(row["descendant_pids"]); survivors=bool(row["survivor_pids"]) or row["job_active_count"]!=0
    if row["orphan_audit"]=="PASS" and survivors: raise ValueError("false orphan PASS")
    if row["termination_result"]=="not-required" and (descendants or row["timed_out"]): raise ValueError("contradictory not-required")
    if row["termination_result"]=="completed" and (not descendants or row["timed_out"] or survivors): raise ValueError("contradictory completed descendants")
    if row["termination_result"]=="terminated" and (not descendants or survivors): raise ValueError("contradictory terminated")
    if row["termination_result"]=="root-terminated" and (descendants or not row["timed_out"] or survivors): raise ValueError("contradictory root termination")
    if row["termination_result"]=="survivors" and (not survivors or row["orphan_audit"]!="FAIL"): raise ValueError("contradictory survivors")
    if row["timed_out"] and (row["exit_code"]==0 or row["termination_result"]=="not-required"): raise ValueError("contradictory timeout")
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
    if (@($state.stages.PSObject.Properties | ForEach-Object Name) -contains $StageName) { throw "Stage already completed: $StageName" }
    $started = [DateTimeOffset]::UtcNow.ToString("o")
    try {
      & $Body
      $map = Get-OutputMap $OutputRoots
      $commandPath = Join-Path $CommandRoot "$($StageName.ToLowerInvariant()).jsonl"
      if (-not (Test-Path -LiteralPath $commandPath -PathType Leaf)) { throw "Stage command log is missing" }
      Assert-CommandLogIntegrity $commandPath
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

  function Initialize-PreflightAttempt {
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
  }

  function Invoke-PreflightBusiness {
    $snapshot = Invoke-LoggedNative "Preflight" "g0-snapshot-before" $PythonExe @("-m", "eval.task5_manifest", "snapshot", "--r7-root", $R7Root, "--receipt", $G0Receipt) -Capture
    Write-AtomicText (Join-Path $AttemptRoot "snapshot-before.json") ($snapshot + [Environment]::NewLine)
    Invoke-LoggedNative "Preflight" "python-auth" $PythonExe @("--version")
    Invoke-LoggedNative "Preflight" "scorer-python-auth" $ScorerPythonExe @("--version")
    Invoke-LoggedNative "Preflight" "scorer-contract" $PythonExe @("-m", "eval.benchmark_contract", "--checkout", (Join-Path $RepoRoot "eval/.omnidocbench"))
    Invoke-LoggedNative "Preflight" "scorer-preflight" $ScorerPythonExe @("scripts/check_omnidocbench_scorer.py", "--checkout", (Join-Path $RepoRoot "eval/.omnidocbench"), "--direct-lock", (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16.txt"), "--transitive-lock", (Join-Path $RepoRoot "eval/requirements-omnidocbench-v16-transitive.txt"), "--require-cdm-tools")
    Invoke-LoggedNative "Preflight" "server" $PythonExe @("scripts/check_server.py", "--server-url", $ServerUrl)
    Invoke-LoggedNative "Preflight" "official-constructor" $PythonExe @("scripts/check_official_paddleocr.py", "--construct", "--server-url", $ServerUrl, "--api-model-name", $ApiModelName)
  }

  function Invoke-PreflightStage {
    Initialize-PreflightAttempt
    Invoke-DurableStage "Preflight" { Invoke-PreflightBusiness; Add-InternalCommandRecord "Preflight" "preflight-complete" $Benchmark } @($ManifestPath, (Join-Path $AttemptRoot "snapshot-before.json"))
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
    $results = Join-Path $CompactRoot "results/$Engine"
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
    $comparison = Join-Path $CompactRoot "comparison"
    New-Item -ItemType Directory -Force -Path $comparison | Out-Null
    $inline = 'import json,sys; from pathlib import Path; from eval.task5_comparison import compare_prediction_dirs; p=compare_prediction_dirs(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]); Path(sys.argv[4]).write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")'
    Invoke-LoggedNative "Compare" "normalized-output" $PythonExe @("-c", $inline, (Join-Path $WorkRoot "paired-official"), (Join-Path $WorkRoot "lightweight"), $ApprovedExcludedStem, (Join-Path $comparison "normalized-output.json"))
    Invoke-LoggedNative "Compare" "trace-diff" $PythonExe @("scripts/compare_inference_traces.py", (Join-Path $WorkRoot "traces/official"), (Join-Path $WorkRoot "traces/lightweight"), "--output", (Join-Path $comparison "trace-diff.json"))
    $contract = [ordered]@{ benchmark=$Benchmark; pages=$ExpectedPages; paired_pages=$ExpectedPairs; approved_exclusion=$ApprovedExcludedStem; formula="CDM"; table="TEDS" }
    Write-AtomicJson (Join-Path $comparison "input-contract.json") $contract
    Add-InternalCommandRecord "Compare" "input-contract" ($contract | ConvertTo-Json -Compress)
  }

  function Assert-ExactFileSet([string]$Directory, [string[]]$Expected) {
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) { throw "Compact evidence directory is missing" }
    Assert-NoReparsePoint $Directory "Compact evidence directory"
    $actual = @(Get-ChildItem -Force -LiteralPath $Directory -File | ForEach-Object Name | Sort-Object)
    $wanted = @($Expected | Sort-Object)
    if (($actual -join "`n") -cne ($wanted -join "`n")) { throw "Compact evidence exact filename set mismatch" }
    if (@(Get-ChildItem -Force -LiteralPath $Directory -Directory).Count -ne 0) { throw "Compact evidence topology contains an unexpected subdirectory" }
  }

  function Assert-ExactChildDirectories([string]$Directory, [string[]]$Expected) {
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) { throw "Compact evidence topology directory is missing" }
    Assert-NoReparsePoint $Directory "Compact evidence topology directory"
    if (@(Get-ChildItem -Force -LiteralPath $Directory -File).Count -ne 0) { throw "Compact evidence topology contains an unexpected file" }
    $actual = @(Get-ChildItem -Force -LiteralPath $Directory -Directory | ForEach-Object Name | Sort-Object)
    $wanted = @($Expected | Sort-Object)
    if (($actual -join "`n") -cne ($wanted -join "`n")) { throw "Compact evidence topology directory set mismatch" }
  }

  function Assert-CompactEvidenceComplete {
    $resultFiles = @("metric.json", "metric-cdm.json", "run-summary.json", "run-summary-cdm.json", "provenance.json", "provenance-cdm.json")
    Assert-ExactChildDirectories $CompactRoot @("results", "comparison")
    Assert-ExactChildDirectories (Join-Path $CompactRoot "results") @("official", "lightweight")
    Assert-ExactFileSet (Join-Path $CompactRoot "results/official") $resultFiles
    Assert-ExactFileSet (Join-Path $CompactRoot "results/lightweight") $resultFiles
    Assert-ExactFileSet (Join-Path $CompactRoot "comparison") @("input-contract.json", "normalized-output.json", "trace-diff.json", "directml-attestation.json", "decision.json")
  }

  function Assert-NoRootCompactAuthority {
    foreach ($name in @("results", "comparison", "receipt.sha256.json")) {
      if (Test-Path -LiteralPath (Join-Path $Task5Root $name)) { throw "Forbidden root-level Task 5 evidence authority exists: $name" }
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
    $comparison = Join-Path $CompactRoot "comparison"
    $profiles = @(Get-ChildItem -LiteralPath (Join-Path $WorkRoot "profiles") -Filter "layout-profile*.json" -File)
    if ($profiles.Count -ne 1) { throw "missing profile" }
    $attestation = Invoke-LoggedNative "Decide" "directml-attestation" $PythonExe @("-m", "eval.directml_attestation", "--profile", $profiles[0].FullName, "--stats", (Join-Path $WorkRoot "lightweight/_run_stats.json"), "--allow-fail-verdict") -Capture
    $strictJson = 'import json,sys; seen=lambda pairs: (_ for _ in ()).throw(ValueError("duplicate JSON key")) if len(pairs)!=len(dict(pairs)) else dict(pairs); value=json.loads(sys.argv[1],parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON")),object_pairs_hook=seen); assert isinstance(value,dict)'
    Invoke-LoggedNative "Decide" "directml-strict-json" $PythonExe @("-c", $strictJson, $attestation)
    Write-AtomicText (Join-Path $comparison "directml-attestation.json") ($attestation + [Environment]::NewLine)
    $r = Join-Path $CompactRoot "results"
    Invoke-LoggedNative "Decide" "decision" $PythonExe @("-m", "eval.task5_decision", "decide", "--official-non-cdm", (Join-Path $r "official/metric.json"), "--official-cdm", (Join-Path $r "official/metric-cdm.json"), "--lightweight-non-cdm", (Join-Path $r "lightweight/metric.json"), "--lightweight-cdm", (Join-Path $r "lightweight/metric-cdm.json"), "--output-report", (Join-Path $comparison "normalized-output.json"), "--trace-report", (Join-Path $comparison "trace-diff.json"), "--provider-attestation", (Join-Path $comparison "directml-attestation.json"), "--lightweight-stats", (Join-Path $r "lightweight/run-summary.json"), "--public-contracts-pass", "--output", (Join-Path $comparison "decision.json"))
    $measuredDecision = Read-Json (Join-Path $comparison "decision.json")
    if ($measuredDecision.amd_adaptation.verdict -eq "PASS") { Assert-ProviderMajority (Read-Json (Join-Path $comparison "directml-attestation.json")) }
    $after = Invoke-LoggedNative "Decide" "g0-snapshot-after" $PythonExe @("-m", "eval.task5_manifest", "snapshot", "--r7-root", $R7Root, "--receipt", $G0Receipt) -Capture
    Write-AtomicText (Join-Path $AttemptRoot "snapshot-after.json") ($after + [Environment]::NewLine)
    $beforeObject = Read-Json (Join-Path $AttemptRoot "snapshot-before.json")
    $afterObject = Read-Json (Join-Path $AttemptRoot "snapshot-after.json")
    if (($beforeObject | ConvertTo-Json -Compress -Depth 20) -cne ($afterObject | ConvertTo-Json -Compress -Depth 20)) { throw "G0 integrity mismatch" }
    Assert-StageStartIntegrity (Read-State) "Decide" "manifest-revalidate-seal"
  }

  function Invoke-DecisionTool([string[]]$Arguments) {
    $code = 'import sys; from eval.task5_decision import main; raise SystemExit(main(sys.argv[1:]))'
    Invoke-DirectPython $code $Arguments | Out-Null
  }

  function Test-ByteEqual([string]$Left, [string]$Right) {
    if (-not (Test-Path -LiteralPath $Left -PathType Leaf) -or -not (Test-Path -LiteralPath $Right -PathType Leaf)) { return $false }
    $a = [IO.File]::ReadAllBytes($Left); $b = [IO.File]::ReadAllBytes($Right)
    if ($a.Length -ne $b.Length) { return $false }
    for ($i=0; $i -lt $a.Length; $i++) { if ($a[$i] -ne $b[$i]) { return $false } }
    return $true
  }

  function Write-CandidateAtomic {
    Assert-CompactEvidenceComplete
    $state = Read-State
    Assert-RecordedStagesIntegrity $state
    if ($state.status -ne "active") { throw "Attempt is not active at seal boundary" }
    $state.status = "sealed"
    Write-AtomicJson $StageStatePath $state
    $decision = Read-Json (Join-Path $CompactRoot "comparison/decision.json")
    $candidate = [ordered]@{schema=1;attempt_id=$AttemptId;manifest_sha256=Get-Sha256 $ManifestPath;strict_equivalence=$decision.strict_equivalence.verdict;amd_adaptation=$decision.amd_adaptation.verdict;g0_closure="PASS";effective_only_with_valid_receipt=$true}
    $candidatePath = Join-Path $AttemptRoot "selected-attempt.json"
    if (Test-Path -LiteralPath $candidatePath) { throw "Selection candidate already exists" }
    Write-AtomicJson $candidatePath $candidate
  }

  function Get-AttemptReceiptPaths {
    $base = "attempts/$AttemptId"
    $paths = @("manifest.json", "$base/stage-state.json", "$base/snapshot-before.json", "$base/snapshot-after.json", "$base/selected-attempt.json")
    foreach ($engine in @("official", "lightweight")) {
      foreach ($name in @("metric.json", "metric-cdm.json", "run-summary.json", "run-summary-cdm.json", "provenance.json", "provenance-cdm.json")) { $paths += "$base/compact/results/$engine/$name" }
    }
    foreach ($name in @("input-contract.json", "normalized-output.json", "trace-diff.json", "directml-attestation.json", "decision.json")) { $paths += "$base/compact/comparison/$name" }
    return @($paths | Sort-Object)
  }

  function New-TemporaryCandidatePointer {
    $candidate = Join-Path $AttemptRoot "selected-attempt.json"
    $temporary = Join-Path $Task5Root (".selected-attempt.{0}.validate.tmp" -f [Guid]::NewGuid().ToString("N"))
    [IO.File]::WriteAllBytes($temporary, [IO.File]::ReadAllBytes($candidate))
    return $temporary
  }

  function Validate-LocalSelection {
    $receipt = Join-Path $AttemptRoot "receipt.sha256.json"
    Invoke-DecisionTool @("validate-receipt", "--task5-root", $Task5Root, "--receipt", $receipt)
    $temporary = New-TemporaryCandidatePointer
    try { Invoke-DecisionTool @("validate-selection", "--task5-root", $Task5Root, "--pointer", $temporary) }
    finally { if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force } }
  }

  function Publish-RootSelection {
    $candidate = Join-Path $AttemptRoot "selected-attempt.json"
    $pointer = Join-Path $Task5Root "selected-attempt.json"
    if (Test-Path -LiteralPath $pointer) {
      if (-not (Test-ByteEqual $candidate $pointer)) { throw "A valid root selection already exists for another attempt" }
      Invoke-DecisionTool @("validate-selection", "--task5-root", $Task5Root, "--pointer", $pointer)
      return
    }
    $temporary = Join-Path $Task5Root (".selected-attempt.{0}.publish.tmp" -f [Guid]::NewGuid().ToString("N"))
    $publishedHere = $false
    try {
      $bytes = [IO.File]::ReadAllBytes($candidate)
      $stream = [IO.FileStream]::new($temporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None, 4096, [IO.FileOptions]::WriteThrough)
      try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
      Move-Item -LiteralPath $temporary -Destination $pointer
      $publishedHere = $true
      Invoke-DecisionTool @("validate-selection", "--task5-root", $Task5Root, "--pointer", $pointer)
    } catch {
      if ($publishedHere -and (Test-Path -LiteralPath $pointer) -and (Test-ByteEqual $candidate $pointer)) { Remove-Item -LiteralPath $pointer -Force }
      elseif ((Test-Path -LiteralPath $pointer) -and (Test-ByteEqual $candidate $pointer)) { Invoke-DecisionTool @("validate-selection", "--task5-root", $Task5Root, "--pointer", $pointer); return }
      throw
    } finally { if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force } }
  }

  function Complete-Receipt {
    $state = Read-State
    Assert-RecordedStagesIntegrity $state
    if ($state.status -ne "sealed") { throw "Receipt requires a sealed attempt" }
    $receipt = Join-Path $AttemptRoot "receipt.sha256.json"
    if (Test-Path -LiteralPath $receipt) { throw "receipt mutation or replacement is forbidden" }
    $paths = Get-AttemptReceiptPaths
    $args = @("-m", "eval.task5_decision", "receipt", "--task5-root", $Task5Root)
    foreach ($path in $paths) { $args += @("--path", $path) }
    $args += @("--output", $receipt)
    Invoke-DecisionTool $args[2..($args.Count-1)]
    Validate-LocalSelection
    Publish-RootSelection
    $decision = Read-Json (Join-Path $CompactRoot "comparison/decision.json")
    $strictVerdict = $decision.strict_equivalence.verdict
    $amdVerdict = $decision.amd_adaptation.verdict
    Write-Host "strict_equivalence=$strictVerdict"
    Write-Host "amd_adaptation=$amdVerdict"
  }

  function Resume-SealedAttemptSelection {
    $state = Read-Json $StageStatePath
    if ($state.status -ne "sealed") { throw "Existing attempt is not sealed; use a new AttemptId" }
    if (-not (Test-Path -LiteralPath (Join-Path $AttemptRoot "selected-attempt.json") -PathType Leaf) -or -not (Test-Path -LiteralPath (Join-Path $AttemptRoot "receipt.sha256.json") -PathType Leaf)) { throw "sealed attempt may only retry pointer publication after local receipt completion" }
    Validate-LocalSelection
    Publish-RootSelection
  }

  function Require-Stages([string[]]$Names) {
    $state = Read-State
    Assert-RecordedStagesIntegrity $state
    foreach ($name in $Names) {
      if (@($state.stages.PSObject.Properties | ForEach-Object Name) -notcontains $name) { throw "Missing completed predecessor: $name" }
    }
  }

  Assert-NoRootCompactAuthority
  $rootPointer = Join-Path $Task5Root "selected-attempt.json"
  if (Test-Path -LiteralPath $rootPointer -PathType Leaf) {
    Invoke-DecisionTool @("validate-selection", "--task5-root", $Task5Root, "--pointer", $rootPointer)
    $selectedRoot = Read-Json $rootPointer
    if ($selectedRoot.attempt_id -cne $AttemptId) { throw "A valid root selection already exists for another attempt" }
    Write-Host "selected_attempt=$AttemptId"
    return
  }
  if (Test-Path -LiteralPath $AttemptRoot -PathType Container) {
    $existingState = Read-Json $StageStatePath
    if ($existingState.status -eq "sealed") {
      Resume-SealedAttemptSelection
      Write-Host "selected_attempt=$AttemptId"
      return
    }
  }

  switch ($Stage) {
    "Preflight" {
      Invoke-PreflightStage
    }
    "Official" { Require-Stages @("Preflight"); Invoke-DurableStage "Official" { Invoke-Official } @((Join-Path $WorkRoot "paired-official"), (Join-Path $WorkRoot "traces/official")) }
    "Lightweight" { Require-Stages @("Preflight", "Official"); Invoke-DurableStage "Lightweight" { Invoke-Lightweight } @((Join-Path $WorkRoot "lightweight"), (Join-Path $WorkRoot "traces/lightweight"), (Join-Path $WorkRoot "profiles")) }
    "Score" { Require-Stages @("Preflight", "Official", "Lightweight"); Invoke-DurableStage "Score" { Invoke-Score } @((Join-Path $CompactRoot "results")) }
    "Compare" { Require-Stages @("Preflight", "Official", "Lightweight", "Score"); Invoke-DurableStage "Compare" { Invoke-Compare } @((Join-Path $CompactRoot "comparison/input-contract.json"), (Join-Path $CompactRoot "comparison/normalized-output.json"), (Join-Path $CompactRoot "comparison/trace-diff.json")) }
    "Decide" { Require-Stages @("Preflight", "Official", "Lightweight", "Score", "Compare"); Invoke-DurableStage "Decide" { Invoke-Decide } @((Join-Path $CompactRoot "comparison/directml-attestation.json"), (Join-Path $CompactRoot "comparison/decision.json"), (Join-Path $AttemptRoot "snapshot-after.json")); Write-CandidateAtomic; Complete-Receipt }
    "All" {
      Invoke-PreflightStage
      Invoke-DurableStage "Official" { Invoke-Official } @((Join-Path $WorkRoot "paired-official"), (Join-Path $WorkRoot "traces/official"))
      Invoke-DurableStage "Lightweight" { Invoke-Lightweight } @((Join-Path $WorkRoot "lightweight"), (Join-Path $WorkRoot "traces/lightweight"), (Join-Path $WorkRoot "profiles"))
      Invoke-DurableStage "Score" { Invoke-Score } @((Join-Path $CompactRoot "results"))
      Invoke-DurableStage "Compare" { Invoke-Compare } @((Join-Path $CompactRoot "comparison/input-contract.json"), (Join-Path $CompactRoot "comparison/normalized-output.json"), (Join-Path $CompactRoot "comparison/trace-diff.json"))
      Invoke-DurableStage "Decide" { Invoke-Decide } @((Join-Path $CompactRoot "comparison/directml-attestation.json"), (Join-Path $CompactRoot "comparison/decision.json"), (Join-Path $AttemptRoot "snapshot-after.json"))
      Write-CandidateAtomic
      Complete-Receipt
    }
  }
} catch {
  if ($null -ne $StageStatePath -and (Test-Path -LiteralPath $StageStatePath -PathType Leaf)) {
    try {
      $failedState = Read-Json $StageStatePath
      if ($failedState.status -ne "invalid" -and $failedState.status -ne "sealed") {
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
