# Runs illustrate.py on this (CPU-only) PC at BelowNormal priority so the wall
# screen stays smooth, and reports wall time + the worker's real peak RAM.
#
# Windows venv gotcha: .venv\Scripts\python.exe is only a launcher that spawns
# the real interpreter as a CHILD process. So priority must be set before
# launch (a BelowNormal parent's children inherit BelowNormal), and RAM must be
# read from the child, not the launcher.
param([string]$In, [string]$Out, [string]$Prompt, [int]$Seed = 23, [string]$Extra = "")
$env:HF_HUB_OFFLINE = "1"   # models are already local; skip HF network checks
$dir = $PSScriptRoot
[Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
$pyArgs = @("`"$dir\illustrate.py`"", "`"$In`"", "-o", "`"$Out`"", "--remove-bg", "--pastel",
            "--seed", $Seed, "--extra-prompt", "`"$Prompt`"") + ($Extra -split " " | Where-Object { $_ })
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath "$dir\.venv\Scripts\python.exe" -ArgumentList $pyArgs -PassThru -NoNewWindow `
     -RedirectStandardOutput "$dir\run.log" -RedirectStandardError "$dir\run.err"
$null = $p.Handle   # cache the handle, or ExitCode comes back empty after exit
# Two peaks: working set counts memory-mapped model files the OS can reclaim;
# private (PeakPagedMemorySize64 = peak commit) is what the process really holds.
$peakWS = 0; $peakPriv = 0; $prio = "?"
while (-not $p.HasExited) {
  foreach ($k in Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)") {
    $gp = Get-Process -Id $k.ProcessId -ErrorAction SilentlyContinue; $prio = $k.Priority
    if ($gp) {
      if ($gp.PeakWorkingSet64 -gt $peakWS) { $peakWS = $gp.PeakWorkingSet64 }
      if ($gp.PeakPagedMemorySize64 -gt $peakPriv) { $peakPriv = $gp.PeakPagedMemorySize64 }
    }
  }
  Start-Sleep -Seconds 1
}
"exit={0}  time={1:N0}s  peakPrivate={2:N2}GB  peakWorkingSet={3:N2}GB  workerBasePriority={4} (6=BelowNormal)" -f `
  $p.ExitCode, $sw.Elapsed.TotalSeconds, ($peakPriv / 1GB), ($peakWS / 1GB), $prio
