# G0-E overnight runner. One watchdog, one python tree, auto-resume on crash/stale.
#
#   cd E:\cv_codex\ecommerce-agentic-rag
#   $env:TAU3_AGENT_BASE_URL = "http://127.0.0.1:8123/v1"
#   $env:TAU3_USER_API_KEY = "<key>"
#   $env:TAU3_NL_ASSERTIONS_MODEL = "deepseek/deepseek-chat"
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_g0e_local_watchdog.ps1
#
# Active tau2 checkout + data live under external\tau2-bench (not the abandoned archive).

$ErrorActionPreference = "Stop"

$ProjectRoot    = "E:\cv_codex\ecommerce-agentic-rag"
$TauRoot        = "E:\cv_codex\external\tau2-bench"
$Python         = Join-Path $TauRoot ".venv\Scripts\python.exe"
$SaveTo         = "tau3_g0e_train_qwen3_4b_temp08_k8"
$Expected       = 592
$StaleMin       = 45
$PollSec        = 60
$MaxConcurrency = 1

if ($env:TAU2_DATA_DIR) {
    $TauDataDir = $env:TAU2_DATA_DIR
} else {
    $TauDataDir = Join-Path $TauRoot "data"
}
$Results   = Join-Path $TauDataDir "simulations\$SaveTo\results.json"
$Artifacts = Join-Path $TauDataDir "simulations\$SaveTo\artifacts"
$LogDir    = Join-Path $ProjectRoot "logs"
$WatchLog  = Join-Path $LogDir "g0e_watchdog.log"
$RunnerLog = Join-Path $LogDir "g0e_runner_latest.log"
$LockFile  = Join-Path $LogDir "g0e_watchdog.lock"

foreach ($name in @("TAU3_AGENT_BASE_URL", "TAU3_NL_ASSERTIONS_MODEL")) {
    if (-not (Get-Item "env:$name" -ErrorAction SilentlyContinue).Value) {
        throw "Missing env:$name"
    }
}
if (-not $env:TAU3_USER_API_KEY -and -not $env:ERAG_LLM_API_KEY -and -not $env:ARAG_LLM_API_KEY) {
    throw "Missing TAU3_USER_API_KEY (or ERAG_/ARAG_ fallback)"
}
if (-not (Test-Path $Python)) {
    throw "tau2 python not found: $Python"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Path $LockFile) {
    $old = Get-Content $LockFile -ErrorAction SilentlyContinue
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
        throw "Watchdog already running (pid $old). Kill it first."
    }
}
Set-Content $LockFile $PID -Encoding ascii

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content $WatchLog $line -Encoding utf8
    Write-Host $line
}

function SimCount {
    if (Test-Path $Results) {
        try {
            $n = & $Python -c "import json;from pathlib import Path;p=Path(r'$Results');print(len(json.loads(p.read_text(encoding='utf-8')).get('simulations') or []))"
            return [int]$n
        } catch {
            Log "SimCount warning: $($_.Exception.Message)"
        }
    }
    if (Test-Path $RunnerLog) {
        $tail = Get-Content $RunnerLog -Tail 200 -EA SilentlyContinue
        for ($i = $tail.Count - 1; $i -ge 0; $i--) {
            if ($tail[$i] -match 'Status: (\d+)/(\d+) complete') {
                return [int]$Matches[1]
            }
        }
    }
    return 0
}

function NewestLogAgeMin {
    if (-not (Test-Path $Artifacts)) { return $null }
    $f = Get-ChildItem $Artifacts -Recurse -Filter task.log -EA SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $f) { return $null }
    return ((Get-Date) - $f.LastWriteTime).TotalMinutes
}

function Kill-Runner {
    try {
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
            Where-Object { $_.CommandLine -match 'run_tau3_retail_v1|_tau3_cli_with_frozen_judge' } |
            ForEach-Object { & taskkill /F /T /PID $_.ProcessId 2>$null | Out-Null }
        Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -EA SilentlyContinue |
            Where-Object { $_.CommandLine -match 'g0e_runner\.cmd' } |
            ForEach-Object { & taskkill /F /T /PID $_.ProcessId 2>$null | Out-Null }
    } catch {
        Log "Kill-Runner warning: $($_.Exception.Message)"
    }
}

function Start-RunnerProcess {
    $nl = $env:TAU3_NL_ASSERTIONS_MODEL
    $cmd = @"
@echo off
set TAU2_DATA_DIR=$TauDataDir
cd /d "$ProjectRoot"
echo === runner start %DATE% %TIME% ===>> "$RunnerLog"
"$Python" -m scripts.run_tau3_retail_v1 --tau-root "$TauRoot" --phase teacher --agent-name llm_agent --agent-model hosted_vllm/Qwen3-4B-Instruct-2507 --user-model deepseek/deepseek-chat --nl-assertions-model "$nl" --agent-temperature 0.8 --user-temperature 0.0 --seed 300 --pass-k 8 --max-steps 200 --save-to $SaveTo --max-concurrency $MaxConcurrency >> "$RunnerLog" 2>&1
echo === runner exit %ERRORLEVEL% %DATE% %TIME% ===>> "$RunnerLog"
exit /b %ERRORLEVEL%
"@
    $bat = Join-Path $LogDir "g0e_runner.cmd"
    Set-Content $bat $cmd -Encoding ascii
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $bat -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden
    Log "started runner cmd pid=$($p.Id) tau_root=$TauRoot data_dir=$TauDataDir"
    return @{ Proc = $p; StartedAt = Get-Date }
}

try {
    Log "watchdog pid=$PID tau_root=$TauRoot data_dir=$TauDataDir expected=$Expected stale=${StaleMin}m poll=${PollSec}s max_concurrency=$MaxConcurrency"
    $run = $null
    $lastCount = -1
    $lastBumpAt = Get-Date

    while ($true) {
        $count = SimCount
        if ($count -ne $lastCount) {
            $lastCount = $count
            $lastBumpAt = Get-Date
        }
        if ($count -ge $Expected) { Log "done $count/$Expected"; break }

        if ($null -eq $run -or $run.Proc.HasExited) {
            if ($null -ne $run -and $run.Proc.HasExited) {
                Log "runner exited code=$($run.Proc.ExitCode) progress=$count/$Expected"
            }
            Kill-Runner
            Start-Sleep 2
            $run = Start-RunnerProcess
            $lastBumpAt = Get-Date
        }

        $runMin = ((Get-Date) - $run.StartedAt).TotalMinutes
        $stuckMin = ((Get-Date) - $lastBumpAt).TotalMinutes
        $logAge = NewestLogAgeMin
        $logTxt = if ($null -ne $logAge) { "{0:N1}m" -f $logAge } else { "n/a" }
        Log "progress=$count/$Expected run_age=$([math]::Round($runMin,1))m stuck=$([math]::Round($stuckMin,1))m log_age=$logTxt cmd_pid=$($run.Proc.Id)"

        $stale = ($runMin -gt $StaleMin) -and ($stuckMin -gt $StaleMin)

        if ($stale) {
            Log "stale >${StaleMin}m (checkpoint count), killing runner"
            Kill-Runner
            Log "stale kill done, starting runner"
            try {
                if ($null -ne $run.Proc -and -not $run.Proc.HasExited) {
                    $run.Proc.WaitForExit(5000) | Out-Null
                }
            } catch { }
            Start-Sleep 2
            $run = Start-RunnerProcess
            $lastBumpAt = Get-Date
        }

        Start-Sleep $PollSec
    }
}
finally {
    Remove-Item $LockFile -Force -EA SilentlyContinue
}
