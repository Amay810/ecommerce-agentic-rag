param(
    [string]$TauRoot = "E:\cv_codex\external\tau2-bench",
    [string]$AgentBaseUrl = "http://127.0.0.1:8123/v1",
    [string]$SaveTo = "tau3_hint_pilot_v2_qwen3_4b_k2"
)

$ErrorActionPreference = "Stop"

$keyPath = Join-Path $env:LOCALAPPDATA "ecommerce-agentic-rag\tau3_deepseek.key"
$secureKey = Get-Content -LiteralPath $keyPath | ConvertTo-SecureString
$credential = [System.Net.NetworkCredential]::new("", $secureKey)

$env:DEEPSEEK_API_KEY = $credential.Password
$env:HOSTED_VLLM_API_BASE = $AgentBaseUrl
$env:HOSTED_VLLM_API_KEY = "local-vllm"
$env:TAU3_NL_ASSERTIONS_MODEL = "deepseek/deepseek-chat"
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = $env:NO_PROXY
$env:PYTHONUTF8 = "1"

$tauPython = Join-Path $TauRoot ".venv\Scripts\python.exe"
$launcher = Join-Path $PSScriptRoot "_tau3_semantic_hint_cli.py"

& $tauPython $launcher run `
    --domain retail `
    --task-set-name retail `
    --task-split-name train `
    --task-ids 14 20 29 30 46 59 85 109 `
    --num-trials 2 `
    --agent llm_agent_semantic_hint_v2 `
    --agent-llm hosted_vllm/Qwen3-4B-Instruct-2507 `
    --user user_simulator `
    --user-llm deepseek/deepseek-chat `
    --max-steps 200 `
    --max-errors 10 `
    --seed 300 `
    --max-concurrency 4 `
    --save-to $SaveTo `
    --auto-resume `
    --verbose-logs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
