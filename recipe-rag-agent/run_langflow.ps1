$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

& $python -m langflow run `
    --host 127.0.0.1 `
    --port 7860 `
    --open-browser `
    --components-path (Join-Path $projectRoot "langflow_components") `
    --env-file (Join-Path $projectRoot ".env")