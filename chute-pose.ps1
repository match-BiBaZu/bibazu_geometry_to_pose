[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ChutePoseArguments
)

$geometryRepo = Split-Path -Parent $PSCommandPath
$python = Join-Path $geometryRepo '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Local interpreter not found: $python. Run 'uv sync --extra gui' in $geometryRepo."
}

& $python -m chute_pose.cli @ChutePoseArguments
exit $LASTEXITCODE
