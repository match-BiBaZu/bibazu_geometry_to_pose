[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ChutePoseArguments
)

$geometryRepo = Split-Path -Parent $PSCommandPath
$workspace = Split-Path -Parent $geometryRepo
$python = Join-Path $workspace "BiBaZu_Big_Boi\ReorientationControlGUI\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Shared Reorientation Control interpreter not found: $python"
}

& $python -m chute_pose.cli @ChutePoseArguments
exit $LASTEXITCODE
