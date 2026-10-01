$ErrorActionPreference = 'Stop'
$poseRepo = Split-Path -Parent $PSCommandPath
$posePython = Join-Path $poseRepo '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $posePython)) { throw "Missing local Python environment. Run 'uv sync --extra gui' in $poseRepo" }
Start-Process -FilePath $posePython -ArgumentList ('"' + (Join-Path $poseRepo 'PoseRoadmapGUI.py') + '"') -WorkingDirectory $poseRepo -WindowStyle Hidden
