@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-PoseRoadmapGUI.ps1"
if errorlevel 1 (
  echo Pose Roadmap Generator could not start. See error above.
  pause
  exit /b 1
)
