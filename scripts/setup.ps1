# RAIM setup en Windows (PowerShell). Ejecutar desde la carpeta RAIM.
$ErrorActionPreference = "Stop"
uv sync
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# C++ opcional: requiere VS Build Tools 2022 (C++) + CMake
if (Get-Command cl.exe -ErrorAction SilentlyContinue) { uv pip install ./cpp } else {
  Write-Host "Sin compilador C++: se usará el backend Python puro (correcto, más lento)." }
uv run pytest -q
