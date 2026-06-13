$ErrorActionPreference = "Stop"

$IconIco = Get-ChildItem -File -Filter "*.ico" | Select-Object -First 1
$IconPng = Get-ChildItem -File -Filter "*.png" | Select-Object -First 1

if (-not $IconIco) {
  throw "Missing .ico icon file"
}

if (-not $IconPng) {
  throw "Missing .png icon file"
}

python -m PyInstaller `
  --clean `
  --noconfirm `
  --onedir `
  --windowed `
  --name chess_auto `
  --icon $IconIco.FullName `
  --hidden-import module1_board_detection `
  --hidden-import module2_point_crops `
  --hidden-import module3_piece_recognition `
  --hidden-import module4_fen_generation `
  --hidden-import module5_engine_analysis `
  --add-data "template;template" `
  --add-data "engine;engine" `
  --add-data "test;test" `
  --add-data "$($IconPng.FullName);." `
  --add-data "$($IconIco.FullName);." `
  main.py

Write-Host ""
Write-Host "Build complete: dist\chess_auto\chess_auto.exe"
