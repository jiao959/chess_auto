# chess_auto

Chinese chess board recognition and local analysis assistant for Windows. The app captures a Xiangqi board from the screen, recognizes pieces, generates FEN, calls Pikafish for analysis, and can optionally execute moves automatically.

> Usage scope: this project is intended only for local testing, self-play, offline research, and debugging. Do not use it to gain unfair advantage in online games or competitive play.

## Features

- Automatic screen capture and board region detection
- 90-point board grid extraction
- Piece recognition from board intersections
- Xiangqi FEN generation and legality checks
- Pikafish best-move analysis
- Optional automatic move execution
- Windows packaged build with no Python installation required for end users

## Download And Run

For normal users, download the packaged Windows release from the GitHub Releases page.

1. Download `chess_auto_windows.zip`.
2. Extract the whole folder.
3. Open the extracted `chess_auto` folder.
4. Run `chess_auto.exe`.

Do not run the exe directly inside the zip file. Do not copy only `chess_auto.exe`; the `_internal` folder and bundled engine files are required.

## Runtime Notes

- Windows is required.
- The emulator or game window must already be open before analysis.
- Windows display scaling is recommended to be `100%`.
- Keep the board visible and unobstructed.
- If Windows Defender or another antivirus blocks the app, add the extracted `chess_auto` folder to the trusted list.
- If recognition is inaccurate, keep the game window size and position stable and rerun analysis.

## Build From Source

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Build the Windows package:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

The packaged app will be generated at:

```text
dist\chess_auto\chess_auto.exe
```

To distribute the app, zip the entire folder:

```text
dist\chess_auto
```

## Project Structure

```text
main.py                       GUI entry point
auto_move_worker.py           Automatic move loop
module1_board_detection.py    Screenshot and board detection
module2_point_crops.py        Board grid and point crop generation
module3_piece_recognition.py  Piece recognition
module4_fen_generation.py     Legality checks and FEN generation
module5_engine_analysis.py    Pikafish analysis
engine/                       Engine assets
template/                     Recognition templates
test/棋盘.png                 Board color reference sample used by module 1
build_exe.ps1                 PyInstaller build script
```

Generated folders such as `dist/`, `build/`, `debug_outputs/`, and `__pycache__/` are not source files and are intentionally ignored by Git.

## Release Workflow

1. Build the app with `build_exe.ps1`.
2. Zip the entire `dist\chess_auto` folder as `chess_auto_windows.zip`.
3. Create a GitHub Release, for example `v1.0.0`.
4. Upload `chess_auto_windows.zip` as the release asset.
5. Users download the zip, extract it, and run `chess_auto.exe`.

## Disclaimer

This software is provided for educational and local research use. Users are responsible for complying with the rules of any platform, game, or service they interact with.
