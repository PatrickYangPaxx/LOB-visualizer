@echo off
rem Builds dist\LOBViewer\LOBViewer.exe. Run from this folder, by double-clicking or in a terminal.
cd /d "%~dp0"
tasklist /fi "imagename eq LOBViewer.exe" | find /i "LOBViewer.exe" >nul && (
  echo LOBViewer.exe is still running. Close the viewer and any export windows, then run this again.
  pause
  exit /b 1
)
python -m pip install -r requirements.txt pyinstaller || goto :fail
python -m PyInstaller --noconfirm --clean --name LOBViewer ^
  --collect-all databento --collect-all databento_dbn --collect-all imageio_ffmpeg ^
  --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --exclude-module tkinter ^
  viewer.py || goto :fail
echo.
echo Done: dist\LOBViewer\LOBViewer.exe
pause
exit /b 0
:fail
echo.
echo Build failed, see the messages above.
pause
exit /b 1
