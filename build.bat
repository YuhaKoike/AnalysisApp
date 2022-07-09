set MYDIR=%~dp0
cd %MYDIR%\sorce
py -3.6 -m PyInstaller main.py --onefile --noconsole
pause