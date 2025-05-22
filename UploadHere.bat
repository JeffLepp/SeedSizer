@echo off
REM Allow drag and drop or direct double-click

REM If user dragged a folder onto this .bat file:
IF NOT "%~1"=="" (
    python SeedSizer.py "%~1"
) ELSE (
    python SeedSizer.py .
)
pause