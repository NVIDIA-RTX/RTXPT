@echo off
rem Usage: RegressionTesting-COPY-TO-GOLD.bat [testName]  - no argument promotes all tests
setlocal EnableDelayedExpansion
cd /d "%~dp0RegressionTesting"
set FILTER=*
if not "%~1"=="" set FILTER=%~1-*
set /p CONFIRM=Overwrite GOLD images for '%FILTER%'? Type YES to continue:
if /I not "!CONFIRM!"=="yes" ( echo Aborted. & pause & exit /b 1 )
for %%F in (%FILTER%.png) do (
    set N=%%~nF
    if /I not "!N:~-5!"=="_GOLD" ( copy /y "%%F" "%%~nF_GOLD.png" >nul & echo %%~nF -^> GOLD )
)
pause
