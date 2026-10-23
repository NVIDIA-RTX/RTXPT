@echo off
rem PSNR vs GOLD for denoised reference and DLSS-RR images only; the rest are for A/B eyeballing. Needs ImageMagick.
rem Drop vs RegressionTesting-PSNR-reference.txt: green < 1dB, yellow < 4dB, red >= 4dB.
setlocal EnableDelayedExpansion
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "REFFILE=%~dp0RegressionTesting-PSNR-reference.txt"
cd /d "%~dp0RegressionTesting"
echo(
call :Print "Image" "PSNR" "Ref" "Drop" 0
for %%F in (*-16384spp-denoised.png *-DLSSRR.png) do call :Row "%%~nF"
echo(
pause
exit /b 0

:Row
set "N=%~1"
if not exist "%N%_GOLD.png" ( call :Print "%N%" "no GOLD" "" "" 0 & exit /b )
set "CUR="
for /f "tokens=1" %%p in ('magick compare -metric psnr "%N%.png" "%N%_GOLD.png" null: 2^>^&1') do if not defined CUR set "CUR=%%p"
echo(!CUR!| findstr /r /x "[0-9][0-9]*\.*[0-9]* inf" >nul || ( call :Print "%N%" "ERROR" "" "" 91 & exit /b )
set "REF="
for /f "usebackq tokens=1,2" %%a in ("%REFFILE%") do if /i "%%a"=="%N%" set "REF=%%b"
if not defined REF ( call :Print "%N%" "!CUR!" "-" "-" 0 & exit /b )
if /i "!CUR!"=="inf" ( call :Print "%N%" "!CUR!" "!REF!" "-inf" 92 & exit /b )
call :Centi CUR C
call :Centi REF R
set /a "D=R-C"
set "COL=91"
if !D! lss 400 set "COL=93"
if !D! lss 100 set "COL=92"
call :FmtCenti D DS
call :Print "%N%" "!CUR!" "!REF!" "!DS!" !COL!
exit /b

rem <name> <psnr> <ref> <drop> <ansiColor>
:Print
set "C1=%~1                                "
set "C2=        %~2"
set "C3=        %~3"
set "C4=        %~4"
echo !ESC![%~5m!C1:~0,32!!C2:~-8!  !C3:~-8!  !C4:~-7!!ESC![0m
exit /b

rem decimal string in var %1 -> integer hundredths in var %2
:Centi
set "V=!%1!"
set "I=" & set "F="
for /f "tokens=1,2 delims=." %%a in ("!V!") do ( set "I=%%a" & set "F=%%b" )
set "F=!F!00"
set /a "%2=I*100 + 1!F:~0,2! - 100"
exit /b

rem integer hundredths in var %1 -> signed "+1.23" string in var %2
:FmtCenti
set /a "A=%1"
set "S=+"
if !A! lss 0 ( set "S=-" & set /a "A=-A" )
set /a "W=A/100, P=A%%100"
set "P=0!P!"
set "%2=!S!!W!.!P:~-2!"
exit /b
