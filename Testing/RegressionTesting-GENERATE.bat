@echo off
rem Usage: RegressionTesting-GENERATE.bat <testName> <scene> <exposureOffset> ["cameraPosDirUp"]
rem Camera must be quoted - cmd treats commas as argument separators.
setlocal
if "%~3"=="" ( echo Usage: %~nx0 ^<testName^> ^<scene^> ^<exposureOffset^> ["cameraPosDirUp"] & exit /b 1 )

set EXE="%~dp0..\bin\Rtxpt.exe"
set DENOISER="%~dp0..\Support\denoiser_OptiX\Denoiser.exe"
set OUT=%~dp0RegressionTesting\%~1
set ARGS=--scene %~2.scene.json --width 1920 --height 1080 --stopAnimations --captureSimple --overrideRadianceCacheMode 0 --useNEE 1 --NEEType 2 --useReSTIRDI 0 --useReSTIRGI 0 --standaloneDenoiser 0 --overrideAutoexposureOff --overrideExposureOffset %~3 --disablePostProcessFilters
if not "%~4"=="" set ARGS=%ARGS% --cameraPosDirUp %~4
if not exist "%~dp0RegressionTesting" mkdir "%~dp0RegressionTesting"

echo === %~1 (%~2, exposure %~3)
%EXE% %ARGS% --capturePath "%OUT%-1spp.png"         --overrideToRealtimeMode  --realtimeSamplesPerPixel 1  --realtimeAA 0 --disableFireflyFilters
%EXE% %ARGS% --capturePath "%OUT%-1spp-DLSSRR.png"  --overrideToRealtimeMode  --realtimeSamplesPerPixel 1  --realtimeAA 3 --disableFireflyFilters
%EXE% %ARGS% --capturePath "%OUT%-32spp.png"        --overrideToRealtimeMode  --realtimeSamplesPerPixel 32 --realtimeAA 0
%EXE% %ARGS% --capturePath "%OUT%-32spp-DLSSRR.png" --overrideToRealtimeMode  --realtimeSamplesPerPixel 32 --realtimeAA 3
%EXE% %ARGS% --capturePath "%OUT%-16384spp.png"     --overrideToReferenceMode --referenceSamplesPerPixel 16384 --realtimeAA 0
rem Only the denoised reference is compared (lower noise floor); -hdr 0 as captures are tonemapped sRGB
%DENOISER% -v 1 -hdr 0 -i "%OUT%-16384spp.png" -o "%OUT%-16384spp-denoised.png" || echo !!! Denoiser failed on %OUT%-16384spp.png
