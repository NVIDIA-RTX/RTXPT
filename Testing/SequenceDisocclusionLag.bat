@set SUFFIX=
@set /p SUFFIX=Optional file name suffix (e.g. A gives frame_000_A.png, Enter for none): 
@set SUFFIXARG=
@if not "%SUFFIX%"=="" set SUFFIXARG=--captureSuffix "%SUFFIX%"
"%~dp0..\bin\Rtxpt.exe" --scene bistro-programmer-art.scene.json --width 1920 --height 1080 --propShowTags LagTestWallLamp --propCameraAttach SHIP_LagTestWallLamp --captureSequence --sequenceFrameCount 8 --sequenceFPS 15 --sequenceWarmupStart 4.0 --sequenceRecordStart 5.82 --useNEE 1 --NEEType 2 --useReSTIRDI 0 --useReSTIRGI 0 --realtimeSamplesPerPixel 1 --standaloneDenoiser 0 --realtimeAA 0 --overrideToRealtimeMode --overrideAutoexposureOff --overrideExposureOffset 4.0 --disableFireflyFilters --disablePostProcessFilters --capturePath "%~dp0SequenceDisocclusionLag\frame.png" %SUFFIXARG%
