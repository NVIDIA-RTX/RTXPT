@set SUFFIX=
@set /p SUFFIX=Optional file name suffix (e.g. A gives frame_000_A.png, Enter for none): 
@set SUFFIXARG=
@if not "%SUFFIX%"=="" set SUFFIXARG=--captureSuffix "%SUFFIX%"
"%~dp0..\bin\Rtxpt.exe" --scene bistro-programmer-art.scene.json --width 1920 --height 1080 --cameraPosDirUp -34.041859,3.547544,-5.002510,-0.087543,-0.315515,-0.944874,-0.029109,0.948921,-0.314169 --captureSequence --sequenceFrameCount 6 --sequenceFPS 60 --sequenceWarmupStart 0.0 --sequenceRecordStart 3.40 --useNEE 1 --NEEType 2 --useReSTIRDI 0 --useReSTIRGI 0 --realtimeSamplesPerPixel 1 --standaloneDenoiser 0 --realtimeAA 0 --overrideToRealtimeMode --overrideAutoexposureOff --overrideExposureOffset 4.0 --disableFireflyFilters --disablePostProcessFilters --capturePath "%~dp0SequenceWeightBoost\frame.png" %SUFFIXARG%
