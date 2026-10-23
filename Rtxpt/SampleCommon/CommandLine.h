/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <string>
#include <optional>

struct CommandLineOptions
{
	std::string scene;
	bool nonInteractive = false;
	bool noWindow = false;
	bool debug = false;
	bool gpuValidation = false;
	uint32_t width = 1920;
	uint32_t height = 1080;
	bool fullscreen = false;
	bool borderless = false;
	std::string adapter;
    int adapterIndex = -1;
	bool useVulkan = false;
    bool stopAnimations = false;
	bool disableSER = false;

    std::string capturePath = "";
    std::string captureSuffix = "";
    bool captureSimple = false;
    bool captureSequence = false;
    float sequenceWarmupStart = 0;
    float sequenceRecordStart = 0;
    float sequenceFPS = 60.0f;
    int sequenceFrameCount = 0;

    std::string cameraPosDirUp = "";

    int  UseNEE                     = 1;
    int  NEEType                    = 2;
    int  UseReSTIRDI                = false;
    int  UseReSTIRGI                = true;
    int  RealtimeSamplesPerPixel    = 1;
    int  ReferenceSamplesPerPixel   = 4096;
    int  StandaloneDenoiser         = true;
    int  RealtimeAA                 = 3;

    bool OverrideToRealtimeMode     = false;
    bool OverrideToReferenceMode    = false;

    bool OverrideAutoexposureOff    = false;
    float OverrideExposureOffset    = FLT_MAX;
    
    bool DisableFireflyFilters      = false;
    bool DisablePostProcessFilters  = false;
    bool AsyncPrefix                = true;

    int  OverrideRadianceCacheMode  = -1;   // -1 keeps the default

    int  OverrideSubSamplingMode    = -1;   // -1 keeps the performance preset's; 0 - disabled, 1 - three quarters, 2 - half, 3 - quarter
    int  SubSamplingPattern         = 3;    // SubSamplingPattern in SampleUI.h
    int  SubSamplingBlockSize       = 8;    // N for SubSamplingPattern::BlockStratified and ::SobolBlock

    int  OverrideDLSSMode           = -1;   // -1 keeps the preset's / UI's; otherwise forces StreamlineInterface::DLSSMode every frame
    std::string PerfLogPath         = "";   // if set, per-frame GPU timings of the path tracing passes are appended there as CSV

    std::string PropShowTags        = "";
    std::string PropCameraAttach    = "";

	CommandLineOptions(){}

	bool InitFromCommandLine(int argc, char const* const* argv);
};