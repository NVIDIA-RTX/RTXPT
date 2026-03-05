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
	uint32_t width = 1920;
	uint32_t height = 1080;
	bool fullscreen = false;
	std::string adapter;
    int adapterIndex = -1;
	bool useVulkan = false;
    bool stopAnimations = false;
	bool disableSER = false;

    std::string capturePath = "";
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

    std::string PropShowTags        = "";
    std::string PropCameraAttach    = "";

	CommandLineOptions(){}

	bool InitFromCommandLine(int argc, char const* const* argv);
};