/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "CommandLine.h"
#include <cxxopts.hpp>
#include <donut/core/log.h>
#include <filesystem>

bool CommandLineOptions::InitFromCommandLine(int _argc, char const* const* _argv)
{
	using namespace cxxopts;
    
	try
	{
		std::filesystem::path exe_path = _argv[0];
		Options options(exe_path.filename().string(), "RTX Path Tracing is a code sample that strives to embody years of ray tracing and neural graphics research and experience. It is intended as a starting point for a path tracer integration, as a reference for various integrated SDKs, and/or for learning and experimentation.");

		bool help = false;

		options.add_options()
			("s,scene", "Preferred scene to load (.scene.json)", value(scene))
			("nonInteractive", "Indicates that RTXPT will start in non-interactive mode, disabling popups and windows that require input", value(nonInteractive))
			("noWindow", "Start PT-SDK without a window. This mode is useful when generating screenshots from command line.", value(noWindow))
			("d,debug", "Enables the D3D12/VK debug layer and NVRHI validation layer", value(debug))
			("width", "Window width", value(width))
			("height", "Window height", value(height))
			("f,fullscreen", "run in fullscreen mode", value(fullscreen))
			("a,adapter", "--adapter must be followed by a string used to match the preferred adapter, e.g --adapter NVIDIA or --adapter RTX", value(adapter))
			("h,help", "Print the help message", value(help))
			("d3d12", "Render using DirectX 12 (default)")
			("vk", "Render using Vulkan", value(useVulkan))
            ("stopAnimations", "Always start the scene with animations disabled", value(stopAnimations))
            ("noSER", "Disable Shader Execution Reordering", value(disableSER))
            ("adapterIndex", "--adapterIndex must be followed by a number used to identify the preferred adapter index, e.g '--adapterIndex 0' or '--adapterIndex 1'; default is -1 (automatic)", value(adapterIndex))
            ("captureSimple", "Trigger simple screenshot capture with default warmup; use capturePath to specify output file.", value(captureSimple))
            ("captureSequence", "Trigger sequence capture; use capturePath to specify output file, sequenceWarmupTime, sequenceStartTime, sequenceFPS and sequenceFrames to control.", value(captureSequence))
            ("capturePath", "Specify output file or path for various capture commands.", value(capturePath))
            ("sequenceWarmupStart", "Scene time to start warmup when using captureSequence.", value(sequenceWarmupStart))
            ("sequenceRecordStart", "Scene time to start recording when using captureSequence.", value(sequenceRecordStart))
            ("sequenceFPS", "Fixed FPS at which to run when using captureSequence.", value(sequenceFPS))
            ("sequenceFrameCount", "Number of frames to record when using captureSequence", value(sequenceFrameCount))

            ("useNEE", "Enable/disable NEE (on by default)", value(UseNEE))
            ("NEEType", "Set NEE type: 0 - uniform sampling; 1 - power-based sampling, 2 (default) - NEE-AT", value(NEEType))
            ("useReSTIRDI", "Enable/disable ReSTIR DI", value(UseReSTIRDI))
            ("useReSTIRGI", "Enable/disable ReSTIR GI", value(UseReSTIRGI))
            ("realtimeSamplesPerPixel", "Number of spp in realtime mode",     value(RealtimeSamplesPerPixel))
            ("referenceSamplesPerPixel", "Number of spp in reference mode", value(ReferenceSamplesPerPixel))
            ("standaloneDenoiser", "Enable/disable standalone denoiser active when DLSS-RR not enabled (on by default)", value(StandaloneDenoiser))
            ("realtimeAA", "Set realtime AA mode: 0 - none; 1 - TAA; 2 - DLSS; 3 - DLSS-RR (AA+denoiser, default)", value(RealtimeAA))
            ("overrideToRealtimeMode", "Override scene default", value(OverrideToRealtimeMode))
            ("overrideToReferenceMode", "Override scene default", value(OverrideToReferenceMode))
            ("overrideAutoexposureOff", "Override scene default", value(OverrideAutoexposureOff))
            ("overrideExposureOffset", "Override scene default", value(OverrideExposureOffset))
            ("disableFireflyFilters", "Override realtime and reference mode firefly filters to off", value(DisableFireflyFilters))
            ("disablePostProcessFilters", "Disable post-process filters like bloom", value(DisablePostProcessFilters))
            ("cameraPosDirUp", "Specify camera location to set after loading the scene; format is 9 comma separated values; can be obtained via UI Camera->Copy TO clipboard", value(cameraPosDirUp))
            ("propShowTags", "To show special props only if tag found in list (comma separated values)", value(PropShowTags))
            ("propCameraAttach", "After load, try to attach camera to the named prop", value(PropCameraAttach))            
            ;

		int argc = _argc;
		char const* const* argv = _argv;
		options.parse(argc, argv);

		if (help)
		{
			std::string helpMessage = options.help();
			donut::log::info("%s", helpMessage.c_str());
			return false;
		}

		return true;
	}
	catch (const exceptions::exception& e)
	{
		std::string errorMessage = e.what();
		donut::log::error("%s", errorMessage.c_str());
		return false;
	}
}