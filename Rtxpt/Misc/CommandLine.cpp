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
			("a,adapter", "-adapter must be followed by a string used to match the preferred adapter, e.g -adapter NVIDIA or -adapter RTX", value(adapter))
			("screenshotFileName", "Will save a screenshot with the specified name.", value(screenshotFileName))
			("screenshotFrameIndex", "Will capture a screenshot at this specific frame index. Application will terminate after screenshot is taken.", value(screenshotFrameIndex))
			("h,help", "Print the help message", value(help))
			("d3d12", "Render using DirectX 12 (default)")
			("vk", "Render using Vulkan", value(useVulkan));

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