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
	std::string screenshotFileName;
	uint32_t screenshotFrameIndex = 0xFFFFFFFF;
	bool useVulkan = false;

	CommandLineOptions(){}

	bool InitFromCommandLine(int argc, char const* const* argv);
};