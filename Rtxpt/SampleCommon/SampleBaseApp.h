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

#include <memory>

#include "donut/app/DeviceManager.h"
#include "donut/core/log.h"

#include "SampleCommon.h"
#include "CommandLine.h"

#define RTXPT_ENABLE_VIDEO_MEMORY_INFO 1

#if RTXPT_ENABLE_VIDEO_MEMORY_INFO
#include <dxgi1_4.h>    // IDXGIAdapter3 (DXGI 1.4)
#include <wrl/client.h> // Microsoft::WRL::ComPtr
#endif

namespace donut::engine
{
	class ShaderFactory;
}

class Sample;
class SampleUI;

class SampleBaseApp
{
public:
	SampleBaseApp();
	~SampleBaseApp();

	enum class InitReturnCodes
	{
		Success,
		FailProcessingCommandLine,
		FailToCreateDevice,
		FailDeviceFeatureSupport
	};

	InitReturnCodes Init(int argc, const char* const* argv);
	void End();

	void RunMainLoop();

    bool IsSERSupported() const;
    bool QueryVideoMemoryInfo(uint64_t& outBudget, uint64_t& outCurrentUsage, uint64_t& outAvailableForReservation, uint64_t& outCurrentReservation); // RTXPT_ENABLE_VIDEO_MEMORY_INFO for this to work, otherwise return false

private:
	virtual std::unique_ptr<Sample> CreateMainRenderPass(donut::app::DeviceManager& deviceManager, const CommandLineOptions& cmdLineOptions) = 0;

	// Initialization methods
	void RegisterDonutCallback();
	void SampleLogCallback(donut::log::Severity severity, const char* message);
	donut::app::DeviceCreationParameters GetDefaultDeviceParams() const;
	bool ProcessCommandLine(int argc, char const* const* argv,
		donut::app::DeviceCreationParameters& deviceParams, std::string& preferredScene);
	bool InitDeviceAndWindow(const donut::app::DeviceCreationParameters& deviceParams);
	bool CheckDeviceFeatureSupport(const donut::app::DeviceCreationParameters& deviceParams);
	void CreateShaderFactory();

	donut::log::Callback m_DonutDefaultCallback = nullptr;
	FPSLimiter m_FPSLimiter;

	CommandLineOptions m_CmdLine;

	std::unique_ptr<donut::app::DeviceManager> m_DeviceManager;
	std::shared_ptr<donut::engine::ShaderFactory> m_ShaderFactory;
	std::unique_ptr<Sample> m_MainSceneRender; // 3d render of the scene. Where Path Tracing happens
	std::unique_ptr<SampleUI> m_UIRender;

#if RTXPT_ENABLE_VIDEO_MEMORY_INFO // & DX12
    Microsoft::WRL::ComPtr<IDXGIAdapter3>   m_d3dAdapter;
#endif

    void * m_NVAPIValidationHandle = nullptr;
};
