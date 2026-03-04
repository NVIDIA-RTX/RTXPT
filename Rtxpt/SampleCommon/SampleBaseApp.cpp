/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SampleBaseApp.h"

#include <string>
#include "../Misc/Korgi.h"
#include "../SampleUI.h"
#include "LocalConfig.h"

#include <donut/app/ApplicationBase.h>
#include "../Sample.h"

extern SampleUIData g_sampleUIData;
extern const char* g_windowTitle;

#if RTXPT_D3D12_WITH_NVAPI
#include <nvApi.h>

#if 0
// Validation callback
static void __stdcall myValidationMessageCallback(void* pUserData, NVAPI_D3D12_RAYTRACING_VALIDATION_MESSAGE_SEVERITY severity, const char* messageCode, const char* message, const char* messageDetails)
{
    const char* severityString = "unknown";
    switch (severity)
    {
    case NVAPI_D3D12_RAYTRACING_VALIDATION_MESSAGE_SEVERITY_ERROR: severityString = "error"; break;
    case NVAPI_D3D12_RAYTRACING_VALIDATION_MESSAGE_SEVERITY_WARNING: severityString = "warning"; break;
    }

    donut::log::warning("NVAPI Ray Tracing Validation message: %s: [%s] %s\n%s", severityString, messageCode, message, messageDetails);
}
#endif
#endif


SampleBaseApp::SampleBaseApp()
{
    RegisterDonutCallback(); // Register a custom donut callback to filter errors

    korgi::Init(); // MIDI Input for parameter control

    // Init graphics API
    nvrhi::GraphicsAPI api = donut::app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    m_DeviceManager = std::unique_ptr<donut::app::DeviceManager>(donut::app::DeviceManager::Create(api));

    m_DeviceManager->SetFrameTimeUpdateInterval(1.0f);
}

SampleBaseApp::~SampleBaseApp()
{
    m_DeviceManager->Shutdown(); // Is this explicitly necessary?
    korgi::Shutdown();
}

SampleBaseApp::InitReturnCodes SampleBaseApp::Init(int argc, const char* const* argv)
{
    donut::app::DeviceCreationParameters deviceParams = GetDefaultDeviceParams();

    std::string preferredScene = "bistro-programmer-art.scene.json"; // "kitchen.scene.json"; 
    LocalConfig::PreferredSceneOverride(preferredScene);

    // Process command line arguments
    if (!ProcessCommandLine(argc, argv, deviceParams, preferredScene))
    {
        return InitReturnCodes::FailProcessingCommandLine;
    }

    if (!InitDeviceAndWindow(deviceParams))
    {
        return InitReturnCodes::FailToCreateDevice;
    }

    // Check API feature support
    if (!CheckDeviceFeatureSupport(deviceParams))
    {
        return InitReturnCodes::FailDeviceFeatureSupport;
    }

    CreateShaderFactory();

    // -- Register render passes into Donut -- 
    // Create the main render pass. This is where path tracing happens
    m_MainSceneRender = CreateMainRenderPass(*m_DeviceManager, m_CmdLine);
    m_MainSceneRender->Init(preferredScene, m_ShaderFactory);
    m_DeviceManager->AddRenderPassToBack(m_MainSceneRender.get());

#if DONUT_WITH_DX12 && (RTXPT_D3D_AGILITY_SDK_VERSION >= 619)   // temporary
    // When using AgilitySDK >= 619, we require shader model 6.9
    if (m_DeviceManager->GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        ID3D12Device* d3d12Device = static_cast<ID3D12Device*>(
            m_DeviceManager->GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device)
            );

        D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_9 };

        HRESULT hr = d3d12Device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel));
        assert(SUCCEEDED(hr));
        if (shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_9)
        {
            donut::log::fatal("Shader Model 6.9 is required when compiled with Agility SDK 1.619 or newer, but is unsupported on the current device. Please check for newer graphics drivers, or recompile without Agility SDK");
            return InitReturnCodes::FailToCreateDevice;
        }
    }
#endif

    // Optionally create the UP render pass. This exposes run time parameter controls.
    if (!m_CmdLine.noWindow)
    {
        m_UIRender = std::make_unique<SampleUI>(m_DeviceManager.get(), *this, *m_MainSceneRender, g_sampleUIData, IsSERSupported(), m_CmdLine);
        m_UIRender->Init(m_ShaderFactory);
        m_DeviceManager->AddRenderPassToBack(m_UIRender.get());
    }

    LocalConfig::PostAppInit(g_sampleUIData);

#if RTXPT_ENABLE_VIDEO_MEMORY_INFO // & DX12
    auto device = m_DeviceManager->GetDevice();
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        ID3D12Device* d3dDevice = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);

        LUID luid = d3dDevice->GetAdapterLuid();
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        CreateDXGIFactory1(IID_PPV_ARGS(&factory));
        factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&m_d3dAdapter));
        //m_d3dAdapter->QueryVideoMemoryInfo()
    }
#endif

    return InitReturnCodes::Success;
}

bool SampleBaseApp::QueryVideoMemoryInfo(uint64_t& outBudget, uint64_t& outCurrentUsage, uint64_t& outAvailableForReservation, uint64_t& outCurrentReservation)
{
#if RTXPT_ENABLE_VIDEO_MEMORY_INFO // & DX12
    DXGI_QUERY_VIDEO_MEMORY_INFO info;
    if (FAILED(m_d3dAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        return false;
    outBudget = info.Budget;
    outCurrentUsage = info.CurrentUsage;
    outAvailableForReservation = info.AvailableForReservation;
    outCurrentReservation = info.CurrentReservation;
    return true;
#else
    return false;
#endif
}

void SampleBaseApp::End()
{
    // Destroy resources in reverse creation order
    if (m_UIRender)
    {
        m_DeviceManager->RemoveRenderPass(m_UIRender.get());
        m_UIRender.reset();
    }

#if RTXPT_D3D12_WITH_NVAPI
    if (m_NVAPIValidationHandle != nullptr)
    {
        auto nativeObj = m_DeviceManager->GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        NvAPI_D3D12_FlushRaytracingValidationMessages(nativeObj);
    }
#endif

    m_DeviceManager->RemoveRenderPass(m_MainSceneRender.get());
    m_MainSceneRender.reset();

    m_ShaderFactory.reset();
}

void SampleBaseApp::RunMainLoop()
{
    m_DeviceManager->RunMessageLoop();
}

void SampleBaseApp::RegisterDonutCallback()
{
    // Get the default call back first so we can pass messages through to it.
    m_DonutDefaultCallback = donut::log::GetCallback();

    // Register our custom callback to intercept and filter streamline errors
    donut::log::SetCallback([this](donut::log::Severity severity, const char* message)
        {
            this->SampleLogCallback(severity, message);
        });
}

void SampleBaseApp::SampleLogCallback(donut::log::Severity severity, const char* message)
{
    // This lets us demote some of Streamline errors that aren't errors into warnings
    if (severity == donut::log::Severity::Error)
    {
        std::string msg(message);
        if (msg.find("Don't know the size") != std::string::npos)
            severity = donut::log::Severity::Warning;
        if (msg.find("dlss_gEntry.cpp") != std::string::npos)
        {
            if (msg.find("Unable to find DRS context") != std::string::npos
                || msg.find("NGX indicates DLSS-G is not available") != std::string::npos)
                severity = donut::log::Severity::Warning;
        }
        if (msg.find("Missing NGX context") != std::string::npos
            || msg.find("Unable to find NGX ") != std::string::npos
            || msg.find("NvAPI_D3D_Sleep") != std::string::npos)
            severity = donut::log::Severity::Warning;
    }

    // Pass all other messages to donut's default callback
    m_DonutDefaultCallback(severity, message);
}

donut::app::DeviceCreationParameters SampleBaseApp::GetDefaultDeviceParams() const
{
    donut::app::DeviceCreationParameters deviceParams;
    deviceParams.backBufferWidth = 0;   // initialized from CmdLine
    deviceParams.backBufferHeight = 0;  // initialized from CmdLine
    deviceParams.swapChainSampleCount = 1;
    deviceParams.swapChainBufferCount = c_SwapchainCount;
    deviceParams.startFullscreen = false;
    deviceParams.startBorderless = false;
    deviceParams.vsyncEnabled = true;
    deviceParams.enableRayTracingExtensions = true;
#if DONUT_WITH_DX12
#if defined(RTXPT_D3D_AGILITY_SDK_VERSION)
    deviceParams.featureLevel = D3D_FEATURE_LEVEL_12_2;
    // TODO: Redefining this isn't needed. Take the ones from AgilitySDK
    static const UUID D3D12ExperimentalShaderModels = { 0x76f5573e, 0xf13a, 0x40f5, {0xb2, 0x97, 0x81, 0xce, 0x9e, 0x18, 0x93, 0x3f} };
    static const UUID D3D12StateObjectsExperiment = { 0x398a7fd6, 0xa15a, 0x42c1, {0x96, 0x05, 0x4b, 0xd9, 0x99, 0x9a, 0x61, 0xaf} };
    static const UUID D3D12RaytracingExperiment = { 0xb56e238b, 0xe886, 0x46d8, {0x9b, 0xe1, 0x34, 0x10, 0x30, 0x31, 0x45, 0x09} };
    UUID Features[] = { D3D12ExperimentalShaderModels }; //, D3D12StateObjectsExperiment }; //, D3D12RaytracingExperiment };
    HRESULT ok = D3D12EnableExperimentalFeatures(_countof(Features), Features, nullptr, nullptr);
#else
    deviceParams.featureLevel = D3D_FEATURE_LEVEL_12_1;
#endif
#endif
#if defined(_DEBUG)
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableWarningsAsErrors = true;
    deviceParams.enableNvrhiValidationLayer = true;
    deviceParams.enableGPUValidation = false;       // <- this severely impact performance but is good to enable from time to time
#endif
    deviceParams.supportExplicitDisplayScaling = true;

#if DONUT_WITH_STREAMLINE
    deviceParams.checkStreamlineSignature = true;   // <- Set to false if you're using a local build of streamline
    deviceParams.streamlineAppId = 231313132;
#if defined(_DEBUG)
    deviceParams.enableStreamlineLog = true;
#endif
#endif

#if DONUT_WITH_VULKAN
    deviceParams.requiredVulkanDeviceExtensions.push_back("VK_KHR_buffer_device_address");
    deviceParams.requiredVulkanDeviceExtensions.push_back("VK_KHR_format_feature_flags2");

    // Attachment 0 not written by fragment shader; undefined values will be written to attachment (OMM baker)
    deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x0000000023e43bb7);

    // vertex shader writes to output location 0.0 which is not consumed by fragment shader (OMM baker)
    deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x000000000609a13b);

    // vkCmdDispatch(): Buffer format mismatch for t_NeighborOffsets (RG8_SNORM vs shader expecting RG32_FLOAT)
    // DX12 handles this automatically; it works correctly, just Vulkan is stricter about format matching
    deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x00000000c5a3822a);

    // vkCmdPipelineBarrier2(): pDependencyInfo.pBufferMemoryBarriers[0].dstAccessMask bit VK_ACCESS_SHADER_READ_BIT
    // is not supported by stage mask (Unhandled VkPipelineStageFlagBits)
    // Vulkan validation layer not supporting OMM?
    deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x00000000591f70f2);

    // vkCmdPipelineBarrier2(): pDependencyInfo->pBufferMemoryBarriers[0].dstAccessMask(VK_ACCESS_SHADER_READ_BIT) is not supported by stage mask(VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT)
    // Vulkan Validaiotn layer not supporting OMM bug
    deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x000000005e6e827d);
#endif

    deviceParams.enablePerMonitorDPI = true;

    return deviceParams;
}

bool SampleBaseApp::ProcessCommandLine(int argc, char const* const* argv,
    donut::app::DeviceCreationParameters& deviceParams, std::string& preferredScene)
{
#if 1 // use a bit larger window by default if screen large enough
    glfwInit();
    const auto primMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = (primMonitor != nullptr) ? glfwGetVideoMode(primMonitor) : (nullptr);
    if (mode->width > 2560 && mode->height > 1440)
    {
        m_CmdLine.width = 2560;
        m_CmdLine.height = 1440;
    }
#endif
    if (!m_CmdLine.InitFromCommandLine(__argc, __argv))
    {
        return false;
    }

    if (!m_CmdLine.scene.empty())
    {
        preferredScene = m_CmdLine.scene;
    }

    if (m_CmdLine.nonInteractive)
    {   
        donut::log::EnableOutputToMessageBox(false);
        HelpersSetNonInteractive();
    }

    if (m_CmdLine.debug)
    {
        deviceParams.enableDebugRuntime = true;
        deviceParams.enableNvrhiValidationLayer = true;
    }

    deviceParams.backBufferWidth = m_CmdLine.width;
    deviceParams.backBufferHeight = m_CmdLine.height;
    deviceParams.startFullscreen = m_CmdLine.fullscreen;
    deviceParams.adapterIndex = m_CmdLine.adapterIndex;

    return true;
}

bool SampleBaseApp::InitDeviceAndWindow(const donut::app::DeviceCreationParameters& deviceParams)
{
    if (m_CmdLine.noWindow)
    {
        if (!m_DeviceManager->CreateInstance(deviceParams))
        {
            donut::log::fatal("CreateDeviceAndSwapChain failed: Cannot initialize a graphics device with the requested parameters");
            return false;
        }
    }
    else
    {
        if (!m_DeviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_windowTitle))
        {
            donut::log::fatal("Cannot initialize a graphics device with the requested parameters");
            return false;
        }
        HelpersRegisterActiveWindow();
    }

#if 0 && RTXPT_D3D12_WITH_NVAPI
    static bool NVAPI_VALIDATION = false;
    auto device = m_DeviceManager->GetDevice();
    if (NVAPI_VALIDATION && device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        NvAPI_Status res;
        auto nativeObj = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        //res = NvAPI_D3D12_EnableRaytracingValidation(nativeObj, NVAPI_D3D12_RAYTRACING_VALIDATION_FLAG_NONE);
        res = NvAPI_D3D12_RegisterRaytracingValidationMessageCallback(nativeObj, &myValidationMessageCallback, (void*)this, &this->m_NVAPIValidationHandle);
        assert( res == NvAPI_Status:: NVAPI_OK );
    }
#endif

    return true;
}

bool SampleBaseApp::CheckDeviceFeatureSupport(const donut::app::DeviceCreationParameters& deviceParams)
{
    auto device = m_DeviceManager->GetDevice();
    if (!device->queryFeatureSupport(nvrhi::Feature::RayTracingPipeline))
    {
        donut::log::fatal("The graphics device does not support Ray Tracing Pipelines");
        return false;
    }

    if (!device->queryFeatureSupport(nvrhi::Feature::RayQuery))
    {
        donut::log::fatal("The graphics device does not support Ray Queries");
        return false;
    }

    return true;
}

void SampleBaseApp::CreateShaderFactory()
{
    const char* shaderTypeName = donut::app::GetShaderTypeName(m_DeviceManager->GetGraphicsAPI());
    const std::filesystem::path appDirectory = donut::app::GetDirectoryWithExecutable();
    std::filesystem::path frameworkShaderPath = appDirectory / "ShaderPrecompiled/framework" / shaderTypeName;
    std::filesystem::path appShaderPath = appDirectory / "ShaderPrecompiled/Rtxpt" / shaderTypeName;
    std::filesystem::path nrdShaderPath = appDirectory / "ShaderPrecompiled/nrd" / shaderTypeName;
    std::filesystem::path ommShaderPath = appDirectory / "ShaderPrecompiled/omm" / shaderTypeName;

    std::shared_ptr<donut::vfs::RootFileSystem> rootFS = std::make_shared<donut::vfs::RootFileSystem>();
    rootFS->mount("/ShaderPrecompiled/donut", frameworkShaderPath);
    rootFS->mount("/ShaderPrecompiled/app", appShaderPath);
    rootFS->mount("/ShaderPrecompiled/nrd", nrdShaderPath);
    rootFS->mount("/ShaderPrecompiled/omm", ommShaderPath);

    auto device = m_DeviceManager->GetDevice();
    m_ShaderFactory = std::make_shared<donut::engine::ShaderFactory>(device, rootFS, "/ShaderPrecompiled");
}

bool SampleBaseApp::IsSERSupported() const
{
    auto device = m_DeviceManager->GetDevice();

    const bool usingDX12 = device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12;
    const bool deviceSupportsSER = device->queryFeatureSupport(nvrhi::Feature::ShaderExecutionReordering);
    const bool SERSupported = usingDX12 && deviceSupportsSER && !m_CmdLine.disableSER; // SER Only enabled in DX12 for now

    return SERSupported;
}
