/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Sample.h"
#include <SampleCommon/SampleBaseApp.h>
#include <SampleCommon/PTPipelineBaker.h>

#include "SampleCommon/SplashScreen.h"

#include "LiveLink/LiveLinkServer.h"
#include <donut/core/log.h>

// IntroRenderer: Simplified renderer for introductory samples
// Currently just uses the base Sample class as-is
// TODO: Override methods to simplify/disable advanced features
class AdvancedPathTracer : public Sample
{
public:
    AdvancedPathTracer(donut::app::DeviceManager& deviceManager, const CommandLineOptions& cmdLine)
        : Sample(deviceManager, cmdLine)
    {
        // Blender Live Link (see Rtxpt/LiveLink/LiveLinkServer.h and Docs/LiveLink.md)
        if (cmdLine.liveLink)
        {
            m_liveLink = std::make_unique<rtxpt::livelink::LiveLinkServer>();
            if (!m_liveLink->Start((uint16_t)cmdLine.liveLinkPort))
                donut::log::warning("Failed to start Blender Live Link server on port %d", (int)cmdLine.liveLinkPort);
        }
    }

    virtual void Animate(float fElapsedTimeSeconds) override
    {
        Sample::Animate(fElapsedTimeSeconds);

        // Apply any camera/reload commands received from the Blender Live Link add-on
        // since the last frame.
        if (m_liveLink && m_liveLink->IsRunning())
        {
            for (const auto& cmd : m_liveLink->PopCommands())
            {
                switch (cmd.type)
                {
                case rtxpt::livelink::CommandType::Camera:
                    LiveLinkApplyCamera(cmd.cameraPosition, cmd.cameraDirection, cmd.cameraUp, cmd.cameraVerticalFovRadians, cmd.cameraZNear);
                    break;
                case rtxpt::livelink::CommandType::Reload:
                    SetCurrentScene(cmd.reloadScenePath, true);
                    break;
                case rtxpt::livelink::CommandType::Hello:
                    donut::log::info("LiveLink: client connected (%s)", cmd.helloInfo.c_str());
                    break;
                }
            }
        }
    }

    virtual void SampleRenderCode(nvrhi::IFramebuffer* framebuffer, nvrhi::CommandListHandle commandList, const SampleConstants& constants) override
    {
        if (m_ui.ActualUseRTXDIPasses())
            m_rtxdiPass->BeginFrame(commandList, *m_renderTargets, m_bindingLayout, m_bindingSet);

        PathTrace(framebuffer, constants);

        Denoise(framebuffer);
    }

    virtual void CreateRTPipelines() override
    {
        auto pipelineBaker = GetRTPipelineBaker();
        using SM = donut::engine::ShaderMacro;

        // these don't actually compile any shaders - this happens later in m_ptPipelineBaker->Update
        m_ptPipelineReference = pipelineBaker->CreateVariant("PathTracerSample.hlsl", { SM("PATH_TRACER_MODE", "PATH_TRACER_MODE_REFERENCE") }, "REF");
        m_ptPipelineBuildStablePlanes = pipelineBaker->CreateVariant("PathTracerSample.hlsl", { SM("PATH_TRACER_MODE", "PATH_TRACER_MODE_BUILD_STABLE_PLANES") }, "BUILD");
        m_ptPipelineFillStablePlanes = pipelineBaker->CreateVariant("PathTracerSample.hlsl", { SM("PATH_TRACER_MODE", "PATH_TRACER_MODE_FILL_STABLE_PLANES") }, "FILL");
        m_ptPipelineTestRaygenPPHDR = pipelineBaker->CreateVariant("TestRaygenPP.hlsl", { SM("PP_TEST_HDR", "1") }, "TESTRG", true);
        m_ptPipelineEdgeDetection = pipelineBaker->CreateVariant("TestRaygenPP.hlsl", { SM("PP_EDGE_DETECTION", "1") }, "EDGY", true);
    }

    virtual void DestroyRTPipelines() override
    {
        m_ptPipelineReference = nullptr;
        m_ptPipelineBuildStablePlanes = nullptr;
        m_ptPipelineFillStablePlanes = nullptr;
        m_ptPipelineTestRaygenPPHDR = nullptr;
        m_ptPipelineEdgeDetection = nullptr;
    }

    virtual std::string GetMaterialSpecializationShader() const override {
        return "PathTracerMaterialSpecializations.hlsl";
    }

private:
    // Blender Live Link (see Rtxpt/LiveLink/LiveLinkServer.h and Docs/LiveLink.md); null unless
    // RTXPT was started with --liveLink.
    std::unique_ptr<rtxpt::livelink::LiveLinkServer> m_liveLink;
};

class AdvancedSample : public SampleBaseApp
{
    std::unique_ptr<Sample> CreateMainRenderPass(donut::app::DeviceManager& deviceManager, const CommandLineOptions& cmdLineOptions) override
    {
        return std::make_unique<AdvancedPathTracer>(deviceManager, cmdLineOptions);
    }
};

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    SplashScreen splashScreen;
    splashScreen.Start(L"loading_splash.png");

    AdvancedSample example;

    // Run the sample app
    const auto status = example.Init(__argc, __argv);

    splashScreen.Stop();

    if (status == SampleBaseApp::InitReturnCodes::Success)
    {
        example.RunMainLoop();

        example.End();
    }

    return static_cast<int>(status);
}
