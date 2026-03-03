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

// IntroRenderer: Simplified renderer for introductory samples
// Currently just uses the base Sample class as-is
// TODO: Override methods to simplify/disable advanced features
class AdvancedPathTracer : public Sample
{
public:
    using Sample::Sample;

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
