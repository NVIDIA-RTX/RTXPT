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

#include "PathTracer/Config.h"
#include "SampleCommon.h"

#include "CommandLine.h"
#include "SampleUI.h"

#include <donut/app/ApplicationBase.h>
#include <donut/core/vfs/VFS.h>
#include <donut/render/GBufferFillPass.h>
#include <donut/render/PixelReadbackPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/app/Camera.h>

#include "RTXDI/RtxdiPass.h"
#include "NRD/NrdIntegration.h"
#include "PathTracer/StablePlanes.hlsli"
#if DONUT_WITH_STREAMLINE
#include <donut/app/StreamlineInterface.h>
#endif

#include "RenderTargets.h"
#include "PostProcess.h"
#include "SampleConstantBuffer.h"
#include "AccumulationPass.h"
#include "ExtendedScene.h"

#include "Lighting/Distant/EnvMapBaker.h"
#include "Lighting/LightsBaker.h"

#include "ShaderDebug.h"

// can be upgraded for special normalmap type (i.e. DXGI_FORMAT_BC5_UNORM) or single channel masks (i.e. DXGI_FORMAT_BC4_UNORM)
enum class TextureCompressionType
{
    Normalmap,
    GenericSRGB,        // maps to BC7_UNORM_SRGB
    GenericLinear,      // maps to BC7_UNORM
};

struct MaterialShadingProperties
{
    bool AlphaTest;
    bool HasTransmission;
    bool NoTransmission;
    bool OnlyDeltaLobes;
    bool NoTextures;
    bool ExcludeFromNEE;

    bool operator==(const MaterialShadingProperties& other) const { return AlphaTest==other.AlphaTest && HasTransmission==other.HasTransmission && NoTransmission==other.NoTransmission && OnlyDeltaLobes==other.OnlyDeltaLobes && NoTextures==other.NoTextures && ExcludeFromNEE==other.ExcludeFromNEE; };
    bool operator!=(const MaterialShadingProperties& other) const { return !(*this==other); }

    static MaterialShadingProperties Compute(const struct MaterialPT & material);
};

class Sample : public donut::app::ApplicationBase
{
    static constexpr uint32_t c_PathTracerVariants   = 6; // see shaders.cfg and CreatePTPipeline for details on variants

public:
    using ApplicationBase::ApplicationBase;

    Sample(donut::app::DeviceManager* deviceManager, CommandLineOptions& cmdLine, SampleUIData& ui);
    virtual ~Sample();

    //std::shared_ptr<donut::vfs::IFileSystem> GetRootFs() const                      { return m_RootFS; }
    std::shared_ptr<donut::engine::ShaderFactory> GetShaderFactory() const          { return m_shaderFactory; }
    std::shared_ptr<donut::engine::Scene>   GetScene() const                        { return m_scene; }
    std::vector<std::string> const &        GetAvailableScenes() const              { return m_sceneFilesAvailable; }
    std::string                             GetCurrentSceneName() const             { return m_currentSceneName; }
    const DebugFeedbackStruct &             GetFeedbackData() const                 { return m_feedbackData; }
    const DeltaTreeVizPathVertex *          GetDebugDeltaPathTree() const           { return m_debugDeltaPathTree; }
    uint                                    GetSceneCameraCount() const             { return (uint)m_scene->GetSceneGraph()->GetCameras().size() + 1; }
    uint &                                  SelectedCameraIndex()                   { return m_selectedCameraIndex; }   // 0 is default fps free flight, above (if any) will just use current scene camera

    void                                    UpdateSubInstanceContents();
    void                                    UploadSubInstanceData(nvrhi::ICommandList* commandList);
    
    void                                    SetUIPick()                             { m_pick = true; }

    std::shared_ptr<donut::engine::Material> FindMaterial( int materialID ) const;
    
    void                                    CollectUncompressedTextures();
    uint                                    UncompressedTextureCount() const        { return (uint)m_uncompressedTextures.size(); }
    bool                                    CompressTextures();
    void                                    SaveCurrentCamera();
    void                                    LoadCurrentCamera();

    float                                   GetCameraVerticalFOV() const            { return m_cameraVerticalFOV; }
    void                                    SetCameraVerticalFOV(float cameraFOV)   { m_cameraVerticalFOV = cameraFOV; }

    float                                   GetAvgTimePerFrame() const;

    bool                                    Init(const std::string& preferredScene);
    void                                    SetCurrentScene(const std::string& sceneName, bool forceReload = false);

    virtual void                            SceneUnloading() override;
    virtual bool                            LoadScene(std::shared_ptr<donut::vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName) override;
    virtual void                            SceneLoaded() override;
    virtual bool                            ShouldRenderUnfocused() override;
    virtual bool                            KeyboardUpdate(int key, int scancode, int action, int mods) override;
    virtual bool                            MousePosUpdate(double xpos, double ypos) override;
    virtual bool                            MouseButtonUpdate(int button, int action, int mods) override;
    virtual bool                            MouseScrollUpdate(double xoffset, double yoffset) override;
    virtual void                            Animate(float fElapsedTimeSeconds) override;

    bool                                    CreatePTPipeline(donut::engine::ShaderFactory& shaderFactory);
    void                                    DestroyOpacityMicromaps(nvrhi::ICommandList* commandList);
    void                                    CreateOpacityMicromaps();
    void                                    CreateBlases(nvrhi::ICommandList* commandList);
    void                                    CreateTlas(nvrhi::ICommandList* commandList);
    void                                    CreateAccelStructs(nvrhi::ICommandList* commandList);
    void                                    UpdateAccelStructs(nvrhi::ICommandList* commandList);
    void                                    BuildTLAS(nvrhi::ICommandList* commandList, uint32_t frameIndex) const;
    void                                    TransitionMeshBuffersToReadOnly(nvrhi::ICommandList* commandList);
    void                                    BackBufferResizing() override;
    void                                    CreateRenderPasses(bool& exposureResetRequired, nvrhi::CommandListHandle initializeCommandList);
    void                                    PreUpdatePathTracing(bool resetAccum, nvrhi::CommandListHandle commandList);
    void                                    PostUpdatePathTracing();
    void                                    UpdatePathTracerConstants( PathTracerConstants & constants, const PathTracerCameraData & cameraData );
    void                                    RtxdiSetupFrame(nvrhi::IFramebuffer* framebuffer, PathTracerCameraData cameraData, uint2 renderDims);

    void                                    Denoise(nvrhi::IFramebuffer* framebuffer);
    void                                    PathTrace(nvrhi::IFramebuffer* framebuffer, const SampleConstants & constants);
    void                                    PreRenderScripts();
    void                                    StreamlinePreRender();
    void                                    Render(nvrhi::IFramebuffer* framebuffer) override;
    void                                    PostProcessAA(nvrhi::IFramebuffer* framebuffer, bool reset);

    void                                    PreUpdateLighting(nvrhi::CommandListHandle commandList, bool& needNewBindings);     // this can (re)allocate buffers depending on scene changes
    void                                    UpdateLighting(nvrhi::CommandListHandle commandList);                               // this will process and fill up all lighting buffers

    donut::math::float2                     ComputeCameraJitter( uint frameIndex );

    std::string                             GetResolutionInfo() const;
    std::string                             GetFPSInfo() const              { return m_fpsInfo; }

    void                                    DebugDrawLine( float3 start, float3 stop, float4 col1, float4 col2 );
    const donut::app::FirstPersonCamera &   GetCurrentCamera( ) const { return m_camera; }

    void                                    ResetSceneTime( ) { m_sceneTime = 0.; }

    bool                                    IsEnvMapLoaded() const { return true; } // with the new EnvMapBaker it's always present (just black)

    const std::shared_ptr<EnvMapBaker> &    GetEnvMapBaker() const { return m_envMapBaker; }
    const std::shared_ptr<LightsBaker> &    GetLightsBaker() const { return m_lightsBaker; }
    const std::shared_ptr<MaterialsBaker> & GetMaterialsBaker() const { return m_materialsBaker; }
    const std::shared_ptr<class OmmBaker> & GetOMMBaker() const { return m_ommBaker; }

private:
    void                                    UpdateCameraFromScene( const std::shared_ptr<donut::engine::PerspectiveCamera> & sceneCamera );
    void                                    UpdateViews( nvrhi::IFramebuffer* framebuffer );
    void                                    DenoisedScreenshot( nvrhi::ITexture * framebufferTexture ) const;

private:
    std::shared_ptr<donut::vfs::RootFileSystem> m_RootFS;

    // scene
    std::vector<std::string>                    m_sceneFilesAvailable;
    std::string                                 m_currentSceneName;
    std::filesystem::path                       m_currentScenePath;
    std::shared_ptr<donut::engine::ExtendedScene>   m_scene;
    double                                      m_sceneTime = 0.;           // if m_ui.LoopLongestAnimation then it loops with longest animation
    uint                                        m_selectedCameraIndex = 0;  // 0 is first person camera, the rest (if any) are scene cameras


    // device setup
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<donut::engine::DescriptorTableManager> m_DescriptorTable;
    std::unique_ptr<donut::engine::BindingCache> m_bindingCache;
    nvrhi::CommandListHandle                    m_commandList;
    nvrhi::BindingLayoutHandle                  m_bindingLayout;
    nvrhi::BindingSetHandle                     m_bindingSet;
    nvrhi::BindingLayoutHandle                  m_bindlessLayout;

    std::unique_ptr<donut::render::TemporalAntiAliasingPass> m_temporalAntiAliasingPass;

    // rendering
    std::unique_ptr<RenderTargets>              m_renderTargets;
    std::vector <std::shared_ptr<donut::engine::Light>> m_lights;
    std::unique_ptr<ToneMappingPass>            m_toneMappingPass;
    nvrhi::BufferHandle                         m_constantBuffer;

    std::vector<SubInstanceData>                m_subInstanceData;
    nvrhi::BufferHandle                         m_subInstanceBuffer;            // per-instance-geometry data, indexed with (InstanceID()+GeometryIndex())
    uint                                        m_subInstanceCount;

    // lighting
    std::string                                 m_envMapLocalPath;
    std::shared_ptr<EnvMapBaker>                m_envMapBaker;
    EnvMapSceneParams                           m_envMapSceneParams;
    std::shared_ptr<LightsBaker>                m_lightsBaker;
    std::shared_ptr<class MaterialsBaker>       m_materialsBaker;
    std::shared_ptr<class OmmBaker>             m_ommBaker;

    // utility
    std::shared_ptr<class GPUSort>              m_gpuSort;

#if USE_PRECOMPUTED_SOBOL_BUFFER
    nvrhi::BufferHandle                         m_precomputedSobolBuffer;
#endif

    // raytracing basics
    nvrhi::rt::AccelStructHandle                m_topLevelAS;

    // camera
    donut::app::FirstPersonCamera               m_camera;
    std::shared_ptr<donut::engine::PlanarView>  m_view;
    std::shared_ptr<donut::engine::PlanarView>  m_viewPrevious;
    float                                       m_cameraVerticalFOV = 60.0f;
    float                                       m_cameraZNear = 0.001f;
    float                                       m_cameraZFar = 100000.0f;
    dm::float3                                  m_lastCamPos = { 0,0,0 };
    dm::float3                                  m_lastCamDir = { 0,0,0 };
    dm::float3                                  m_lastCamUp = { 0,0,0 };


    std::chrono::high_resolution_clock::time_point m_benchStart = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point m_benchLast = std::chrono::high_resolution_clock::now();
    int                                         m_benchFrames = 0;

    std::shared_ptr<PostProcess>                m_postProcess;

    //Debugging and debug viz
    nvrhi::BufferHandle                         m_feedback_Buffer_Gpu;
    nvrhi::BufferHandle                         m_feedback_Buffer_Cpu;
    nvrhi::BufferHandle                         m_debugLineBufferCapture;
    nvrhi::BufferHandle                         m_debugLineBufferDisplay;
    nvrhi::ShaderHandle                         m_linesVertexShader;
    nvrhi::ShaderHandle                         m_linesPixelShader;

    std::vector<DebugLineStruct>                m_cpuSideDebugLines;

    nvrhi::InputLayoutHandle                    m_linesInputLayout;
    nvrhi::GraphicsPipelineHandle               m_linesPipeline;
    nvrhi::BindingLayoutHandle                  m_linesBindingLayout;
    nvrhi::BindingSetHandle                     m_linesBindingSet;
    uint2                                       m_pickPosition = 0u;
    bool                                        m_pick = false;         // this is both for pixel and material debugging
    DebugFeedbackStruct                         m_feedbackData;

    DeltaTreeVizPathVertex                      m_debugDeltaPathTree[cDeltaTreeVizMaxVertices];
    nvrhi::BufferHandle                         m_debugDeltaPathTree_Gpu;
    nvrhi::BufferHandle                         m_debugDeltaPathTree_Cpu;
    nvrhi::BufferHandle                         m_debugDeltaPathTreeSearchStack;

    // all UI-tweakable settings are here
    SampleUIData& m_ui;

    // The command line settings are here
    CommandLineOptions                          m_cmdLine;

    // path tracing
    nvrhi::ShaderLibraryHandle                  m_PTShaderLibrary[c_PathTracerVariants];
    nvrhi::rt::PipelineHandle                   m_PTPipeline[c_PathTracerVariants];
    nvrhi::rt::ShaderTableHandle                m_PTShaderTable[c_PathTracerVariants];
    int                                         m_accumulationSampleIndex = 0;  // accumulated so far in the past, so if 0 this is the first.
    int                                         m_accumulationSampleTarget = 0; // the target to how many we want accumulated (set by UI)

    uint64_t                                    m_frameIndex = 0;
    uint                                        m_sampleIndex = 0;            // per-frame sampling index; same as m_accumulationSampleIndex in accumulation mode, otherwise in realtime based on frameIndex%something 
    SampleConstants                             m_currentConstants = {};

    std::unique_ptr<NrdIntegration>             m_nrd[cStablePlaneCount];       // reminder: when switching between ReLAX/ReBLUR, change settings, reset these to 0 and they'll get re-created in CreateRenderPasses!
    std::unique_ptr<RtxdiPass>                  m_rtxdiPass;
    std::unique_ptr<AccumulationPass>           m_accumulationPass;
    std::shared_ptr<ShaderDebug>                m_shaderDebug;

    nvrhi::ShaderHandle                         m_exportVBufferCS;
    nvrhi::ComputePipelineHandle                m_exportVBufferPSO;

    // texture compression: used but not compressed textures
    std::map<std::shared_ptr<donut::engine::LoadedTexture>, TextureCompressionType> m_uncompressedTextures;

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    // Blue noise texture to be used with stochastic texture filtering
    std::shared_ptr<donut::engine::LoadedTexture>   m_STBNTexture;
#endif

#if DONUT_WITH_STREAMLINE
    donut::app::StreamlineInterface::DLSSSettings   m_recommendedDLSSSettings = {};
    donut::app::StreamlineInterface::DLSSRROptions  m_lastDLSSRROptions;
#endif
    uint2                                       m_renderSize;   // native render resolution
    uint2                                       m_displaySize;  // final output resolution
    float                                       m_displayAspectRatio = 1.0f;

    std::string                                 m_fpsInfo;
    bool                                        m_windowIsInFocus = true;
};

