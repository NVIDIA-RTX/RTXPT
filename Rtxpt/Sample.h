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

#include "Shaders/PathTracer/Config.h"
#include "SampleCommon/SampleCommon.h"

#include "SampleCommon/CommandLine.h"
#include "SampleCommon/BlasPool.h"
#include "SampleCommon/BlasCompactor.h"
#include "SampleUI.h"
#include <memory>

#include <donut/app/ApplicationBase.h>
#include <donut/core/vfs/VFS.h>
#include <donut/render/BloomPass.h>
#include <donut/app/Camera.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>

#include "RTXDI/RtxdiPass.h"
#include "NRD/NrdIntegration.h"
#if DONUT_WITH_STREAMLINE
#include <donut/app/StreamlineInterface.h>
#endif

#include "SampleCommon/RenderTargets.h"
#include "ProcessingPasses/PostProcess.h"
#include "Shaders/SampleConstantBuffer.h"
#include "Shaders/GPUTLASCullData.h"
#include "ProcessingPasses/AccumulationPass.h"
#include "SampleCommon/ExtendedScene.h"

#include "Lighting/Distant/EnvMapBaker.h"
#include "Lighting/LightsBaker.h"

#include "Misc/ShaderDebug.h"

#include <map>
#include "SceneRenderCache.h"
#include <donut/engine/ThreadPool.h>

class DenoisingGuidesBaker;
class CaptureScriptManager;
class ComputePipelineBaker;
class ComputeShaderVariant;
class SharcIntegration;

struct SampleCapabilities
{
    bool NeedsGBuffer              = false; // BaseColor / SpecNormal / RoughnessMetal / MaterialInfo
    bool NeedsDirectLightingBuffer = false; // DirectLighting texture
    bool EnvMapRasterPrecompute    = false; // EnvMapBaker raster prefilter path
    bool UseNeeAtLightSampling     = true;  // false ⇒ ImportanceSamplingType = 0, useNeeat = false
};

class Sample : public donut::app::ApplicationBase
{
    // static constexpr uint32_t c_PathTracerVariants   = 6; // see shaders.cfg and CreatePTPipeline for details on variants

public:
    using ApplicationBase::ApplicationBase;

    Sample(donut::app::DeviceManager& deviceManager,
        const CommandLineOptions& cmdLine);
    virtual ~Sample();

    //std::shared_ptr<donut::vfs::IFileSystem> GetRootFs() const                      { return m_RootFS; }
    std::shared_ptr<donut::engine::ShaderFactory> GetShaderFactory() const          { return m_shaderFactory; }
    std::shared_ptr<donut::engine::CommonRenderPasses> GetCommonPasses() const      { return m_CommonPasses; }
    std::shared_ptr<donut::engine::Scene>   GetScene() const                        { return m_scene; }
    std::vector<std::string> const &        GetAvailableScenes() const              { return m_sceneFilesAvailable; }
    std::string                             GetCurrentSceneName() const             { return m_currentSceneName; }
    const DebugFeedbackStruct &             GetFeedbackData() const                 { return m_feedbackData; }
    const DeltaTreeVizPathVertex *          GetDebugDeltaPathTree() const           { return m_debugDeltaPathTree; }
    uint                                    GetSceneCameraCount() const             { return (uint)m_scene->GetSceneGraph()->GetCameras().size() + 1; }
    std::string                             GetSceneCameraName(uint index) const;   // the scene's own name for camera `index`, or "Scene cam <index>" when it has none
    uint &                                  SelectedCameraIndex()                   { return m_selectedCameraIndex; }   // 0 is default fps free flight, above (if any) will just use current scene camera
    const std::unique_ptr<class GameScene> &     GetGame() const                   { return m_sampleGame; }

    void                                    UpdateSubInstanceContents();
    void                                    UploadSubInstanceData(nvrhi::ICommandList* commandList);
    
    void                                    SetUIPick()                             { m_pick = true; }

    std::shared_ptr<donut::engine::Material> FindMaterial( int materialID ) const;
    
    void                                    CollectUncompressedTextures();
    auto &                                  GetUncompressedTextures()               { return m_uncompressedTextures; }
    void                                    SaveCurrentCamera() const;
    void                                    LoadCurrentCamera();
    std::string                             GetCurrentCameraPosDirUp() const;
    bool                                    SetCurrentCameraPosDirUp(const std::string & val);

    float                                   GetCameraVerticalFOV() const            { return m_cameraVerticalFOV; }
    void                                    SetCameraVerticalFOV(float cameraFOV)   { m_cameraVerticalFOV = cameraFOV; }

    float                                   GetAvgTimePerFrame() const;

    void                                    Init(const std::string& preferredScene, const std::shared_ptr<donut::engine::ShaderFactory>& shaderFactory);
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

    virtual const char *                    GetWindowTitle() { return "Untitled"; };

    void                                    GetPTPipelineGlobalMacros(std::vector<donut::engine::ShaderMacro> & macros);
    bool                                    CreatePTPipeline(donut::engine::ShaderFactory& shaderFactory);
    void                                    DestroyOpacityMicromaps(nvrhi::ICommandList* commandList);
    void                                    CreateOpacityMicromaps();
    void                                    CreateBlases(nvrhi::ICommandList* commandList);
    void                                    ReleaseBlases();
    void                                    CreateTlas(nvrhi::ICommandList* commandList);
    void                                    CreateAccelStructs(nvrhi::ICommandList* commandList);
    void                                    RecreateAccelStructs(nvrhi::ICommandList* commandList);
    void                                    UpdateSkinnedBLASs(nvrhi::ICommandList* commandList, uint32_t frameIndex) const;
    void                                    BuildTLAS(nvrhi::ICommandList* commandList) const;
    // Sizes m_tlasInstanceBuffer.
    void                                    EnsureTLASInstanceBuffer(size_t instanceCount);
    // Picks the GPU or CPU builder for this frame; the single TLAS call site in Render().
    void                                    UpdateTLAS(nvrhi::ICommandList* commandList);
    // GPU cull-and-build - one compute thread per mesh instance fills m_tlasInstanceBuffer, which then
    // feeds buildTopLevelAccelStructFromBuffer (Shaders/TLASCull.hlsl). Returns false when its resources
    // are not ready, so UpdateTLAS can fall back rather than leave the TLAS stale.
    bool                                    BuildTLASGPU(nvrhi::ICommandList* commandList);
    bool                                    EnsureGPUTLASResources();
    // (Re)builds the static per-mesh bounds and per-instance BLAS-address tables the cull pass reads.
    void                                    UploadTLASInstanceData(nvrhi::ICommandList* commandList);
    void                                    TransitionMeshBuffersToReadOnly(nvrhi::ICommandList* commandList);
    void                                    BackBufferResizing() override;
    void                                    CreateRenderPasses(bool& exposureResetRequired, nvrhi::CommandListHandle initializeCommandList);
    virtual void                            CreateCustomRenderPasses();
    // DLSS-RR guide production lives in DenoisingGuidesBaker, which is owned by the derived sample; base does nothing.
    virtual void                            PrepareDLSSRRGuides(nvrhi::ICommandList* commandList) { }
    // Whether this sample's frame up to the DLSS-RR inputs may be recorded on the compute queue (see ShouldRecordPrefixAsync)
    virtual bool                            SupportsAsyncPrefix() const { return false; }
    bool                                    IsAsyncPrefixAvailable() const { return SupportsAsyncPrefix() && m_asyncPrefixCapable; }
    void                                    PreUpdatePathTracing(bool resetAccum, nvrhi::CommandListHandle commandList);
    void                                    PostUpdatePathTracing();
    void                                    UpdatePathTracerConstants( PathTracerConstants & constants, const PathTracerCameraData & cameraData );
    // UI firefly threshold -> shader fireflyFilterThreshold; the one place for this scaling (SHaRC training uses it too)
    static float                            FireflyFilterThreshold(float uiThreshold, float preExposedGrayLuminance);
    void                                    RtxdiSetupFrame(nvrhi::IFramebuffer* framebuffer, PathTracerCameraData cameraData, uint2 renderDims);

    // Extendable sample interface
    virtual void                            SampleRenderCode(nvrhi::IFramebuffer* framebuffer, nvrhi::CommandListHandle commandList, const SampleConstants& constants, bool needNewPasses) = 0; // TODO: Rename this
    // Default implementations create/destroy the shared post-process raygen pipelines
    // (m_ptPipelineTestRaygenPPHDR / m_ptPipelineEdgeDetection). Derived classes must
    // chain to these (call Sample::CreateRTPipelines() / Sample::DestroyRTPipelines())
    // so the shared PostProcess UI toggles work in every sample.
    virtual void                            CreateRTPipelines();
    virtual void                            DestroyRTPipelines();
    virtual std::string                     GetMaterialSpecializationShader() const = 0;
    virtual std::unique_ptr<SampleUI>       CreateUI(donut::app::DeviceManager* deviceManager, class SampleBaseApp& baseApp, SampleUIData& uiData, bool NVAPI_SERSupported, const struct CommandLineOptions& cmdLine) = 0;

    void                                    Denoise(nvrhi::IFramebuffer* framebuffer);
    void                                    Render(nvrhi::IFramebuffer* framebuffer) override;
    void                                    PostProcessAA(nvrhi::IFramebuffer* framebuffer, bool reset);

    void                                    PreUpdateLighting(nvrhi::CommandListHandle commandList, bool& needNewBindings);     // this can (re)allocate buffers depending on scene changes
    void                                    UpdateLighting(nvrhi::CommandListHandle commandList);                               // this will process and fill up all lighting buffers

    donut::math::float2                     ComputeCameraJitter( uint64_t frameIndex );

    std::string                             GetResolutionInfo() const;
    std::string                             GetFPSInfo() const              { return m_fpsInfo; }

    void                                    DebugDrawLine( float3 start, float3 stop, float4 col1, float4 col2 );
    const donut::app::FirstPersonCamera &   GetCurrentCamera( ) const { return m_camera; }
    const auto &                            GetCurrentView( ) const { return m_view; }

    void                                    SetSceneTime( double sceneTime );
    double                                  GetSceneTime( );

    bool                                    IsEnvMapLoaded() const      { return true; } // with the new EnvMapBaker it's always present (just black)
    const std::string &                     GetEnvMapLocalPath()        { return m_envMapLocalPath; }
    const std::string &                     GetEnvMapOverrideSource()   { return m_envMapOverride; }
    void                                    SetEnvMapOverrideSource(const std::string & envMapOverride);
    const std::vector<std::filesystem::path> & GetEnvMapMediaList()     { return m_envMapMediaList; }

    const std::shared_ptr<EnvMapBaker> &    GetEnvMapBaker() const { return m_envMapBaker; }
    const std::shared_ptr<LightsBaker> &    GetLightsBaker() const { return m_lightsBaker; }
    const std::shared_ptr<MaterialsBaker> & GetMaterialsBaker() const { return m_materialsBaker; }
    const std::shared_ptr<class OmmBaker>   GetOMMBaker() const { return m_ommBaker; }
    const std::unique_ptr<class ZoomTool> & GetZoomTool() const { return m_zoomTool; }
    const ToneMappingPass *                 GetToneMappingPass() const { return m_toneMappingPass.get(); }    // null until render passes are created
    donut::engine::BindingCache &           GetBindingCache() { return *m_bindingCache; }

    // SHaRC radiance cache. Null unless the derived sample owns one (AdvancedPathTracer).
    virtual class SharcIntegration*         GetSharc() { return nullptr; }

    // Called by CaptureScriptManager right before it ends the process with std::exit (no destructors run), after the
    // last frame's command list has been submitted.
    virtual void                            BeforeCaptureExit() { }

    GLFWwindow *                            GetGLFWWindow() const { return GetDeviceManager()->GetWindow(); }

    int                                     GetAccumulationSampleIndex() const { return m_accumulationSampleIndex; }

    uint2                                   GetRenderSize() const               { return m_renderSize;  } // native render resolution
    uint2                                   GetDisplaySize() const              { return m_displaySize; } // final output resolution

    const std::unique_ptr<CaptureScriptManager> & GetCaptureScriptManager() const { return m_captureScriptManager; }

    bool                                    HasAsyncLoadingInProgress() const   { return m_asyncLoadingInProgress || m_ui.ShaderAndACRefreshDelayedRequest > 0; }

    bool                                    AccumulationCompleted() const       { return m_accumulationCompleted; }

    double                                  GetAvgTimeInRender() const          { return m_avgTimeInLastRender; }
    double                                  GetAvgTimeInPresent() const         { return m_avgTimeInLastPresent; }
    double                                  GetAvgTimeInFrameGap() const        { return m_avgTimeInLastFrameGap; }

protected:
    // Frame lifecycle hooks - internal to the Sample render flow, not part of the public surface.
    // PostRender() is invoked via RAII_SCOPE at the top of Render() so it runs on every exit path.
    void                                    PreRender();
    void                                    StreamlinePreRender();
    void                                    PostRender();

    // Called when render targets have been recreated (e.g. after window resize)
    virtual void OnRenderTargetsRecreated() { }

    // Called during binding set creation to allow derived classes to add custom bindings
    // The environment IBL / reflection texture slots (t80-t83) have null placeholders by default
    virtual void AddCustomBindings(nvrhi::BindingSetDesc& bindingSetDesc) { }

    // Called at the end of GetPTPipelineGlobalMacros so derived samples can contribute
    // their own global path-tracer shader macros. A change in the produced macros is detected by 
    // PTPipelineBaker::Update and triggers a recompile.
    virtual void GetCustomPTPipelineGlobalMacros(std::vector<donut::engine::ShaderMacro>& /*macros*/) { }

    // Called every frame after LightsBaker has produced its outputs and before the
    // global SampleConstants are uploaded to the GPU. Derived classes can run their
    // own per-frame light update work here (recording GPU work on the provided command
    // list) and fill in additional fields of LightControlData.
    virtual void UpdateCustomLighting(nvrhi::ICommandList* /*commandList*/, struct LightControlData& /*lightControl*/) { }

    // Called every frame just before the global SampleConstants are uploaded to the GPU, so derived
    // samples can fill sample-specific sub-structs. The command list is provided for any associated
    // GPU work (eg cache clear). Must run before the upload.
    virtual void UpdateCustomConstants(struct SampleConstants& /*constants*/, nvrhi::ICommandList* /*commandList*/) { }

    // Called every frame for derived samples to (re)create resolution-dependent resources and the bindless
    // descriptors that go with them. Deliberately much earlier than UpdateCustomConstants: it runs before
    // UpdateBindlessValidationData snapshots the descriptor table and before the binding set is rebuilt. A
    // descriptor created any later is described as 'not allocated' for that frame, which trips the bindless
    // validation trap in Debug (see Rtxpt/Shaders/Bindless.hlsli) even though the descriptor itself is fine.
    virtual void UpdateCustomResources() { }

    // Invalidates the binding set, forcing recreation on next frame
    // Call this from derived classes when custom bindings need to be updated
    void InvalidateBindingSet() { m_bindingSet = m_bindingSetMain = m_bindingSetTail = m_bindingSetPrefixEarly = m_bindingSetPrefixLate = nullptr; }
    // Refreshes t_BindlessValidationData (see Rtxpt/Shaders/Bindless.hlsli). Sets needNewBindings if the buffer
    // had to be reallocated, since the binding set holds a raw reference to it. Must run after everything that
    // can create descriptors this frame and before the binding set is rebuilt.
    void UpdateBindlessValidationData(nvrhi::ICommandList* commandList, bool& needNewBindings);
    void RecreateBindingSet();
    
    // Per-sample feature flags. Set in derived constructor body before
    // CreateRenderPasses / BackBufferResizing run.
    SampleCapabilities                          m_sampleCapabilities;

    // all UI-tweakable settings are here
    SampleUIData& m_ui;
    std::unique_ptr<RtxdiPass>                  m_rtxdiPass;
    std::unique_ptr<RenderTargets>              m_renderTargets;
    nvrhi::BindingLayoutHandle                  m_bindingLayout;
    nvrhi::BindingLayoutHandle                  m_bindlessLayout;
    // SM6.6 dynamic resources (ResourceDescriptorHeap) - see Rtxpt/Shaders/Bindless.hlsli. Consumes no root
    // parameter: it only makes nvrhi set D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED. It MUST be
    // the LAST layout of any pipeline that uses it - nvrhi pairs binding set i with pipeline layout i by
    // position, so a pipeline can only safely be dispatched with fewer binding sets than layouts when the
    // omitted ones are trailing (which is what lets IntroSample keep passing two).
    nvrhi::BindingLayoutHandle                  m_bindlessHeapLayout;
    nvrhi::BindingSetHandle                     m_bindingSet;               // the global set to bind now: one of the below
    nvrhi::BindingSetHandle                     m_bindingSetMain;
    // Variants of m_bindingSetMain for the async prefix and the post-ShaderDebug tail, see RecreateBindingSet
    nvrhi::BindingSetHandle                     m_bindingSetPrefixEarly;
    nvrhi::BindingSetHandle                     m_bindingSetPrefixLate;
    nvrhi::BindingSetHandle                     m_bindingSetTail;
    nvrhi::TextureHandle                        m_prefixDummyUAV;
    nvrhi::BufferHandle                         m_dummyRawBufferUAV;

    std::shared_ptr<donut::engine::DescriptorTableManager> m_DescriptorTable;
    // Home for descriptors that shaders fetch through ResourceDescriptorHeap. Deliberately a SEPARATE table from
    // m_DescriptorTable: on Vulkan ResourceDescriptorHeap maps to the mutable descriptor set, which is not the
    // set m_DescriptorTable owns, and IDescriptorTable::getFirstDescriptorIndexInHeap() hard-returns 0 there
    // (vulkan-backend.h). Allocating from a table built on the mutable layout makes DescriptorHandle::
    // GetIndexInHeap() mean the right thing on both backends - an absolute heap index on D3D12, where both
    // tables share one shader-visible heap, and a slot within the mutable set on Vulkan.
    std::shared_ptr<donut::engine::DescriptorTableManager> m_bindlessHeapDescriptorTable;
    // Per-descriptor metadata for the bindless validation layer, bound at t50. Always allocated and always bound
    // so the global binding layout is identical in Debug and Release; only populated when the shader side is
    // actually compiled in (RTXPT_SHADER_DEBUG).
    nvrhi::BufferHandle                         m_bindlessValidationBuffer;
    std::vector<uint32_t>                       m_bindlessValidationData;

    // QoL accessors for derived samples
    const donut::engine::PlanarView& GetView() const { return *m_view; }
    std::shared_ptr<class PTPipelineBaker>  GetRTPipelineBaker() const { return m_ptPipelineBaker; }
    std::shared_ptr<donut::engine::DescriptorTableManager> GetBindlessHeapDescriptorTable() const { return m_bindlessHeapDescriptorTable; }
    std::shared_ptr<ComputePipelineBaker>   GetComputePipelineBaker() const { return m_computePipelineBaker; }

    // Shared post-process raygen pipelines, created/destroyed by
    // Sample::CreateRTPipelines / Sample::DestroyRTPipelines.
    std::shared_ptr<class PTPipelineVariant>    m_ptPipelineTestRaygenPPHDR;
    std::shared_ptr<class PTPipelineVariant>    m_ptPipelineEdgeDetection;

    std::unique_ptr<CaptureScriptManager>       m_captureScriptManager;

private:
    void                                    UpdateCameraFromScene( const std::shared_ptr<donut::engine::PerspectiveCamera> & sceneCamera );
    void                                    UpdateViews( nvrhi::IFramebuffer* framebuffer );
    void                                    UpdateTemporalAntiAliasingPass();
    void                                    DenoisedScreenshot( nvrhi::ITexture * framebufferTexture ) const;
    void                                    PostProcessPreToneMapping(nvrhi::ICommandList* commandList, const donut::engine::ICompositeView& compositeView);
    void                                    PostProcessPostToneMapping(nvrhi::ICommandList* commandList, const donut::engine::ICompositeView& compositeView);
    bool                                    ShouldRecordPrefixAsync(bool needNewPasses);
    void                                    CloseShaderDebug();
    std::string                             GetEnvMapActualPath() const;
    // DLSS-RR reads copies of Depth and motion vectors, so the next frame's async prefix can rewrite the originals meanwhile
    bool                                    UseRRDepthMVCopies() const { return m_ui.EnableAsyncPrefix && IsAsyncPrefixAvailable() && m_ui.RealtimeMode && m_ui.RealtimeAA == 3; }

protected:
    std::shared_ptr<donut::vfs::RootFileSystem> m_RootFS;

    // scene
    std::vector<std::string>                    m_sceneFilesAvailable;
    std::string                                 m_currentSceneName;
    std::filesystem::path                       m_currentScenePath;
    std::shared_ptr<ExtendedScene>              m_scene;
    double                                      m_sceneTime = 0.;           // if m_ui.LoopLongestAnimation then it loops with longest animation
    float                                       m_lastDeltaTime = 0.0f;
    uint                                        m_selectedCameraIndex = 0;  // 0 is first person camera, the rest (if any) are scene cameras

    // device setup
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_CommonPasses;
    std::unique_ptr<donut::engine::BindingCache> m_bindingCache;
    nvrhi::CommandListHandle                    m_commandList;              // the list being recorded: m_prefixCommandList during an async prefix
    nvrhi::CommandListHandle                    m_graphicsCommandList;
    nvrhi::CommandListHandle                    m_prefixCommandList;
    bool                                        m_lastFrameAsyncPrefix = false;
    bool                                        m_asyncPrefixCapable = false;   // D3D12 with enhanced barriers, which the compute-queue state audit assumed
    uint64_t                                    m_lastGraphicsSubmit = 0;       // the async prefix waits on this before rewriting what DLSS-RR reads
    int64_t                                     m_lastPrefixPassesKey = -1;     // see ShouldRecordPrefixAsync
    void*                                       m_dlssgInputsFence = nullptr;   // DLSSGState::inputsProcessingCompletionFence (ID3D12Fence), while DLSS-G is on
    uint64_t                                    m_dlssgInputsFenceValue = 0;

    std::unique_ptr<donut::render::TemporalAntiAliasingPass> m_temporalAntiAliasingPass;   // only exists while TAA is active
    bool                                        m_temporalAntiAliasingFeedbackValid = false;  // false until a newly created pass writes its history

    // rendering
    std::vector <std::shared_ptr<donut::engine::Light>> m_lights;
    std::unique_ptr<donut::render::BloomPass>   m_bloomPass;
    std::unique_ptr<ToneMappingPass>            m_toneMappingPass;
    nvrhi::BufferHandle                         m_constantBuffer;

    std::vector<SubInstanceData>                m_subInstanceData;
    nvrhi::BufferHandle                         m_subInstanceBuffer;            // per-instance-geometry data, indexed with (InstanceID()+GeometryIndex())
    uint                                        m_subInstanceCount;

    // lighting
    std::string                                 m_envMapLocalPath;
    
    std::filesystem::path                       m_envMapMediaFolder;
    std::vector<std::filesystem::path>          m_envMapMediaList;

    std::string                                 m_envMapOverride;

    std::shared_ptr<EnvMapBaker>                m_envMapBaker;

    bool                                        m_bakeDirectionalLightsIntoEnvMap = true;
    EnvMapSceneParams                           m_envMapSceneParams;
    std::shared_ptr<LightsBaker>                m_lightsBaker;
    std::shared_ptr<class MaterialsBaker>       m_materialsBaker;
    std::shared_ptr<class OmmBaker>             m_ommBaker;
    std::shared_ptr<class PTPipelineBaker>      m_ptPipelineBaker;
    std::shared_ptr<ComputePipelineBaker>       m_computePipelineBaker;

    // utility
    std::shared_ptr<class GPUSort>              m_gpuSort;

    // raytracing basics
    nvrhi::rt::AccelStructHandle                m_topLevelAS;
    mutable std::vector<nvrhi::rt::InstanceDesc> m_tlasInstanceDescs;  // persistent scratch for BuildTLAS - kept across frames so its capacity is reused instead of reallocating every frame (mutable: BuildTLAS is const and this is scratch, not logical state)

    // === GPU TLAS cull-and-build (UpdateTLAS / BuildTLASGPU / Shaders/TLASCull.hlsl) ===
    nvrhi::BufferHandle                         m_tlasInstanceBuffer;      // GPUTLASInstanceDesc per mesh instance; filled by the cull dispatch, never from the CPU
    nvrhi::BufferHandle                         m_tlasInstanceDataBuffer;  // TLASInstanceData per mesh instance
    nvrhi::BufferHandle                         m_tlasMeshBoundsBuffer;    // TLASObjectBounds per mesh
    uint32_t                                    m_tlasInstanceCount = 0;
    // Set whenever BLAS handles change (CreateAccelStructs); consumed by UploadTLASInstanceData.
    bool                                        m_tlasInstanceDataDirty = true;
    uint32_t                                    m_tlasInstanceDataOmmBuildsLeft = 0xFFFFFFFFu;
    uint64_t                                    m_tlasInstanceDataOmmGeneration = 0;
    nvrhi::ShaderHandle                         m_tlasCullCS;
    nvrhi::ComputePipelineHandle                m_tlasCullPSO;
    nvrhi::BindingLayoutHandle                  m_tlasCullBindingLayout;
    nvrhi::BindingSetHandle                     m_tlasCullBindingSet;
    nvrhi::BufferHandle                         m_tlasCullCB;
    // Raw handles the current binding set was built against, to detect a buffer swap.
    nvrhi::IBuffer*                             m_tlasCullBoundInstanceData = nullptr;
    nvrhi::IBuffer*                             m_tlasCullBoundInstanceAccel = nullptr;
    nvrhi::IBuffer*                             m_tlasCullBoundMeshBounds = nullptr;
    nvrhi::IBuffer*                             m_tlasCullBoundOutput = nullptr;
    // Dedicated to per-frame work. Must NOT be shared with async loading: ThreadPool::WaitForTasks() waits on a
    // pool-wide counter, so an unrelated long task queued elsewhere would stall the frame.
    //
    // (The ~ThreadPool lost-wakeup race this comment used to flag as a known issue was fixed upstream by Donut
    // commit d380178, which now sets m_terminate under m_mutex; that fix came in with the Donut submodule bump.)
    mutable donut::engine::ThreadPool           m_frameWorkerPool;

    // Per-frame scene-derived cache; opened just after Scene::Refresh(), closed in PostRender(). mutable because const
    // render helpers (BuildTLAS) read it and its lazy fill is a cache detail, not logical state.
    mutable SceneRenderCache                    m_sceneRenderCache;

    // camera
    donut::app::FirstPersonCamera               m_camera;
    std::shared_ptr<donut::engine::PlanarView>  m_view;
    std::shared_ptr<donut::engine::PlanarView>  m_viewPrevious;
    PathTracerCameraData                        m_lastCameraData = {};   // last frame's ptConsts.camera; prevCamera's source
    float                                       m_cameraVerticalFOV = dm::radians(60.0f);
    float                                       m_cameraZNear = 0.001f;
    float                                       m_cameraZFar = 100000.0f;
    dm::float3                                  m_lastCamPos = { 0,0,0 };
    dm::float3                                  m_lastCamDir = { 0,0,0 };
    dm::float3                                  m_lastCamUp = { 0,0,0 };


    std::chrono::high_resolution_clock::time_point m_benchStart = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point m_benchLast = std::chrono::high_resolution_clock::now();
    int                                         m_benchFrames = 0;

    std::chrono::high_resolution_clock::time_point m_renderStartTime;
    double                                      m_timeInLastRender = 0.0;
    double                                      m_avgTimeInLastRender = 0.0;

    std::chrono::high_resolution_clock::time_point m_presentStartTime;
    double                                      m_timeInLastPresent = 0.0;
    double                                      m_avgTimeInLastPresent = 0.0;

    std::chrono::high_resolution_clock::time_point m_afterPresentTime;
    double                                      m_timeInLastFrameGap = 0.0;
    double                                      m_avgTimeInLastFrameGap = 0.0;

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
    DebugFeedbackStruct                         m_feedbackData = {};    // zero-init: only written by the pick readback, and the debug UI reads it unconditionally

    DeltaTreeVizPathVertex                      m_debugDeltaPathTree[cDeltaTreeVizMaxVertices];
    nvrhi::BufferHandle                         m_debugDeltaPathTree_Gpu;
    nvrhi::BufferHandle                         m_debugDeltaPathTree_Cpu;
    nvrhi::BufferHandle                         m_debugDeltaPathTreeSearchStack;

    // The command line settings are here
    const CommandLineOptions&                   m_cmdLine;

    // path tracing
    int                                         m_accumulationSampleIndex = 0;  // accumulated so far in the past, so if 0 this is the first.

    uint64_t                                    m_frameIndex = 0;
    uint                                        m_sampleIndex = 0;            // per-frame sampling index; same as m_accumulationSampleIndex in accumulation mode, otherwise in realtime based on frameIndex%something 
    SampleConstants                             m_currentConstants = {};

    std::unique_ptr<NrdIntegration>             m_nrd[cDenoisingLayerCount];       // reminder: when switching between ReLAX/ReBLUR, change settings, reset these to 0 and they'll get re-created in CreateRenderPasses!
    std::unique_ptr<AccumulationPass>           m_accumulationPass;
    std::shared_ptr<ShaderDebug>                m_shaderDebug;

    nvrhi::ShaderHandle                         m_exportVBufferCS;
    nvrhi::ComputePipelineHandle                m_exportVBufferPSO;

    // texture compression: used but not compressed textures
    std::map<std::shared_ptr<donut::engine::LoadedTexture>, TextureCompressionType> m_uncompressedTextures;

#if DONUT_WITH_STREAMLINE
    donut::app::StreamlineInterface::DLSSSettings   m_recommendedDLSSSettings = {};
    donut::app::StreamlineInterface::DLSSRROptions  m_lastDLSSRROptions;
#endif
    uint2                                       m_renderSize;   // native render resolution
    uint2                                       m_displaySize;  // final output resolution
    float                                       m_displayAspectRatio = 1.0f;

    std::string                                 m_fpsInfo;
    bool                                        m_windowIsInFocus = true;

    std::unique_ptr<class GameScene>            m_sampleGame;

    ProgressBar                                 m_progressLoading;
    ProgressBar                                 m_progressInitializingRenderer;

    std::unique_ptr<class ZoomTool>             m_zoomTool;

    bool                                        m_asyncLoadingInProgress = false;

    // Static BLASes live suballocated inside these; a skinned one stays NVRHI's, because it is refit
    // in place every frame and so is never built with AllowCompaction.
    std::unique_ptr<bvh::BlasPool>              m_blasPool;
    std::unique_ptr<bvh::BlasCompactor>         m_blasCompactor;
    // Pool generation the BLAS address table was baked at; any allocation or free moves it.
    uint64_t                                    m_tlasInstanceDataBlasGeneration = ~0ull;
    bool                                        m_accumulationCompleted = false;
};

