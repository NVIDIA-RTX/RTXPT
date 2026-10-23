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
#include "IntroSampleUI.h"
#include "AORenderer.h"
#include "DepthHierarchyRenderer.h"
#include "ReflectionRenderer.h"
#include "RTGIRenderer.h"
#include "IndirectPathTracer.h"
#include "SHaRCRenderer.h"
#include <Rtxdi/DI/ReSTIRDI.h>
#include <SampleCommon/SampleBaseApp.h>
#include <vector>
#include <array>

struct PrepareLightsTask;       // RTXDI/ShaderParameters.h
struct PolymorphicLightInfo;    // Shaders/PathTracer/Lighting/PolymorphicLight.h
struct PolymorphicLightInfoEx;  // Shaders/PathTracer/Lighting/PolymorphicLight.h
#include <SampleCommon/PTPipelineBaker.h>
#include <donut/engine/View.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/CommonRenderPasses.h>

// Simplified renderer for introductory samples with improved reflections.
// Features:
// - Environment IBL: GGX-prefiltered specular + SH-based diffuse irradiance
//   derived from the environment map / procedural sky
// - Hierarchical screen-space reflections
// - Split-sum IBL with BRDF LUT
// - Ground truth ambient occlusion (GTAO)
class IntroPathTracer : public Sample
{
public:

    IntroPathTracer(donut::app::DeviceManager& deviceManager, const CommandLineOptions& cmdLine);

    virtual const char* GetWindowTitle() override { return "IntroSample"; };    // future: change to ScalableSample?

    virtual void SampleRenderCode(nvrhi::IFramebuffer* framebuffer, nvrhi::CommandListHandle commandList, const SampleConstants& constants, bool needNewPasses) override;

    // Derives the GGX-prefiltered specular + diffuse-irradiance cubes from the environment
    // map / procedural sky, reprocessing only when the environment contents change.
    void ProcessEnvIBL(nvrhi::CommandListHandle commandList);

    void DirectLightingPass(nvrhi::CommandListHandle commandList, uint sharcCapacity);

    void ReSTIRDIDirectLightingPass(nvrhi::CommandListHandle commandList, uint sharcCapacity);

    void DeferredLightingCombine(nvrhi::CommandListHandle commandList, bool useRTReflections);

    void PrepareGuideBuffers(nvrhi::CommandListHandle commandList);

    void CopyGBufferHistory(nvrhi::CommandListHandle commandList);

    void PopulateGBuffer(nvrhi::CommandListHandle commandList);

    void ReferencePathTracer(nvrhi::CommandListHandle commandList);

    virtual void CreateRTPipelines() override;

    virtual void DestroyRTPipelines() override;

    virtual void CreateCustomRenderPasses() override;

    virtual std::string GetMaterialSpecializationShader() const override;
    
    virtual void SceneLoaded() override;
    
    // Override to add reflection textures to the main binding set
    virtual void AddCustomBindings(nvrhi::BindingSetDesc& bindingSetDesc) override;

    // Prepares the Intro RTXDI light buffers (local analytic + directional + emissive triangles)
    // and fills the light-region indices in lightControl. Runs unconditionally every frame, before
    // any pass that reads the buffer (realtime direct lighting / RTGI as well as the reference PT),
    // so all of them can rely on a fully-populated buffer. Called from SampleRenderCode.
    void PrepareLights(nvrhi::ICommandList* commandList, struct LightControlData& lightControl);

    // The procedural sky's sun is not a scene-graph Light (it is baked into the environment cubemap),
    // so it never appears in m_lights. Synthesize a directional analytic light from the sky's sun
    // parameters and append it to the infinite-light group. No-op when a procedural sky is not active.
    void AddProceduralSunLight(std::vector<PolymorphicLightInfo>& infiniteBase, std::vector<PolymorphicLightInfoEx>& infiniteEx);

    virtual void OnRenderTargetsRecreated() override;

    virtual std::unique_ptr<SampleUI> CreateUI(donut::app::DeviceManager* deviceManager, class SampleBaseApp& baseApp, SampleUIData& uiData, bool NVAPI_SERSupported, const struct CommandLineOptions& cmdLine) override;

    // Intro-only UI/tuning state (step dropdown + ReSTIR DI knobs). Owned here rather
    // than in the shared SampleUIData; IntroSampleUI edits it via this accessor.
    IntroSampleUIData&                          GetIntroUIData() { return m_introUIData; }

private:
    IntroSampleUIData                           m_introUIData;

    std::unique_ptr<AORenderer>                 m_aoRenderer;
    std::unique_ptr<DepthHierarchyRenderer>     m_depthHierarchyRenderer;
    std::unique_ptr<ReflectionRenderer>         m_reflectionRenderer;
    nvrhi::TextureHandle                        m_prevDepth;       // Full-res R32_FLOAT:  previous frame depth
    nvrhi::TextureHandle                        m_prevSpecNormal;  // Full-res R32_UINT:   previous frame packed shading normal (ReSTIR DI temporal reuse)
    std::shared_ptr<class PTPipelineVariant>    m_GBufferPipeline;
    std::shared_ptr<class PTPipelineVariant>    m_realTimePathTracer;

    // Direct lighting pass - managed by ComputePipelineBaker for hot reload
    std::shared_ptr<ComputeShaderVariant>       m_directLightingVariant;
    std::shared_ptr<ComputeShaderVariant>       m_stochasticLightingVariant;       // 1-sample MIS direct lighting
    std::shared_ptr<ComputeShaderVariant>       m_RTXDIStreamingRISVariant;        // RTXDI local-light Streaming RIS
    std::shared_ptr<ComputeShaderVariant>       m_reSTIRDIInitialVariant;
    std::shared_ptr<ComputeShaderVariant>       m_reSTIRDITemporalVariant;
    std::shared_ptr<ComputeShaderVariant>       m_reSTIRDISpatialVariant;
    std::shared_ptr<ComputeShaderVariant>       m_reSTIRDIFinalVariant;

    // Deferred lighting combine - managed by ComputePipelineBaker for hot reload
    std::shared_ptr<ComputeShaderVariant>       m_deferredLightingVariant;
    std::shared_ptr<ComputeShaderVariant>       m_rtgiDebugOutputVariant;
    std::shared_ptr<ComputeShaderVariant>       m_sharcDebugCacheVariant;

    // Simplified hit groups, baked from the shared MaterialsBaker against RTXPT_HIT_SET_SIMPLIFIED.
    std::shared_ptr<class PTPipelineBaker>      m_simplifiedRTPipelines;

    // Allocates the environment IBL textures (prefiltered specular + irradiance) if needed
    void AllocateEnvIBLTextures();

    // RTXDI resources for the learning path. The Intro sample maintains its own
    // analytic-light buffer (separate from LightsBaker's t_Lights) so the two
    // bake implementations can run side by side. The buffer is sized for two
    // ping-pong regions; the active sub-regions (local + infinite) are advertised
    // through LightControlData::LocalRegion* / InfiniteRegion* fields in the
    // global SampleConstants.
    //
    // Mirrors the LightsBaker layout: the Base half (PolymorphicLightInfo) and the
    // Extended half (PolymorphicLightInfoEx) live in two parallel SoA buffers so that
    // shaders that only need Base don't pay for Extended cache traffic, and so the
    // bake-cost comparison vs LightsBaker is apples-to-apples.
    void EnsureIntroRTXDIResources(uint32_t maxAnalyticLights);

    // Emissive-triangle bake (mirrors RTXDI's PrepareLightsPass, see Shaders/IntroSample/IntroPrepareLights.hlsl).
    // Enumerates emissive geometries into one PrepareLightsTask each and returns the total triangle count T.
    struct EmissiveGeometryKey
    {
        const donut::engine::MeshInstance* instance;
        const donut::engine::MeshGeometry* geometry;
        uint32_t triangleCount;
        bool operator==(const EmissiveGeometryKey&) const = default;
    };
    uint32_t BuildIntroEmissiveTriangleTasks(std::vector<PrepareLightsTask>& outTasks,
        std::vector<EmissiveGeometryKey>* outKeys = nullptr) const;
    // Creates/grows the task buffer, bake constant buffer, binding layout and binding set used by the bake.
    void EnsureIntroPrepareLightsResources(uint32_t taskCount);

    void CreateIntroNeighborOffsetsBuffer();
    void EnsureIntroReSTIRDIResources();
    // Recreates the global binding set only when materials are ready (see Sample::RecreateBindingSet).
    void RefreshBindingSetIfNeeded();
    void InitializeIntroReSTIRNeighborOffsets(nvrhi::CommandListHandle commandList);
    void FillIntroReSTIRDIConstants(nvrhi::CommandListHandle commandList);

    donut::math::int2 GetViewportComputeDispatchSize() const;
    void DispatchComputePassState(
        nvrhi::CommandListHandle commandList,
        nvrhi::IComputePipeline* pipeline,
        const SampleMiniConstants& miniConstants) const;
    bool DispatchComputePass(
        nvrhi::CommandListHandle commandList,
        const std::shared_ptr<ComputeShaderVariant>& variant,
        const SampleMiniConstants& miniConstants,
        const char* perfMarkerName = nullptr) const;

    nvrhi::BufferHandle                         m_RTXDILightsBase;   // PolymorphicLightInfo entries
    nvrhi::BufferHandle                         m_RTXDILightsEx;     // PolymorphicLightInfoEx entries
    uint32_t                                    m_RTXDILightCapacity = 0;
    uint32_t                                    m_RTXDIFrameIndex = 0;

    // Offset-based remapping is valid only while each slot still denotes the same light.
    // Reset reuse on layout changes instead of maintaining a per-light mapping table.
    struct LightLayout
    {
        std::vector<const donut::engine::Light*> analyticLights;
        std::vector<const donut::engine::Light*> infiniteLights;
        std::vector<EmissiveGeometryKey> emissiveGeometries;
        uint32_t capacity = 0;
        bool environmentLight = false;
        bool operator==(const LightLayout&) const = default;
    };
    LightLayout                                 m_previousLightLayout;
    bool                                        m_previousLightsReady = false;
    bool                                        m_lightLayoutMatchesPrevious = false;
    bool                                        m_reSTIRHistoryValid = false;
    std::array<uint32_t, 4>                      m_previousReSTIRCandidateCounts = {};

    // Emissive-triangle bake resources (GPU PrepareLights). Triangles are appended into the
    // local light region of the SoA Base/Ex buffers via a dedicated UAV binding set.
    std::shared_ptr<ComputeShaderVariant>       m_introPrepareLightsVariant;
    nvrhi::BindingLayoutHandle                  m_introPrepareLightsBindingLayout;
    nvrhi::BindingSetHandle                     m_introPrepareLightsBindingSet;
    nvrhi::BufferHandle                         m_introPrepareLightsTaskBuffer;
    nvrhi::BufferHandle                         m_introPrepareLightsCB;
    uint32_t                                    m_introPrepareLightsTaskCapacity = 0;
    // Handles captured in the current bake binding set; used to detect when it must be rebuilt.
    nvrhi::IBuffer*                             m_introPrepareLightsBoundBase = nullptr;
    nvrhi::IBuffer*                             m_introPrepareLightsBoundEx = nullptr;
    nvrhi::IBuffer*                             m_introPrepareLightsBoundTasks = nullptr;
    nvrhi::IBuffer*                             m_introPrepareLightsBoundSubInstance = nullptr;
    nvrhi::IBuffer*                             m_introPrepareLightsBoundInstance = nullptr;
    nvrhi::IBuffer*                             m_introPrepareLightsBoundGeometry = nullptr;
    nvrhi::IBuffer*                             m_introPrepareLightsBoundMaterial = nullptr;

    // Direct RTXDI ReSTIR DI (not RtxdiPass)
    std::unique_ptr<rtxdi::ReSTIRDIContext>     m_restirDIContext;
    nvrhi::BufferHandle                         m_lightReservoirBuffer;
    nvrhi::BufferHandle                         m_neighborOffsetsBuffer;
    nvrhi::BufferHandle                         m_introReSTIRConstBuffer;
    bool                                        m_neighborOffsetsInitialized = false;

    // Environment IBL (prefiltered from the environment map / procedural sky)
    nvrhi::TextureHandle                        m_envGGXFilteredCube;   // GGX-prefiltered mip chain (for specular)
    nvrhi::TextureHandle                        m_envIrradianceCube;    // Low-res irradiance cubemap (for diffuse)

    // === RTGI System ===
    std::unique_ptr<RTGIRenderer>               m_rtgiRenderer;

    // === Indirect Path Tracer (MIS mode) ===
    std::unique_ptr<IndirectPathTracer>          m_indirectPathTracer;
    std::shared_ptr<ComputeShaderVariant>       m_directSpecularLightingVariant; // deferred lighting combine reading fully-shaded specular directly from u_SSRResult (MIS indirect + RT reflections)

    // === SHaRC (Spatial Hash Radiance Cache) ===
    std::unique_ptr<SHaRCRenderer>              m_sharcRenderer;

    // === DLSS-RR guide buffer preparation ===
    std::shared_ptr<ComputeShaderVariant>       m_prepareGuideBuffersVariant;
    
};
