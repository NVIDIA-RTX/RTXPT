/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define USE_DENOISING_NRD 1     // <- this define should come from elsewhere

#include "../Shaders/PathTracer/Config.h"

#include "RenderTargets.h"

#include <donut/engine/FramebufferFactory.h>

#include <donut/core/math/math.h>
using namespace donut::math;

#include "../Shaders/PathTracer/DenoisingLayers.hlsli"

#include "../RTXDI/ShaderParameters.h"
#include "../Shaders/Misc/SubSampling.h"
#include <donut/core/log.h>

#include <cmath>
#include <algorithm>
#include <cstdint>

using namespace dm;
using namespace donut;
using namespace donut::math;

void RenderTargets::Init(
        nvrhi::IDevice* device,
        donut::math::uint2 renderSize, 
        donut::math::uint2 displaySize,
        int backbufferCount,
        OptionalBuffers optionalBuffers
    )
{
    m_UseReverseProjection = optionalBuffers.UseReverseProjection;
    m_BackbufferCount = backbufferCount;
    m_device = device;
    RenderSize = renderSize;
    DisplaySize = displaySize;

    nvrhi::TextureDesc desc;
    desc.width = renderSize.x;
    desc.height = renderSize.y;

    desc.isVirtual = false; //device->queryFeatureSupport(nvrhi::Feature::VirtualResources); <- codepath not up to date, needs refactoring

    desc.sampleCount = 1; assert(m_SampleCount == 1);
    desc.dimension =  nvrhi::TextureDimension::Texture2D;
    desc.keepInitialState = true;
    desc.mipLevels = 1;

    desc.format = nvrhi::Format::R32_FLOAT;
    desc.isTypeless = false;
    desc.isUAV = true;
    desc.isRenderTarget = false;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.debugName = "Depth";
    //desc.clearValue = m_UseReverseProjection ? nvrhi::Color(0.f) : nvrhi::Color(1.f);
    //desc.useClearValue = true;
    Depth = device->createTexture(desc);

    desc.isTypeless = false;
    desc.isRenderTarget = false;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.clearValue = nvrhi::Color(0.f);
    desc.isUAV = true;
    desc.format = nvrhi::Format::RGBA16_FLOAT;	// allow for .z component too
    desc.isRenderTarget = true;
    desc.debugName = "ScreenMotionVectors";
    ScreenMotionVectors = device->createTexture(desc);
    desc.isRenderTarget = false;
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.debugName = "DenoiserMotionVectors";
    DenoiserMotionVectors = device->createTexture(desc);

    desc.format = nvrhi::Format::R32_UINT;
    desc.debugName = "Throughput";
    Throughput = device->createTexture(desc);

    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.debugName = "StableRadianceBuffer";
    StableRadiance = device->createTexture(desc);
    
    desc.format = nvrhi::Format::R32_UINT;
    desc.arraySize = 4;
    desc.dimension = nvrhi::TextureDimension::Texture2DArray;
    desc.debugName = "DenoisingLayersHeader";
    DenoisingLayersHeader = device->createTexture(desc);
    desc.dimension = nvrhi::TextureDimension::Texture2D;
    desc.arraySize = 1;

    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.debugName = "DenoiserDiffRadianceHitDist";
    DenoiserDiffRadianceHitDist = device->createTexture(desc);
    desc.debugName = "DenoiserOutDiffRadianceHitDist";
    for (int i = 0; i < cDenoisingLayerCount; ++i)
        DenoiserOutDiffRadianceHitDist[i] = device->createTexture(desc);

    desc.debugName = "DenoiserSpecRadianceHitDist";
    DenoiserSpecRadianceHitDist = device->createTexture(desc);
    desc.debugName = "DenoiserOutSpecRadianceHitDist";
    for (int i = 0; i < cDenoisingLayerCount; ++i)
        DenoiserOutSpecRadianceHitDist[i] = device->createTexture(desc);

    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.clearValue = nvrhi::Color(0.0f, 0.0f, 0.0f, 0.0f);   // avoid the debug layer warnings... not actually cleared except for debug purposes
#if ENABLE_DEBUG_VIZUALISATIONS
    desc.format = nvrhi::Format::RGBA8_UNORM;
    desc.debugName = "DenoiserOutValidation";
    DenoiserOutValidation = device->createTexture(desc);
#endif

    desc.format = nvrhi::Format::R32_FLOAT;
    desc.debugName = "DenoiserViewspaceZ";
    DenoiserViewspaceZ = device->createTexture(desc);

    desc.format = nvrhi::Format::R10G10B10A2_UNORM;
    desc.debugName = "DenoiserNormalRoughness";
    DenoiserNormalRoughness = device->createTexture(desc);

    desc.format = nvrhi::Format::RGBA32_FLOAT;
    desc.debugName = "SecondarySurfacePositionNormal";
    SecondarySurfacePositionNormal = device->createTexture(desc);

    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.debugName = "SecondarySurfaceRadiance";
    SecondarySurfaceRadiance = device->createTexture(desc);

    desc.format = nvrhi::Format::R32_FLOAT;   // currently shaders expect FP32 but could be tweaked to FP16
    desc.debugName = "SpecularHitT";
    SpecularHitT = device->createTexture(desc);

    desc.debugName = "ScratchFloat1";
    ScratchFloat1 = device->createTexture(desc);
    
    desc.isUAV = false;
    desc.isRenderTarget = true;
    desc.useClearValue = false;
    desc.clearValue = nvrhi::Color(1.f);
    desc.sampleCount = m_SampleCount;
    desc.dimension = m_SampleCount > 1 ? nvrhi::TextureDimension::Texture2DMS : nvrhi::TextureDimension::Texture2D;
    desc.keepInitialState = true;

    desc.useClearValue = false;
    desc.clearValue = nvrhi::Color(0.f);
    desc.isTypeless = false;
    desc.isUAV = true;
    desc.isRenderTarget = false;
    desc.format = nvrhi::Format::RGBA32_FLOAT;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.debugName = "AccumulatedRadiance";
    AccumulatedRadiance = device->createTexture(desc);

#if 0
    nvrhi::Format radianceFormat = nvrhi::Format::R11G11B10_FLOAT;
#else
    nvrhi::Format radianceFormat = nvrhi::Format::RGBA16_FLOAT;
#endif

    desc.useClearValue = false;
    desc.format = radianceFormat;  // keep in float for now in case we need 
    desc.debugName = "OutputColor";
    desc.clearValue = nvrhi::Color(1.0f, 1.0f, 0.0f, 0.0f);   // avoid the debug layer warnings... not actually cleared except for debug purposes
    desc.isUAV = true;
    OutputColor = device->createTexture(desc);

    desc.format = nvrhi::Format::R8_UNORM;
    desc.isUAV = true;
    desc.debugName = "DenoiserDisocclusionThresholdMix";
    DenoiserDisocclusionThresholdMix = device->createTexture(desc);
    desc.debugName = "CombinedHistoryClampRelax";
    CombinedHistoryClampRelax = device->createTexture(desc);

    // these are used for DLSS-RR - we could overlap with other buffers to save on RAM but we leave as separate for simplicity
    // see https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md
    desc.format = nvrhi::Format::R11G11B10_FLOAT;       // from docs: "Diffuse component of Reflectance material. Any standard 3-channel format provided at input resolution. The format must be linear, sRGB textures are not supported."
    desc.debugName = "RRDiffuseAlbedo";
    RRDiffuseAlbedo = device->createTexture(desc);
    desc.debugName = "RRSpecAlbedo";
    RRSpecAlbedo = device->createTexture(desc);
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.debugName = "RRNormalsAndRoughness";
    RRNormalsAndRoughness = device->createTexture(desc);
    desc.format = nvrhi::Format::RG16_FLOAT;
    desc.debugName = "RRSpecMotionVectors";
    RRSpecMotionVectors = device->createTexture(desc);
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.debugName = "RRTransparencyLayer";
    // RRTransparencyLayer = device->createTexture(desc); // not currently used

    // GBuffer targets (used by samples that run a deferred-style pass)
    if (optionalBuffers.NeedsGBuffer)
    {
        desc.debugName = "GBufferBaseColor"; // Can reuse RRDiffuseAlbedo?
        desc.format = nvrhi::Format::R11G11B10_FLOAT;
        BaseColor = device->createTexture(desc);
        desc.debugName = "GBufferSpecNormal";
        desc.format = nvrhi::Format::R32_UINT;
        SpecNormal = device->createTexture(desc);
        desc.debugName = "GBufferRoughnessMetal";
        desc.format = nvrhi::Format::RG16_FLOAT;
        RoughnessMetal = device->createTexture(desc);
        desc.debugName = "GBufferMaterialInfo";
        desc.format = nvrhi::Format::R32_UINT;
        MaterialInfo = device->createTexture(desc);
    }

    // !!! NOTE !!! setting desc.width/desc.height to half render size (was render size!)
    desc.width = (renderSize.x + 1) / 2;
    desc.height = (renderSize.y + 1) / 2;
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.debugName = "DenoiserAvgRadianceHalfRes";
    DenoiserAvgLayerRadianceHalfRes = device->createTexture(desc);

    // !!! NOTE !!! setting desc.width/desc.height to display size (was render size!)
    desc.width = displaySize.x;
    desc.height = displaySize.y;

    desc.format = radianceFormat;
    // Restore the render-target group: ProcessedOutputColor and LdrColor below are bound as framebuffer render
    // targets (see ProcessedOutputFramebuffer / LdrFramebuffer at the bottom of this function), and PreUIColor
    // has no UAV flag at all - an UnorderedAccess resting layout is illegal for it. Note this one desc is mutated
    // through every texture below, so a flag set here applies to all of them until changed again.
    desc.isRenderTarget = true;
    desc.initialState = nvrhi::ResourceStates::RenderTarget;
    desc.debugName = "ProcessedOutputColor";
    ProcessedOutputColor = device->createTexture(desc);

    desc.format = nvrhi::Format::SRGBA8_UNORM;
    desc.isUAV = true;
    desc.isTypeless = true;
    desc.debugName = "LdrColor";
    LdrColor = device->createTexture(desc);
    desc.debugName = "LdrColorScratch";
    LdrColorScratch = device->createTexture(desc);
    desc.isTypeless = false;

    desc.isUAV = false;

    desc.debugName = "PreUIColor";
    PreUIColor = device->createTexture(desc);

#if 0
    if (desc.isVirtual)
    {
        uint64_t heapSize = 0;
        nvrhi::ITexture* const textures[] = {
            //HdrColor,
            //ResolvedColor,
            //TemporalFeedback1,
            //TemporalFeedback2,
            LdrColor,
            OutputColor,
            PreUIColor,
            //AmbientOcclusion
        };

        for (auto texture : textures)
        {
            nvrhi::MemoryRequirements memReq = device->getTextureMemoryRequirements(texture);
            heapSize = nvrhi::align(heapSize, memReq.alignment);
            heapSize += memReq.size;
        }

        nvrhi::HeapDesc heapDesc;
        heapDesc.type = nvrhi::HeapType::DeviceLocal;
        heapDesc.capacity = heapSize;
        heapDesc.debugName = "RenderTargetHeap";

        Heap = device->createHeap(heapDesc);

        uint64_t offset = 0;
        for (auto texture : textures)
        {
            nvrhi::MemoryRequirements memReq = device->getTextureMemoryRequirements(texture);
            offset = nvrhi::align(offset, memReq.alignment);

            device->bindTextureMemory(texture, Heap, offset);

            offset += memReq.size;
        }
    }
#endif

    // === Direct Lighting ===
    if (optionalBuffers.NeedsDirectLightingBuffer)
    {
        nvrhi::TextureDesc dlDesc;
        dlDesc.width = renderSize.x;
        dlDesc.height = renderSize.y;
        dlDesc.format = nvrhi::Format::R11G11B10_FLOAT;
        dlDesc.dimension = nvrhi::TextureDimension::Texture2D;
        dlDesc.debugName = "DirectLighting";
        dlDesc.isUAV = true;
        dlDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        dlDesc.keepInitialState = true;
        DirectLighting = device->createTexture(dlDesc);
    }

    ProcessedOutputFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
    ProcessedOutputFramebuffer->RenderTargets = { ProcessedOutputColor };

    LdrFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
    LdrFramebuffer->RenderTargets = { LdrColor };

    { // Denoising layersS
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.isVertexBuffer = false;
        bufferDesc.isConstantBuffer = false;
        bufferDesc.isVolatile = false;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
        bufferDesc.keepInitialState = true;
        bufferDesc.initialState = nvrhi::ResourceStates::Common;

        bufferDesc.structStride = sizeof(DenoisingLayer);
        bufferDesc.byteSize = sizeof(DenoisingLayer) * GenericTSComputeStorageElementCount(RenderSize.x, RenderSize.y, cDenoisingLayerCount);
        bufferDesc.debugName = "DenoisingLayersBuffer";
        DenoisingLayersBuffer = device->createBuffer(bufferDesc);
    }

    {
        nvrhi::BufferDesc surfaceBufferDesc;
        surfaceBufferDesc.byteSize = sizeof(PackedPathTracerSurfaceData) * 2 * renderSize.x * renderSize.y; // *2 is for history!
        surfaceBufferDesc.byteSize = sizeof(PackedPathTracerSurfaceData) * GenericTSComputeStorageElementCount(RenderSize.x, RenderSize.y, 2); // *2 is for history!
        surfaceBufferDesc.structStride = sizeof(PackedPathTracerSurfaceData);
        surfaceBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        surfaceBufferDesc.keepInitialState = true;
        surfaceBufferDesc.debugName = "SurfaceData(GBuffer)";
        surfaceBufferDesc.canHaveUAVs = true;
        SurfaceDataBuffer = device->createBuffer(surfaceBufferDesc);
    }

    {
        nvrhi::BufferDesc pixelListDesc;
        pixelListDesc.byteSize = RTXPT_SUB_SAMPLING_LIST_HEADER_BYTES + sizeof(uint32_t) * renderSize.x * renderSize.y;    // worst case: every pixel selected
        pixelListDesc.canHaveUAVs = true;
        pixelListDesc.canHaveRawViews = true;
        pixelListDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        pixelListDesc.keepInitialState = true;
        pixelListDesc.debugName = "SubSamplingPixelList";
        SubSamplingPixelList = device->createBuffer(pixelListDesc);
    }
}

[[nodiscard]] bool RenderTargets::IsUpdateRequired(donut::math::uint2 renderSize, donut::math::uint2 displaySize, donut::math::uint sampleCount) const
{
    if (any(RenderSize != renderSize) || any(DisplaySize != displaySize) || m_SampleCount != sampleCount) return true;
    return false;
}

void RenderTargets::EnsureTemporalFeedback()
{
    if (TemporalFeedback1 != nullptr)
        return;

    // Same desc these had when Init created them unconditionally.
    nvrhi::TextureDesc desc;
    desc.width = DisplaySize.x;
    desc.height = DisplaySize.y;
    desc.format = nvrhi::Format::RGBA16_SNORM;
    desc.isUAV = true;
    desc.isRenderTarget = true;
    desc.initialState = nvrhi::ResourceStates::RenderTarget;
    desc.keepInitialState = true;
    desc.debugName = "TemporalFeedback1";
    TemporalFeedback1 = m_device->createTexture(desc);
    desc.debugName = "TemporalFeedback2";
    TemporalFeedback2 = m_device->createTexture(desc);
}

void RenderTargets::ReleaseTemporalFeedback()
{
    TemporalFeedback1 = nullptr;
    TemporalFeedback2 = nullptr;
}

void RenderTargets::EnsureRRDepthAndMotionVectors()
{
    if (RRDepth != nullptr)
        return;

    // Same descs as the sources; resting in CopyDest means no barrier for the per-frame copy into them
    nvrhi::TextureDesc desc = Depth->getDesc();
    desc.initialState = nvrhi::ResourceStates::CopyDest;
    desc.debugName = "RRDepth";
    RRDepth = m_device->createTexture(desc);

    desc = ScreenMotionVectors->getDesc();
    desc.initialState = nvrhi::ResourceStates::CopyDest;
    desc.debugName = "RRMotionVectors";
    RRMotionVectors = m_device->createTexture(desc);
}

void RenderTargets::Clear(nvrhi::ICommandList* commandList) 
{
    const nvrhi::FormatInfo& depthFormatInfo = nvrhi::getFormatInfo(Depth->getDesc().format);

    float depthClearValue = m_UseReverseProjection ? 0.f : 1.f;
    commandList->clearTextureFloat(Depth, nvrhi::AllSubresources, nvrhi::Color(depthClearValue));

    commandList->clearTextureFloat(CombinedHistoryClampRelax, nvrhi::AllSubresources, nvrhi::Color(0));
}

uint32_t RenderTargets::GetNumMipLevels(uint32_t width, uint32_t height)
{
    uint32_t maxDim = std::max(width, height);
    uint32_t levelsNum = 1 + (uint32_t)std::log2((float)maxDim);

    return levelsNum;
}