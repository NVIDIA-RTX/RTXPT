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

#include "PathTracer/Config.h"

#include "RenderTargets.h"

#include <donut/engine/FramebufferFactory.h>

#include <donut/core/math/math.h>
using namespace donut::math;
#include "PathTracer/PathTracerShared.h"

#include "PathTracer/StablePlanes.hlsli"

#include "RTXDI/ShaderParameters.h"
#include <donut/core/log.h>

using namespace dm;
using namespace donut;
using namespace donut::math;

void RenderTargets::Init(
        nvrhi::IDevice* device,
        donut::math::uint2 renderSize, 
        donut::math::uint2 displaySize,
        bool enableMotionVectors,
        bool useReverseProjection,
        int backbufferCount)
{
    m_UseReverseProjection = useReverseProjection;
    m_BackbufferCount = backbufferCount;
    m_device = device;
    m_renderSize = renderSize;
    m_displaySize = displaySize;

    nvrhi::TextureDesc desc;
    desc.width = renderSize.x;
    desc.height = renderSize.y;

    desc.isVirtual = false; //device->queryFeatureSupport(nvrhi::Feature::VirtualResources); <- codepath not up to date, needs refactoring

    desc.initialState = nvrhi::ResourceStates::RenderTarget;
    desc.isRenderTarget = true;
    desc.useClearValue = true;
    desc.clearValue = nvrhi::Color(0.f);
    desc.sampleCount = 1; assert(m_SampleCount == 1);
    desc.dimension =  nvrhi::TextureDimension::Texture2D;
    desc.keepInitialState = true;
    desc.isTypeless = false;
    desc.isUAV = false;
    desc.mipLevels = 1;

    desc.format = nvrhi::Format::R32_FLOAT;
    desc.isTypeless = false;
    desc.isUAV = true;
    desc.isRenderTarget = true;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.debugName = "Depth";
    desc.clearValue = useReverseProjection ? nvrhi::Color(0.f) : nvrhi::Color(1.f);
    desc.useClearValue = true;
    Depth = device->createTexture(desc);

    desc.isTypeless = false;
    desc.isRenderTarget = true;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.clearValue = nvrhi::Color(0.f);
    desc.isUAV = true;
    desc.format = nvrhi::Format::RGBA16_FLOAT;	// allow for .z component too
    desc.debugName = "ScreenMotionVectors";
    ScreenMotionVectors = device->createTexture(desc);
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
    desc.debugName = "StablePlanesHeader";
    StablePlanesHeader = device->createTexture(desc);
    desc.dimension = nvrhi::TextureDimension::Texture2D;
    desc.arraySize = 1;

    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.debugName = "DenoiserDiffRadianceHitDist";
    DenoiserDiffRadianceHitDist = device->createTexture(desc);
    desc.debugName = "DenoiserOutDiffRadianceHitDist";
    for (int i = 0; i < cStablePlaneCount; ++i)
        DenoiserOutDiffRadianceHitDist[i] = device->createTexture(desc);

    desc.debugName = "DenoiserSpecRadianceHitDist";
    DenoiserSpecRadianceHitDist = device->createTexture(desc);
    desc.debugName = "DenoiserOutSpecRadianceHitDist";
    for (int i = 0; i < cStablePlaneCount; ++i)
        DenoiserOutSpecRadianceHitDist[i] = device->createTexture(desc);

    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.clearValue = nvrhi::Color(0.0f, 0.0f, 0.0f, 0.0f);   // avoid the debug layer warnings... not actually cleared except for debug purposes
#if ENABLE_DEBUG_VIZUALISATION
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
    
    desc.isUAV = false;
    desc.isRenderTarget = true;
    desc.useClearValue = true;
    desc.clearValue = nvrhi::Color(1.f);
    desc.sampleCount = m_SampleCount;
    desc.dimension = m_SampleCount > 1 ? nvrhi::TextureDimension::Texture2DMS : nvrhi::TextureDimension::Texture2D;
    desc.keepInitialState = true;

    desc.useClearValue = false;
    desc.clearValue = nvrhi::Color(0.f);
    desc.isTypeless = false;
    desc.isUAV = true;
    desc.isRenderTarget = true;
    desc.format = nvrhi::Format::RGBA32_FLOAT;
    desc.initialState = nvrhi::ResourceStates::RenderTarget;
    desc.debugName = "AccumulatedRadiance";
    AccumulatedRadiance = device->createTexture(desc);

    desc.useClearValue = true;
    desc.format = nvrhi::Format::RGBA16_FLOAT;  // keep in float for now in case we need 
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

    // !!! NOTE !!! setting desc.width/desc.height to display size (was render size!)
    desc.width = displaySize.x;
    desc.height = displaySize.y;

    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.debugName = "ProcessedOutputColor";
    ProcessedOutputColor = device->createTexture(desc);
    desc.format = nvrhi::Format::RGBA16_SNORM;
    desc.debugName = "TemporalFeedback1";
    TemporalFeedback1 = device->createTexture(desc);
    desc.debugName = "TemporalFeedback2";
    TemporalFeedback2 = device->createTexture(desc);

    desc.format = nvrhi::Format::SRGBA8_UNORM;
    desc.isUAV = false;
    desc.debugName = "LdrColor";
    LdrColor = device->createTexture(desc);

    desc.debugName = "PreUIColor";
    PreUIColor = device->createTexture(desc);

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

    LdrFramebuffer = std::make_shared<donut::engine::FramebufferFactory>(device);
    LdrFramebuffer->RenderTargets = { LdrColor };

    { // Stable planes
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.isVertexBuffer = false;
        bufferDesc.isConstantBuffer = false;
        bufferDesc.isVolatile = false;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
        bufferDesc.keepInitialState = true;
        bufferDesc.initialState = nvrhi::ResourceStates::Common;

        bufferDesc.structStride = sizeof(StablePlane);
        bufferDesc.byteSize = sizeof(StablePlane) * GenericTSComputeStorageElementCount(m_renderSize.x, m_renderSize.y, cStablePlaneCount);
        bufferDesc.debugName = "StablePlanesBuffer";
        StablePlanesBuffer = device->createBuffer(bufferDesc);
    }

    {
        nvrhi::BufferDesc surfaceBufferDesc;
        surfaceBufferDesc.byteSize = sizeof(PackedPathTracerSurfaceData) * 2 * renderSize.x * renderSize.y; // *2 is for history!
        surfaceBufferDesc.byteSize = sizeof(PackedPathTracerSurfaceData) * GenericTSComputeStorageElementCount(m_renderSize.x, m_renderSize.y, 2); // *2 is for history!
        surfaceBufferDesc.structStride = sizeof(PackedPathTracerSurfaceData);
        surfaceBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        surfaceBufferDesc.keepInitialState = true;
        surfaceBufferDesc.debugName = "SurfaceData(GBuffer)";
        surfaceBufferDesc.canHaveUAVs = true;
        SurfaceDataBuffer = device->createBuffer(surfaceBufferDesc);
    }
}

[[nodiscard]] bool RenderTargets::IsUpdateRequired(donut::math::uint2 renderSize, donut::math::uint2 displaySize, donut::math::uint sampleCount) const
{
    if (any(m_renderSize != renderSize) || any(m_displaySize != displaySize) || m_SampleCount != sampleCount) return true;
    return false;
}

void RenderTargets::Clear(nvrhi::ICommandList* commandList) 
{
    const nvrhi::FormatInfo& depthFormatInfo = nvrhi::getFormatInfo(Depth->getDesc().format);

    float depthClearValue = m_UseReverseProjection ? 0.f : 1.f;
    commandList->clearTextureFloat(Depth, nvrhi::AllSubresources, nvrhi::Color(depthClearValue));

    commandList->clearTextureFloat(CombinedHistoryClampRelax, nvrhi::AllSubresources, nvrhi::Color(0));
}
