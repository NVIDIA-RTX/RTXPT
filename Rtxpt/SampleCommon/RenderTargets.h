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

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>
#include <nvrhi/utils.h>
#include <nvrhi/common/misc.h>
#include <memory>
#include <donut/render/GBuffer.h>

#include "../Shaders/PathTracer/Config.h"

namespace donut::engine
{
    class FramebufferFactory;
}

class RenderTargets// : public donut::render::GBufferRenderTargets
{
    const dm::uint m_SampleCount = 1; // no MSAA supported in this sample
    bool m_UseReverseProjection = false;
    int m_BackbufferCount = -1;
    nvrhi::IDevice* m_device;
public:
    struct OptionalBuffers
    {
        bool NeedsGBuffer              = false;
        bool NeedsDirectLightingBuffer = false;
        bool EnableMotionVectors       = true;  // currently informational; allocation is unconditional
        bool UseReverseProjection      = true;  // selects Depth clear value (0 for reversed, 1 otherwise)
    };

    nvrhi::TextureHandle AccumulatedRadiance;   // used only in non-realtime mode
    nvrhi::TextureHandle LdrColor;              // final, post-tonemapped color
    nvrhi::TextureHandle LdrColorScratch;       // used for ping-ponging post-process stuff vs LdrColor
    nvrhi::TextureHandle OutputColor;           // raw path tracing output goes here (in both realtime and non-realtime modes); this can be input to TAA/DLSS
    nvrhi::TextureHandle ProcessedOutputColor;  // for when post-processing OutputColor (i.e. TAA) (previously ResolvedColor); this is the output of TAA/DLSS in full res, but before tonemapping and without ImGUI
    nvrhi::TextureHandle TemporalFeedback1;     // used by TAA; only allocated while TAA is active, see EnsureTemporalFeedback
    nvrhi::TextureHandle TemporalFeedback2;     // used by TAA
    nvrhi::TextureHandle PreUIColor;            // used DLSS-G

    // note: DLSS-RR also uses ProcessedOutputColor as sl::kBufferTypeScalingOutputColor (-RR output) and OutputColor as sl::kBufferTypeScalingInputColor (-RR input)
    nvrhi::TextureHandle RRDiffuseAlbedo;       // used by DLSS-RR, see: sl::kBufferTypeAlbedo
    nvrhi::TextureHandle RRSpecAlbedo;          // used by DLSS-RR, see: sl::kBufferTypeSpecularAlbedo
    nvrhi::TextureHandle RRNormalsAndRoughness; // used by DLSS-RR, see: sl::kBufferTypeNormals and
    nvrhi::TextureHandle RRSpecMotionVectors;   // used by DLSS-RR, see: sl::kBufferTypeSpecularMotionVectors and sl::DLSSDNormalRoughnessMode::ePacked
    nvrhi::TextureHandle RRTransparencyLayer;   // used by DLSS-RR, see: sl::kBufferTypeTransparencyLayer
    nvrhi::TextureHandle RRDepth;               // DLSS-RR's copy of Depth, so the async prefix can rewrite Depth while DLSS-RR runs
    nvrhi::TextureHandle RRMotionVectors;       // same for ScreenMotionVectors

    nvrhi::TextureHandle Throughput;            // when using PSR we need to remember throughput after perfect speculars with color for RTXDI to know how to do its thing correctly
    nvrhi::TextureHandle Depth;                 // exported by path tracer, used by TAA and others
    nvrhi::TextureHandle ScreenMotionVectors;   // screen space motion vectors, exported by path tracer, used by RTXDI, TAA and others

    nvrhi::TextureHandle DenoiserViewspaceZ;
    nvrhi::TextureHandle DenoiserMotionVectors;
    nvrhi::TextureHandle DenoiserNormalRoughness;

    nvrhi::TextureHandle StableRadiance;                    // radiance that doesn't require denoising; this is technically not needed as a separate buffer, but very useful for debug viz
    nvrhi::TextureHandle DenoisingLayersHeader;
    nvrhi::BufferHandle  DenoisingLayersBuffer;

    nvrhi::BufferHandle  SurfaceDataBuffer;

    nvrhi::BufferHandle  SubSamplingPixelList;              // stream-compacted pixels for the stochastic sub-sampling pattern - layout in Shaders/Misc/SubSampling.h

    nvrhi::TextureHandle DenoiserAvgLayerRadianceHalfRes;

    nvrhi::TextureHandle DenoiserDiffRadianceHitDist;       // input to denoiser
    nvrhi::TextureHandle DenoiserSpecRadianceHitDist;       // input to denoiser
    nvrhi::TextureHandle DenoiserDisocclusionThresholdMix;  // input to denoiser (see IN_DISOCCLUSION_THRESHOLD_MIX)
    
    nvrhi::TextureHandle CombinedHistoryClampRelax;         // all DenoiserDisocclusionThresholdMix combined together - used to tell TAA where to relax disocclusion test to minimize aliasing

    nvrhi::TextureHandle DenoiserOutDiffRadianceHitDist[cDenoisingLayerCount]; // output from denoiser, texture per denoiser instance - search for OUT_DIFF_RADIANCE_HITDIST in NRDDescs.h for more info
    nvrhi::TextureHandle DenoiserOutSpecRadianceHitDist[cDenoisingLayerCount]; // output from denoiser, texture per denoiser instance - search for OUT_SPEC_RADIANCE_HITDIST in NRDDescs.h for more info
    nvrhi::TextureHandle DenoiserOutValidation = nullptr;   // output from denoiser (for validation) - leave nullptr to disable validation

    nvrhi::TextureHandle SecondarySurfacePositionNormal;    // input to restir gi
    nvrhi::TextureHandle SecondarySurfaceRadiance;          // input to restir gi

    // GBuffer
    nvrhi::TextureHandle BaseColor;
    nvrhi::TextureHandle SpecNormal;
    nvrhi::TextureHandle RoughnessMetal;
    nvrhi::TextureHandle MaterialInfo;

    nvrhi::TextureHandle SpecularHitT;                      // input for denoisers to be able to resolve spec motion vectors
    nvrhi::TextureHandle ScratchFloat1;                     // can be used to ping-pong stuff - 32bit float 1

    // === Direct Lighting ===
    nvrhi::TextureHandle DirectLighting;                    // Full-res R11G11B10_FLOAT: direct lighting output

    nvrhi::HeapHandle Heap;

    donut::math::uint2 RenderSize;// size of render targets pre-DLSS
    donut::math::uint2 DisplaySize; // size of render targets post-DLSS

    // Framebuffers are used by the bloom and tone mapping passes
    std::shared_ptr<donut::engine::FramebufferFactory> ProcessedOutputFramebuffer;
    std::shared_ptr<donut::engine::FramebufferFactory> LdrFramebuffer;

    void Init(nvrhi::IDevice* device, donut::math::uint2 renderSize, donut::math::uint2 displaySize, int backbufferCount, OptionalBuffers optionalBuffers);// override;
    [[nodiscard]] bool IsUpdateRequired(donut::math::uint2 renderSize, donut::math::uint2 displaySize, donut::math::uint sampleCount = 1) const;
    void Clear(nvrhi::ICommandList* commandList);

    // TAA history is display-size, so it is only allocated while TAA is active
    void EnsureTemporalFeedback();
    void ReleaseTemporalFeedback();

    // Only allocated while the async compute prefix is enabled
    void EnsureRRDepthAndMotionVectors();

    static uint32_t GetNumMipLevels(uint32_t width, uint32_t height);
};

