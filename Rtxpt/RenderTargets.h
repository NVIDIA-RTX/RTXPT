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

#include "PathTracer/Config.h"

namespace donut::engine
{
    class FramebufferFactory;
}

class RenderTargets// : public donut::render::GBufferRenderTargets
{
    const dm::uint m_SampleCount = 1; // no MSAA supported in this sample
    bool m_UseReverseProjection = false;
    int m_BackbufferCount = 3;
    nvrhi::IDevice* m_device;
public:
    nvrhi::TextureHandle AccumulatedRadiance;   // used only in non-realtime mode
    nvrhi::TextureHandle LdrColor;              // final, post-tonemapped color
    nvrhi::TextureHandle OutputColor;           // raw path tracing output goes here (in both realtime and non-realtime modes); this can be input to TAA/DLSS
    nvrhi::TextureHandle ProcessedOutputColor;  // for when post-processing OutputColor (i.e. TAA) (previously ResolvedColor); this is the output of TAA/DLSS in full res, but before tonemapping and without ImGUI
    nvrhi::TextureHandle TemporalFeedback1;     // used by TAA
    nvrhi::TextureHandle TemporalFeedback2;     // used by TAA
    nvrhi::TextureHandle PreUIColor;            // used DLSS-G

    // note: DLSS-RR also uses ProcessedOutputColor as sl::kBufferTypeScalingOutputColor (-RR output) and OutputColor as sl::kBufferTypeScalingInputColor (-RR input)
    nvrhi::TextureHandle RRDiffuseAlbedo;       // used by DLSS-RR, see: sl::kBufferTypeAlbedo
    nvrhi::TextureHandle RRSpecAlbedo;          // used by DLSS-RR, see: sl::kBufferTypeSpecularAlbedo
    nvrhi::TextureHandle RRNormalsAndRoughness; // used by DLSS-RR, see: sl::kBufferTypeNormals and 
    nvrhi::TextureHandle RRSpecMotionVectors;   // used by DLSS-RR, see: sl::kBufferTypeSpecularMotionVectors and sl::DLSSDNormalRoughnessMode::ePacked

    nvrhi::TextureHandle Throughput;            // when using PSR we need to remember throughput after perfect speculars with color for RTXDI to know how to do its thing correctly
    nvrhi::TextureHandle Depth;                 // exported by path tracer, used by TAA and others
    nvrhi::TextureHandle ScreenMotionVectors;   // screen space motion vectors, exported by path tracer, used by RTXDI, TAA and others

    nvrhi::TextureHandle DenoiserViewspaceZ;
    nvrhi::TextureHandle DenoiserMotionVectors;
    nvrhi::TextureHandle DenoiserNormalRoughness;

    nvrhi::TextureHandle StableRadiance;                    // radiance that doesn't require denoising; this is technically not needed as a separate buffer, but very useful for debug viz
    nvrhi::TextureHandle StablePlanesHeader;
    nvrhi::BufferHandle  StablePlanesBuffer;

    nvrhi::BufferHandle  SurfaceDataBuffer;

    nvrhi::TextureHandle DenoiserDiffRadianceHitDist;       // input to denoiser
    nvrhi::TextureHandle DenoiserSpecRadianceHitDist;       // input to denoiser
    nvrhi::TextureHandle DenoiserDisocclusionThresholdMix;  // input to denoiser (see IN_DISOCCLUSION_THRESHOLD_MIX)
    
    nvrhi::TextureHandle CombinedHistoryClampRelax;         // all DenoiserDisocclusionThresholdMix combined together - used to tell TAA where to relax disocclusion test to minimize aliasing

    nvrhi::TextureHandle DenoiserOutDiffRadianceHitDist[cStablePlaneCount]; // output from denoiser, texture per denoiser instance - search for OUT_DIFF_RADIANCE_HITDIST in NRDDescs.h for more info
    nvrhi::TextureHandle DenoiserOutSpecRadianceHitDist[cStablePlaneCount]; // output from denoiser, texture per denoiser instance - search for OUT_SPEC_RADIANCE_HITDIST in NRDDescs.h for more info
    nvrhi::TextureHandle DenoiserOutValidation = nullptr;   // output from denoiser (for validation) - leave nullptr to disable validation

    nvrhi::TextureHandle SecondarySurfacePositionNormal;    // input to restir gi
    nvrhi::TextureHandle SecondarySurfaceRadiance;          // input to restir gi

    nvrhi::HeapHandle Heap;

    donut::math::uint2 m_renderSize;// size of render targets pre-DLSS
    donut::math::uint2 m_displaySize; // size of render targets post-DLSS

    // in theory we shouldn't need this but it seems to still be used by tonemapper
    std::shared_ptr<donut::engine::FramebufferFactory> LdrFramebuffer;

    void Init(nvrhi::IDevice* device, donut::math::uint2 renderSize, donut::math::uint2 displaySize, bool enableMotionVectors, bool useReverseProjection, int backbufferCount);// override;
    [[nodiscard]] bool IsUpdateRequired(donut::math::uint2 renderSize, donut::math::uint2 displaySize, donut::math::uint sampleCount = 1) const;
    void Clear(nvrhi::ICommandList* commandList); // override;
};

