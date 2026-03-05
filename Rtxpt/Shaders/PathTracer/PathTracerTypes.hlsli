/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_TYPES_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_TYPES_HLSLI__

#include "Config.h"    

#include "Utils/Math/Ray.hlsli"
#include "Rendering/Materials/TexLODHelpers.hlsli"
#include "Scene/Material/TextureSampler.hlsli"
#include "Scene/ShadingData.hlsli"
#include "Scene/Material/ShadingUtils.hlsli"
#include "Rendering/Materials/LobeType.hlsli"
#include "Rendering/Materials/IBSDF.hlsli"
#include "Rendering/Materials/StandardBSDF.hlsli"
#include "PathState.hlsli"
#include "PathTracerDebug.hlsli"
#include "PathTracerHelpers.hlsli"
#include "PathPayload.hlsli"
#include "StablePlanes.hlsli"

#if ACTIVE_LOD_TEXTURE_SAMPLER == LOD_TEXTURE_SAMPLER_EXPLICIT
    #define ActiveTextureSampler ExplicitLodTextureSampler
#elif ACTIVE_LOD_TEXTURE_SAMPLER == LOD_TEXTURE_SAMPLER_RAY_CONES
    #define ActiveTextureSampler ExplicitRayConesLodTextureSampler
#else
    #error please specify texture LOD sampler
#endif

namespace PathTracer
{
    /** Holds path tracer shader working data for state, settings, debugging, denoising and etc. Everything that is shared across (DispatchRays/Compute) draw call, between all pixels, 
        but also some pixel-specific stuff (like pixelPos heh). It's what a PathTracer instance would store if it could be an OOP object.
    */
    struct WorkingContext
    {
        RWTexture2D<float4>     OutputColor;
        PathTracerConstants     PtConsts;
        DebugContext            Debug;
        StablePlanesContext     StablePlanes;
    };

    /** All surface data returned by the Bridge::loadSurface
    */
    struct SurfaceData
    {
        ShadingData     shadingData;
        ActiveBSDF      bsdf;
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES  // otherwise motion vectors not needed
        float3          prevPosW;                           // <-- consider storing delta instead of prevPosW and then fp16 might be enough
#endif
        lpfloat         interiorIoR;    // a.k.a. material IoR
#if !RTXPT_USE_APPROXIMATE_MIS
#if !defined(RTXPT_MATERIAL_IS_EMISSIVE) || RTXPT_MATERIAL_IS_EMISSIVE
        uint            neeTriangleLightIndex;  // 0xFFFFFFFF if none
#endif
#endif
#if !defined(RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY) || RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY
        uint            neeAnalyticLightIndex;  // 0xFFFFFFFF if none
#endif
        static SurfaceData make( ShadingData shadingData, ActiveBSDF bsdf, 
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // otherwise motion vectors not needed
                                float3 prevPosW, 
#endif
                                lpfloat interiorIoR, uint neeTriangleLightIndex, uint neeAnalyticLightIndex )
        { 
            SurfaceData ret; 
            ret.shadingData             = shadingData; 
            ret.bsdf                    = bsdf; 
#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES // otherwise motion vectors not needed
            ret.prevPosW                = prevPosW; 
#endif
            ret.interiorIoR             = interiorIoR; 
#if !RTXPT_USE_APPROXIMATE_MIS
#if !defined(RTXPT_MATERIAL_IS_EMISSIVE) || RTXPT_MATERIAL_IS_EMISSIVE
            ret.neeTriangleLightIndex   = neeTriangleLightIndex;
#endif
#endif
#if !defined(RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY) || RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY
            ret.neeAnalyticLightIndex   = neeAnalyticLightIndex;
#endif
            return ret; 
        }
    };

    // Info used for figuring out MIS from the path's (BSDF) side
    struct NEEBSDFMISInfo
    {
        bool LightSamplingEnabled;      // light sampling disabled, MIS for BSDF side is 1
#if PT_USE_RESTIR_DI
        bool SkipEmissiveBRDF;          // Ignore next bounce reflective (but not transmissive) radiance because ReSTIR-DI or similar collected (or will collect) it
#endif
        bool LightSamplingIsSSC;        // if SSC, light sampler uses screen space tiles, otherwise world space
        uint CandidateSamples;          // for WRS, consisting of Local and Global that can be obtained using LightSampler::GetCandidateSampleCounts
        uint FullSamples;               // actually integrated samples

        // Initialize to empty (NEE disabled or primary bounce or etc.)
        static NEEBSDFMISInfo empty() 
        { 
            NEEBSDFMISInfo ret;
            ret.LightSamplingEnabled     = false;
#if PT_USE_RESTIR_DI
            ret.SkipEmissiveBRDF         = false;
#endif
            ret.LightSamplingIsSSC       = false;
            ret.CandidateSamples         = 0;
            ret.FullSamples              = 0;
            return ret;
        }

        static NEEBSDFMISInfo Unpack16bit( uint packed ) 
        { 
            NEEBSDFMISInfo ret;
            ret.LightSamplingEnabled     = (packed & (1 << 15)) != 0;
#if PT_USE_RESTIR_DI
            ret.SkipEmissiveBRDF         = (packed & (1 << 14)) != 0;
#endif
            ret.LightSamplingIsSSC      = (packed & (1 << 13)) != 0;
            ret.CandidateSamples        = (packed >> 6) & 0x3F;        // ensured to fit with RTXPT_LIGHTING_MAX_SAMPLE_COUNT
            ret.FullSamples             = (packed) & 0x3F;             // ensured to fit with RTXPT_LIGHTING_MAX_SAMPLE_COUNT
            return ret;
        }

        uint    Pack16bit()
        {
            uint packed = 0;
            packed |= ((LightSamplingEnabled?1:0)       << 15);
#if PT_USE_RESTIR_DI
            packed |= ((SkipEmissiveBRDF?1:0)           << 14);
#endif
            packed |= ((LightSamplingIsSSC?1:0)    << 13);
            packed |= (CandidateSamples & 0x3F)        << 6;           // ensured to fit with RTXPT_LIGHTING_MAX_SAMPLE_COUNT
            packed |= (FullSamples      & 0x3F);                       // ensured to fit with RTXPT_LIGHTING_MAX_SAMPLE_COUNT
            return packed;
        }

        static const uint SampleCountLimit()        { return (1 << 6)-1; }  // 63 is max we can pack in 6 bits

        static bool equals( const NEEBSDFMISInfo a, const NEEBSDFMISInfo b )
        {
            return     (a.LightSamplingEnabled  == b.LightSamplingEnabled    )
#if PT_USE_RESTIR_DI
                    && (a.SkipEmissiveBRDF      == b.SkipEmissiveBRDF)
#endif
                    && (a.LightSamplingIsSSC    == b.LightSamplingIsSSC )
                    && (a.CandidateSamples      == b.CandidateSamples        )
                    && (a.FullSamples           == b.FullSamples            );
        }

#if PT_USE_RESTIR_DI
        bool GetSkipEmissiveBRDF()          { return SkipEmissiveBRDF; }
#else
        bool GetSkipEmissiveBRDF()          { return false; }
#endif


    };

#define RTXPT_NEE_RESULT_MANUAL_PACK 1

    // Output part of the interface to the path tracer - this will likely change over time.
    struct NEEResult
    {
#if RTXPT_NEE_RESULT_MANUAL_PACK
        uint2       RadianceAndSpecAvgPkg;
#else
        float4      RadianceAndSpecAvg;             // note: these are also multiplied by path.thp so far
#endif

        NEEBSDFMISInfo BSDFMISInfo;
        
        // initialize to empty
        static NEEResult empty() 
        { 
            NEEResult ret;
#if RTXPT_NEE_RESULT_MANUAL_PACK
            ret.RadianceAndSpecAvgPkg = Fp32ToFp16( float4(0,0,0,0) );
#else
            ret.RadianceAndSpecAvg = float4(0,0,0,0);
#endif
            ret.BSDFMISInfo = NEEBSDFMISInfo::empty();
            return ret; 
        }

        void        AccumulateRadiance( const float3 radiance, const float specAvg )
        {
#if RTXPT_NEE_RESULT_MANUAL_PACK
            RadianceAndSpecAvgPkg = Fp32ToFp16( Fp16ToFp32(RadianceAndSpecAvgPkg) + float4( radiance, specAvg ) );
#else
            RadianceAndSpecAvg += float4( radiance, specAvg );
#endif
        }

#if RTXPT_NEE_RESULT_MANUAL_PACK
        float4      GetRadianceAndSpecAvg() { return Fp16ToFp32(RadianceAndSpecAvgPkg); }
#else
        float4      GetRadianceAndSpecAvg() { return RadianceAndSpecAvg; }
#endif
    };
    
    struct VisibilityPayload
    {
        uint missed;
        static VisibilityPayload make( ) 
        { 
            VisibilityPayload ret; 
            ret.missed = 0; 
            return ret; 
        }
    };
}

#endif // __PATH_TRACER_TYPES_HLSLI__