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
        uint                    PixelID;

        uint2                   GetPixelPos() { return PathIDToPixel(PixelID); }
    };

    /** All surface data returned by the Bridge::loadSurface
    */
    struct SurfaceData
    {
        ShadingData     shadingData;
        ActiveBSDF      bsdf;
        float3          prevPosW;
        lpfloat         interiorIoR;    // a.k.a. material IoR
        uint            neeTriangleLightIndex;  // 0xFFFFFFFF if none
        uint            neeAnalyticLightIndex;  // 0xFFFFFFFF if none
        static SurfaceData make( /*VertexData vd, */ShadingData shadingData, ActiveBSDF bsdf, float3 prevPosW, lpfloat interiorIoR, uint neeTriangleLightIndex, uint neeAnalyticLightIndex )
        { 
            SurfaceData ret; 
            ret.shadingData             = shadingData; 
            ret.bsdf                    = bsdf; 
            ret.prevPosW                = prevPosW; 
            ret.interiorIoR             = interiorIoR; 
            ret.neeTriangleLightIndex   = neeTriangleLightIndex;
            ret.neeAnalyticLightIndex   = neeAnalyticLightIndex;
            return ret; 
        }
    };

    /** Describes a light sample, mainly for use in NEE.
        It is considered in the context of a shaded surface point, from which Distance and Direction to light sample are computed.
        In case of emissive triangle light source, it is advisable to compute anti-self-intersection offset before computing
        distance and direction, even though distance shortening is needed anyways for shadow rays due to precision issues.
        Use ComputeVisibilityRay to correctly compute surface offset.
    */
    struct PathLightSample
    {
        float3  Li;                     ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        //float   Pdf;                    ///< Pdf with respect to solid angle at the shading point with selected light (selectionPDF*solidAnglePdf).
        float   Distance;               ///< Ray distance for visibility evaluation (NOT shortened or offset to avoid self-intersection). Ray starts at shading surface.
        float3  Direction;              ///< Ray direction (normalized). Ray starts at shading surface.
        uint    LightIndex;             ///< Identifier of the source light (index in the light list), 0xFFFFFFFF if not available.
        float   SelectionPdf;           ///< Pdf of just the source light (LightIndex) selection; In contrast to 'PathLightSample::Pdf' which is a 'selectionPDF * solidAnglePdf'.
        float   SolidAnglePdf;
        bool    LightSampleableByBSDF;  ///< Required for MIS vs BSDF; typically emissive and environment samples can be "seen" by BSDF, while analytic Sphere/Point/Spotlight/etc. are virtual, with non-scene representation
        
        // Computes shading surface visibility ray starting position with an offset to avoid self intersection at source, and a
        // shortening offset to avoid self-intersection at the light source end. 
        // Optimal selfIntersectionShorteningK default found empirically.
        Ray ComputeVisibilityRay(const ShadingData shadingData, const float selfIntersectionShorteningK = 0.9985)
        {
            float3 surfaceShadingNormal = shadingData.N;

            // We must use **shading** normal to correctly figure out whether we're solving for BRDF or BTDF lobe (whether we want to cast the ray above or under the triangle).
            float faceSide = dot(surfaceShadingNormal, Direction) >= 0 ? 1 : -1;

            float3 surfaceFaceNormal = shadingData.faceNCorrected * faceSide;
            float3 surfaceWorldPos = ComputeRayOrigin(shadingData.posW, surfaceFaceNormal);
            return Ray::make(surfaceWorldPos, Direction, 0.0, Distance*selfIntersectionShorteningK); 
        }

        static PathLightSample make() 
        { 
            PathLightSample ret; 
            ret.Li = float3(0,0,0); 
            // ret.Pdf = 0; 
            ret.Distance = 0; 
            ret.Direction = float3(0,0,0); 
            ret.LightIndex = 0xFFFFFFFF;
            ret.LightSampleableByBSDF = false;
            ret.SolidAnglePdf = 0;
            ret.SelectionPdf = 0;
            return ret; 
        }

        bool Valid()    
        { 
            return any(Li > 0); 
        }
    };

    // Info used for figuring out MIS from the path's (BSDF) side
    struct NEEBSDFMISInfo
    {
        bool LightSamplingEnabled;      // light sampling disabled, MIS for BSDF side is 1
#if PT_USE_RESTIR_DI
        bool SkipEmissiveBRDF;          // Ignore next bounce reflective (but not transmissive) radiance because ReSTIR-DI or similar collected (or will collect) it
#endif
        bool LightSamplingIsIndirect;   // using indirect part of the LightSampler domain; otherwise using direct
        uint LocalNEESamples;          // 
        uint TotalSamples;              //

        // Initialize to empty (NEE disabled or primary bounce or etc.)
        static NEEBSDFMISInfo empty() 
        { 
            NEEBSDFMISInfo ret;
            ret.LightSamplingEnabled     = false;
#if PT_USE_RESTIR_DI
            ret.SkipEmissiveBRDF         = false;
#endif
            ret.LightSamplingIsIndirect  = false;
            ret.LocalNEESamples         = 0;
            ret.TotalSamples             = 0;
            return ret;
        }

        static NEEBSDFMISInfo Unpack16bit( uint packed ) 
        { 
            NEEBSDFMISInfo ret;
            ret.LightSamplingEnabled     = (packed & (1 << 15)) != 0;
#if PT_USE_RESTIR_DI
            ret.SkipEmissiveBRDF         = (packed & (1 << 14)) != 0;
#endif
            ret.LightSamplingIsIndirect  = (packed & (1 << 13)) != 0;
            ret.LocalNEESamples         = (packed >> 6) & 0x3F;
            ret.TotalSamples             = (packed) & 0x3F;
            return ret;
        }

        uint    Pack16bit()
        {
            uint packed = 0;
            packed |= ((LightSamplingEnabled?1:0)       << 15);
#if PT_USE_RESTIR_DI
            packed |= ((SkipEmissiveBRDF?1:0)           << 14);
#endif
            packed |= ((LightSamplingIsIndirect?1:0)    << 13);
            packed |= (LocalNEESamples & 0x3F)          << 6;     // avoid overflow by limiting sample count to RTXPT_LIGHTING_MAX_TOTAL_SAMPLE_COUNT
            packed |= (TotalSamples     & 0x3F);          // avoid overflow by limiting sample count to RTXPT_LIGHTING_MAX_TOTAL_SAMPLE_COUNT
            return packed;
        }

        static const uint SampleCountLimit()        { return (1 << 6)-1; }  // 63 is max we can pack in 6 bits

        static bool equals( const NEEBSDFMISInfo a, const NEEBSDFMISInfo b )
        {
            return     (a.LightSamplingEnabled     == b.LightSamplingEnabled    )
#if PT_USE_RESTIR_DI
                    && (a.SkipEmissiveBRDF         == b.SkipEmissiveBRDF)
#endif
                    && (a.LightSamplingIsIndirect  == b.LightSamplingIsIndirect )
                    && (a.LocalNEESamples         == b.LocalNEESamples        )
                    && (a.TotalSamples             == b.TotalSamples            );
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
        float       SpecRadianceSourceDistance;         // we actually only really care about specular radiance source distance

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
            ret.SpecRadianceSourceDistance = 0.0;
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

    struct OptimizationHints
    {
        bool    NoTextures;
        bool    NoTransmission;
        bool    OnlyDeltaLobes;

        static OptimizationHints NoHints()
        {
            OptimizationHints ret;
            ret.NoTextures = false;
            ret.NoTransmission = false;
            ret.OnlyDeltaLobes = false;
            return ret;
        }
    
        static OptimizationHints make(bool noTextures, bool noTransmission, bool onlyDeltaLobes)
        {
            OptimizationHints ret;
            ret.NoTextures = noTextures;
            ret.NoTransmission = noTransmission;
            ret.OnlyDeltaLobes = onlyDeltaLobes;
            return ret;
        }
    };
}

#endif // __PATH_TRACER_TYPES_HLSLI__