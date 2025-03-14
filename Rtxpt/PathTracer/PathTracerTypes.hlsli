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
#include "PathState.hlsli"
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

/** Types of samplable lights.
*/
enum class PathLightType : uint32_t // was PathTracer::LightType
{
    EnvMap      = 0,
    Emissive    = 1,
    Analytic    = 2
};

namespace PathTracer
{
    /** Holds path tracer shader working data for state, settings, debugging, denoising and etc. Everything that is shared across (DispatchRays/Compute) draw call, between all pixels, 
        but also some pixel-specific stuff (like pixelPos heh). It's what a PathTracer instance would store if it could be an OOP object.
    */
    struct WorkingContext
    {
        PathTracerConstants     ptConsts;
        DebugContext            debug;
        StablePlanesContext     stablePlanes;
        uint2                   pixelPos;
        uint                    padding0;
        uint                    padding1;
    };

    /** All surface data returned by the Bridge::loadSurface
    */
    struct SurfaceData
    {
        ShadingData     shadingData;
        ActiveBSDF      bsdf;
        float3          prevPosW;
        lpfloat         interiorIoR;    // a.k.a. material IoR
        uint            neeLightIndex;  // 0xFFFFFFFF if none
        static SurfaceData make( /*VertexData vd, */ShadingData shadingData, ActiveBSDF bsdf, float3 prevPosW, lpfloat interiorIoR, uint neeLightIndex )
        { 
            SurfaceData ret; 
            ret.shadingData     = shadingData; 
            ret.bsdf            = bsdf; 
            ret.prevPosW        = prevPosW; 
            ret.interiorIoR     = interiorIoR; 
            ret.neeLightIndex   = neeLightIndex;
            return ret; 
        }
    
        // static SurfaceData make()
        // {
        //     SurfaceData d;
        //     d.shadingData   = ShadingData::make();
        //     d.bsdf          = ActiveBSDF::make();
        //     d.prevPosW      = 0;
        //     d.interiorIoR   = 0;
        //     return d;
        // }
    };

    /** Describes a light sample, mainly for use in NEE.
        It is considered in the context of a shaded surface point, from which Distance and Direction to light sample are computed.
        In case of emissive triangle light source, it is advisable to compute anti-self-intersection offset before computing
        distance and direction, even though distance shortening is needed anyways for shadow rays due to precision issues.
        Use ComputeVisibilityRay to correctly compute surface offset.
    */
    struct PathLightSample  // was PathTracer::LightSample in Falcor
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
        bool SkipEmissiveBRDF;          // Ignore next bounce reflective (but not transmissive) radiance because ReSTIR-DI or similar collected (or will collect) it
        bool LightSamplingIsIndirect;   // using indirect part of the LightSampler domain; otherwise using direct
        uint NarrowNEESamples;          // 
        uint TotalSamples;              //

        // Initialize to empty (NEE disabled or primary bounce or etc.)
        static NEEBSDFMISInfo empty() 
        { 
            NEEBSDFMISInfo ret;
            ret.LightSamplingEnabled     = false;
            ret.SkipEmissiveBRDF         = false;
            ret.LightSamplingIsIndirect  = false;
            ret.NarrowNEESamples         = 0;
            ret.TotalSamples             = 0;
            return ret;
        }

        static NEEBSDFMISInfo Unpack16bit( uint packed ) 
        { 
            NEEBSDFMISInfo ret;
            ret.LightSamplingEnabled     = (packed & (1 << 15)) != 0;
            ret.SkipEmissiveBRDF         = (packed & (1 << 14)) != 0;
            ret.LightSamplingIsIndirect  = (packed & (1 << 13)) != 0;
            ret.NarrowNEESamples         = (packed >> 6) & 0x3F;
            ret.TotalSamples             = (packed) & 0x3F;
            return ret;
        }

        uint    Pack16bit()
        {
            uint packed = 0;
            packed |= ((LightSamplingEnabled?1:0)       << 15);
            packed |= ((SkipEmissiveBRDF?1:0)   << 14);
            packed |= ((LightSamplingIsIndirect?1:0)    << 13);
            packed |= (NarrowNEESamples & 0x3F) << 6;     // avoid overflow by limiting sample count to RTXPT_LIGHTING_NEEAT_MAX_TOTAL_SAMPLE_COUNT
            packed |= (TotalSamples     & 0x3F);          // avoid overflow by limiting sample count to RTXPT_LIGHTING_NEEAT_MAX_TOTAL_SAMPLE_COUNT
            return packed;
        }

        static const uint SampleCountLimit()        { return (1 << 6)-1; }  // 63 is max we can pack in 6 bits

        static bool equals( const NEEBSDFMISInfo a, const NEEBSDFMISInfo b )
        {
            return     (a.LightSamplingEnabled     == b.LightSamplingEnabled    )
                    && (a.SkipEmissiveBRDF         == b.SkipEmissiveBRDF)
                    && (a.LightSamplingIsIndirect  == b.LightSamplingIsIndirect )
                    && (a.NarrowNEESamples         == b.NarrowNEESamples        )
                    && (a.TotalSamples             == b.TotalSamples            );
        }

    };

#define RTXPT_NEE_RESULT_MANUAL_PACK 1

    // Output part of the interface to the path tracer - this will likely change over time.
    struct NEEResult
    {
#if RTXPT_NEE_RESULT_MANUAL_PACK
        uint3       PackedRadiances;
#else
#if RTXPT_DIFFUSE_SPECULAR_SPLIT
        lpfloat3    _diffuseRadiance;
        lpfloat3    _specularRadiance;
#else
    #error current denoiser requires RTXPT_DIFFUSE_SPECULAR_SPLIT
#endif
#endif
        lpfloat     RadianceSourceDistance;         // consider splitting into specular and diffuse
        //lpfloat     ScatterMISWeight;               // MIS weight computed for scatter counterpart (packed to fp16 in path payload) - this is a hack for now

        NEEBSDFMISInfo BSDFMISInfo;
        
        // initialize to empty
        static NEEResult empty() 
        { 
            NEEResult ret;
            ret.SetRadiances(0,0);
            ret.BSDFMISInfo = NEEBSDFMISInfo::empty();
            return ret; 
        }

        void        GetRadiances( inout lpfloat3 diffuseRadiance, inout lpfloat3 specularRadiance )
        {
#if RTXPT_NEE_RESULT_MANUAL_PACK
            Fp16ToFp32(PackedRadiances, diffuseRadiance, specularRadiance);
#else
            diffuseRadiance = _diffuseRadiance ;
            specularRadiance = _specularRadiance;
#endif
        }

        void        SetRadiances( const lpfloat3 diffuseRadiance, const lpfloat3 specularRadiance )
        {
#if RTXPT_NEE_RESULT_MANUAL_PACK
            PackedRadiances = Fp32ToFp16(diffuseRadiance, specularRadiance);
#else
            _diffuseRadiance = diffuseRadiance;
            _specularRadiance = specularRadiance;
#endif
        }

        //bool        Valid()                 { return any((DiffuseRadiance+SpecularRadiance) > 0); }
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
        uint    SERSortKey;

        static OptimizationHints NoHints(uint sortKey)
        {
            OptimizationHints ret;
            ret.NoTextures = false;
            ret.NoTransmission = false;
            ret.OnlyDeltaLobes = false;
            ret.SERSortKey = sortKey;
            return ret;
        }
    
        static OptimizationHints make(bool noTextures, bool noTransmission, bool onlyDeltaLobes, uint sortKey)
        {
            OptimizationHints ret;
            ret.NoTextures = noTextures;
            ret.NoTransmission = noTransmission;
            ret.OnlyDeltaLobes = onlyDeltaLobes;
            ret.SERSortKey = sortKey;
            return ret;
        }
    };

    struct ScatterResult
    {
        bool    Valid;
        bool    IsDelta;
        bool    IsTransmission;
        float   Pdf;
        float3  Dir;
        
        static ScatterResult empty() 
        { 
            ScatterResult ret; 
            ret.Valid = false; 
            ret.IsDelta = false; 
            ret.IsTransmission = false; 
            ret.Pdf = 0.0; 
            ret.Dir = float3(0,0,0); 
            return ret; 
        }
    };
}

#endif // __PATH_TRACER_TYPES_HLSLI__