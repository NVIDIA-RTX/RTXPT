/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __LIGHT_SAMPLER_HLSLI__
#define __LIGHT_SAMPLER_HLSLI__

#if !defined(__cplusplus)
#pragma pack_matrix(row_major)
#endif

#include "LightingTypes.h"
#include "../Utils/Utils.hlsli"
#include "LightingConfig.h"
#include "PolymorphicLight.hlsli"
#include "../Utils/Sampling/Sampling.hlsli"
#include "LightingAlgorithms.hlsli"

//#include "../../ShaderDebug.hlsli"

#define RTXPT_NEE_MIS_HEURISTIC      MISHeuristic::Balance  // MISHeuristic::PowerTwo

// Note: make sure to check IsEmpty() for case where there are no lights. Sampling when 'IsEmpty( ) == true' will result in undefined behaviour (NaNs and etc.)
struct LightSampler
{
    //ConstantBuffer<LightingControlData>         ControlConstants;           ///< all the required constants
    StructuredBuffer<LightingControlData>       ControlBuffer;              ///< control buffer containts constants like numbers of lights and importance sampling stuff; could be converted to ConstantBuffer
    StructuredBuffer<PolymorphicLightInfo>      LightsBuffer;               ///< all scene lights, encoded; NOTE: some can be unused light slots with uninitialized/old data, do not sample directly!
    StructuredBuffer<PolymorphicLightInfoEx>    LightsExBuffer;
    Buffer<uint>                                ProxyCounters;              ///< per light sampling proxy counters
    Buffer<uint>                                ProxyIndices;               ///< indices for proxies pointing to LightsBuffer, sorted 
    RWTexture2D<uint2>                          FeedbackReservoirBuffer;    ///< reservoir storing the 'the most useful' light for each pixel, used for temporal feedback and improving next frame importance sampling.
    Texture3D<uint>                             NarrowSamplingBuffer;
    Texture2D<uint>                             EnvLookupMap;

    lpuint2                                     PixelPos;                   ///< screen pixel being lit (relevant for feedback loop and narrow sampling)
    lpuint2                                     NarrowSamplingTilePos;      ///< tile coord, jitter included
    bool                                        IsDebugPixel;               ///< for visual debugging
    bool                                        IsIndirect;

    static LightSampler make(
          StructuredBuffer<LightingControlData>     controlBuffer
          // ConstantBuffer<LightingControlData>       constants            // there seems to be a compiler error when using this approach
        , StructuredBuffer<PolymorphicLightInfo>    lightsBuffer
        , StructuredBuffer<PolymorphicLightInfoEx>  lightsExBuffer
        , Buffer<uint>                              proxyCounters
        , Buffer<uint>                              proxyIndices
        , Texture3D<uint>                           narrowSamplingBuffer
        , RWTexture2D<uint2>                        feedbackReservoirBuffer
        , Texture2D<uint>                           envLookupMap
        , uint2                                     pixelPos
        , float                                     rayConeWidthOverTotalPathTravel
        , bool                                      isDebugPixel
        ) 
    {
        LightSampler lightSampler;

        lightSampler.IsIndirect                 = rayConeWidthOverTotalPathTravel >= controlBuffer[0].DirectVsIndirectThreshold;

        lightSampler.ControlBuffer              = controlBuffer;
        lightSampler.LightsBuffer               = lightsBuffer;
        lightSampler.LightsExBuffer             = lightsExBuffer;
        lightSampler.ProxyCounters              = proxyCounters;
        lightSampler.ProxyIndices               = proxyIndices;
        lightSampler.FeedbackReservoirBuffer    = feedbackReservoirBuffer;
        lightSampler.NarrowSamplingBuffer       = narrowSamplingBuffer;
        lightSampler.EnvLookupMap               = envLookupMap;
        lightSampler.PixelPos                   = (lpuint2)pixelPos;
        lightSampler.IsDebugPixel               = isDebugPixel;

        lightSampler.NarrowSamplingTilePos      = (lpuint2)((pixelPos+controlBuffer[0].LocalSamplingTileJitter) / RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE.xx);

        // storage for indirect sits in the (expanded) lower half
        if ( lightSampler.IsIndirect )
        {
            lightSampler.PixelPos.y                 += (lpuint)controlBuffer[0].FeedbackBufferHeight;
            lightSampler.NarrowSamplingTilePos.y    += (lpuint)controlBuffer[0].TileBufferHeight;
        }

        return lightSampler;
    }

    bool IsEmpty( )
    {
        return ControlBuffer[0].SamplingProxyCount == 0;
    }

    bool IsTemporalFeedbackRequired( )
    {
        return ControlBuffer[0].TemporalFeedbackRequired;
    }
    
    // returned value is in [0, ControlBuffer[0].TotalLightCount) range used to read from LightsBuffer
    uint SampleGlobal(const float rnd, out float pdf)
    {
        uint totalProxyCount = ControlBuffer[0].SamplingProxyCount;
        uint indexInIndex = clamp( uint(rnd * totalProxyCount), 0, totalProxyCount-1 );    // when rnd guaranteed to be [0, 1), clamp is unnecessary
        uint lightIndex = ProxyIndices[indexInIndex];

        float proxyCountPerLight = (float)ProxyCounters[lightIndex];
        pdf = proxyCountPerLight / float(totalProxyCount);

        return lightIndex;
    }

    void ReadNarrow(uint narrowIndex, out uint lightIndex, out uint proxyCount)
    {
        UnpackMiniListLightAndCount( NarrowSamplingBuffer[ uint3(NarrowSamplingTilePos, narrowIndex) ], lightIndex, proxyCount );
    }

    uint SampleNarrow(const float rnd, out float pdf)
    {
        uint narrowProxyCount = RTXPT_LIGHTING_NARROW_PROXY_COUNT;
        uint indexInIndex = clamp( uint(rnd * narrowProxyCount), 0, narrowProxyCount-1 );    // when rnd guaranteed to be [0, 1), clamp is unnecessary
        
        uint lightIndex; uint proxyCount;
        ReadNarrow(indexInIndex, lightIndex, proxyCount);

        pdf = float(proxyCount) / float(narrowProxyCount);

        // note: app must ensure no bad lights are in the sampling buffer - out of range indices will cause a TDR
        // lightIndex = clamp( lightIndex, 0, ControlBuffer[0].TotalLightCount-1 );

        return lightIndex;
    }

    float SampleGlobalPDF(uint lightIndex)
    {
        float proxyCountPerLight = (float)ProxyCounters[lightIndex];
        return proxyCountPerLight / float(float(ControlBuffer[0].SamplingProxyCount));
    }

    float SampleNarrowPDF(uint lightIndex)
    {
        const uint narrowProxyCount = RTXPT_LIGHTING_NARROW_PROXY_COUNT;

        uint packedValue = NarrowLightBinarySearch( NarrowSamplingBuffer, NarrowSamplingTilePos, narrowProxyCount, lightIndex );

        #if 0 // validation
        for ( int narrowIndex = 0; narrowIndex < narrowProxyCount; narrowIndex++ )
        {
            uint lightIndexR; uint proxyCountR;
            ReadNarrow(narrowIndex, lightIndexR, proxyCountR);
            if( lightIndex == lightIndexR )
            {
                if( packedValue == RTXPT_INVALID_LIGHT_INDEX )
                    DebugPrint("Sort validation failed (not found in binary search but exists)");
                else
                    if( (packedValue & 0x00FFFFFF) != lightIndexR )
                    {
                        DebugPrint("Sort validation failed (different found)");
                        // return float(proxyCountR) / float(narrowProxyCount);
                        break;
                    }
            }
        }
        #endif

        if ( packedValue == RTXPT_INVALID_LIGHT_INDEX )
            return 0.0f;

        uint lightIndexR; uint proxyCountR;
        UnpackMiniListLightAndCount(packedValue, lightIndexR, proxyCountR);
        return float(proxyCountR) / float(narrowProxyCount);
    }

    void InsertFeedbackFromNEE(inout LightFeedbackReservoir feedbackReservoir, const uint lightIndex, const float pixelRadianceContributionAvg, const float randomNumber)
    {
        float feedbackWeight = pixelRadianceContributionAvg;
                
        // could be user option - weight power; add slider, 0.5-2.0?
        // feedbackWeight = pow(feedbackWeight, 0.7);

        // should be user option - give some positive bias [0, 1] for globally improbable lights; this (slightly) helps smaller screen regions feature more prominently in Local sampler
        feedbackWeight /= pow( SampleGlobalPDF(lightIndex), 0.65 );

        feedbackReservoir.Add( randomNumber, lightIndex, feedbackWeight );
    }

    void InsertFeedbackFromBSDF(const uint lightIndex, const float pixelRadianceContributionAvgWithoutBsdfMISWeight, const float bsdfMISWeight, const float randomNumber)
    {
#if RTXPT_LIGHTING_NEEAT_ENABLE_BSDF_FEEDBACK
#if PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES  // <- reconsider this
        if( !IsTemporalFeedbackRequired() )
            return;
        
        LightFeedbackReservoir feedbackReservoir = LoadFeedback();

        float feedbackWeight = pixelRadianceContributionAvgWithoutBsdfMISWeight;

        feedbackWeight *= 0.3; //< should be user option
                
        // // should be user option - give some positive bias [0, 1] for globally improbable lights; this (slightly) helps smaller screen regions feature more prominently in Local sampler
        // feedbackWeight /= pow( SampleGlobalPDF(lightIndex), 0.65 );
        // ^commented out mainly for perf reasons

        feedbackReservoir.Add( randomNumber, lightIndex, feedbackWeight );

        StoreFeedback( feedbackReservoir, true );
            
#endif // PATH_TRACER_MODE!=PATH_TRACER_MODE_BUILD_STABLE_PLANES 
#endif
    }

    float ComputeInternalMIS(const float3 surfacePosW, const PathTracer::PathLightSample lightSample, bool isNarrow, const uint narrowSamples, const uint totalSamples, float bsdfPdf)
    {
        float thisCount;
        float otherCount;
        float thisPdf       = lightSample.SelectionPdf;
        float otherPdf;
        [branch]if ( isNarrow )
        {
            thisCount   = narrowSamples;
            otherCount  = totalSamples - narrowSamples;
            otherPdf    = SampleGlobalPDF(lightSample.LightIndex);
        }
        else
        {
            thisCount   = totalSamples - narrowSamples;
            otherCount  = narrowSamples;
            [branch]if( narrowSamples != 0 )
                otherPdf    = SampleNarrowPDF(lightSample.LightIndex);
            else
                otherPdf    = 0;
        }

        float solidAnglePdf = lightSample.SolidAnglePdf;    //< both 'this' and 'other' are for the same light with same viewer and sample positions, so they will have same SolidAnglePdf
        bsdfPdf = clamp(bsdfPdf, 0, HLF_MAX);               //< we clamp bsdf so it can safely be packed to fp16 - not needed here but needed to match the "other end"

#if defined(RTXPT_NEEAT_MIS_OVERRIDE_BSDF_PDF)
        bsdfPdf = RTXPT_NEEAT_MIS_OVERRIDE_BSDF_PDF;
#endif
#if defined(RTXPT_NEEAT_MIS_OVERRIDE_SOLID_ANGLE_PDF)
        solidAnglePdf = RTXPT_NEEAT_MIS_OVERRIDE_SOLID_ANGLE_PDF;
#endif

        solidAnglePdf *= LightSamplingMISBoost();           //< this could also be done by dividing bsdfPdf by LightSamplingMISBoost()

        float thisMIS = EvalMIS(RTXPT_NEE_MIS_HEURISTIC, thisCount, thisPdf*solidAnglePdf, otherCount, otherPdf*solidAnglePdf, 1, lightSample.LightSampleableByBSDF?bsdfPdf:0); // balance seems a lot less noisy than power

        // solidAnglePdf VALIDATION - some hits expected depending on tuning of values
        #if 0
        {
            PolymorphicLightInfoFull lightInfo = LoadLight(lightSample.LightIndex);
            if( PolymorphicLight::DecodeType(lightInfo) == PolymorphicLightType::kTriangle )
            {
                TriangleLight triangleLight = TriangleLight::Create(lightInfo);
                float solidAnglePdfTest = triangleLight.CalcSolidAnglePdfForMIS(surfacePosW, surfacePosW + lightSample.Direction * lightSample.Distance);
                if( !RelativelyEqual(solidAnglePdf, solidAnglePdfTest, 2e-2f ))
                    DebugPrint( "ERROR: lightIdx {0} solidAngle {1} solidAngleTest {2}", lightSample.LightIndex, solidAnglePdf, solidAnglePdfTest );
            }
#if POLYLIGHT_QT_ENV_ENABLE
            else if ( PolymorphicLight::DecodeType(lightInfo) == PolymorphicLightType::kEnvironmentQuad )
            {
                EnvironmentQuadLight eqLight = EnvironmentQuadLight::Create(lightInfo);
                float solidAnglePdfTest = eqLight.CalcSolidAnglePdfForMIS(surfacePosW, surfacePosW + lightSample.Direction * lightSample.Distance);
                if( !RelativelyEqual(solidAnglePdf, solidAnglePdfTest, 2e-2f ))
                    DebugPrint( "ERROR: lightIdx {0} solidAngle {1} solidAngleTest {2}", lightSample.LightIndex, solidAnglePdf, solidAnglePdfTest );
            }
#endif
        }
        #endif

        return thisMIS / thisCount; // the "/ thisCount" isn't technically part of MIS!! TODO: figure out better naming or pull out
    }

    float ComputeBSDFMIS(const uint lightIndex, lpfloat bsdfPdf, float solidAnglePdf, const uint narrowSamples, const uint totalSamples)
    {
        const uint globalSamples = totalSamples - narrowSamples;
 
        float globPdf = SampleGlobalPDF(lightIndex);
        float narrPdf = SampleNarrowPDF(lightIndex);

#if defined(RTXPT_NEEAT_MIS_OVERRIDE_BSDF_PDF)
        bsdfPdf = RTXPT_NEEAT_MIS_OVERRIDE_BSDF_PDF;
#endif
#if defined(RTXPT_NEEAT_MIS_OVERRIDE_SOLID_ANGLE_PDF)
        solidAnglePdf = RTXPT_NEEAT_MIS_OVERRIDE_SOLID_ANGLE_PDF;
#endif

        solidAnglePdf *= LightSamplingMISBoost();           //< this could also be done by dividing bsdfPdf by LightSamplingMISBoost()

        return EvalMIS(RTXPT_NEE_MIS_HEURISTIC, 1, bsdfPdf, globalSamples, globPdf*solidAnglePdf, narrowSamples, narrPdf*solidAnglePdf); // balance seems a lot less noisy than power
    }

    float ComputeBSDFMISForEmissiveTriangle(const uint emissiveTriangleLightIndex, lpfloat bsdfPdf, const float3 viewerPosition, const float3 lightSamplePosition, const uint narrowSamples, const uint totalSamples)
    {
        if( bsdfPdf == 0 )  //< 0 means delta lobe (zero roughness specular) - in that case LightSampling has zero chance of ever selecting a light, only BSDF can, so MIS is 1
            return 1;

        PolymorphicLightInfoFull lightInfo = LoadLight(emissiveTriangleLightIndex);
        TriangleLight triangleLight = TriangleLight::Create(lightInfo);

        float solidAnglePdf = triangleLight.CalcSolidAnglePdfForMIS(viewerPosition, lightSamplePosition);
        return ComputeBSDFMIS(emissiveTriangleLightIndex, bsdfPdf, solidAnglePdf, narrowSamples, totalSamples);
    }

    float ComputeBSDFMISForEnvironmentQuad(const uint environmentQuadLightIndex, lpfloat bsdfPdf, const uint narrowSamples, const uint totalSamples)
    {
#if POLYLIGHT_QT_ENV_ENABLE
        if( bsdfPdf == 0 )  //< 0 means delta lobe (zero roughness specular) - in that case LightSampling has zero chance of ever selecting a light, only BSDF can, so MIS is 1
            return 1;

        PolymorphicLightInfoFull lightInfo = LoadLight(environmentQuadLightIndex);
        EnvironmentQuadLight eqLight = EnvironmentQuadLight::Create(lightInfo);
        float solidAnglePdf = eqLight.CalcSolidAnglePdfForMIS(0, 0);
        return ComputeBSDFMIS(environmentQuadLightIndex, bsdfPdf, solidAnglePdf, narrowSamples, totalSamples);
#else
        return 1.0;
#endif
    }

    // We can boost the MIS weights for the lighting at the expense of BSDF samples because NEE-AT is shadow-aware and material-aware; some boost is always beneficial but the amount depends on the scene
    float LightSamplingMISBoost()
    {
        return ControlBuffer[0].LightSampling_MIS_Boost;
    }

    PolymorphicLightInfoFull LoadLight(uint index)
    {
        PolymorphicLightInfo infoBase = LightsBuffer[index];  // no bounds checking here
        PolymorphicLightInfoEx infoExtended = (PolymorphicLightInfoEx)0;
        if ( infoBase.HasLightShaping() )
            infoExtended = LightsExBuffer[index];
        return PolymorphicLightInfoFull::make(infoBase, infoExtended);
    }

    LightFeedbackReservoir LoadFeedback()
    {
        return LightFeedbackReservoir::Unpack8Byte(FeedbackReservoirBuffer[PixelPos]);
    }

    void StoreFeedback(const LightFeedbackReservoir feedback, bool skipIfEmpty)
    {
        // skipIfEmpty not implemented
        FeedbackReservoirBuffer[PixelPos] = feedback.Pack8Byte();
    }

    uint ComputeNarrowSampleCount(const uint totalSamples)
    {
#if RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER
        return ::ComputeNarrowSampleCount(ControlBuffer[0].NarrowFeedbackUseRatio, totalSamples);
#else
        return IsIndirect?(0):(::ComputeNarrowSampleCount(ControlBuffer[0].NarrowFeedbackUseRatio, totalSamples));
#endif
    }

    // this is a local direction - envMap.ToLocal(worldDir)
    uint LookupEnvLightByDirection( float3 localDir )
    {
        float2 uv = ndir_to_oct_equal_area_unorm(localDir);
        
        uint width, height;
        EnvLookupMap.GetDimensions(width, height);

        uint2 coord = uint2(uv * float2(width.xx));
        return EnvLookupMap.Load(uint3(coord, 0));
    }

#if 0
    // Check if the index is matching the triangle - for debugging only!!!
    bool ValidateTriangleLightIndex(uint lightIndex, float3 v0, float3 v1, float3 v2, float3 faceNormal)
    {
        if ( lightIndex >= ControlBuffer[0].TotalLightCount )
        {
            DebugPrint( "Bad light index {0}", lightIndex );
            return false;
        }
        PolymorphicLightInfoFull lightPacked = LoadLight(lightIndex);
        
        if ( PolymorphicLight::DecodeType(lightPacked) != PolymorphicLightType::kTriangle )
        {
            DebugPrint( "Good light index {0}, bad light type", lightIndex );
            return false;
        }

        TriangleLight light = TriangleLight::Create(lightPacked);
        
        float3 l0 = light.base;
        float3 l1 = light.base+light.edge1;
        float3 l2 = light.base+light.edge2;

        float scale = length(v0-v1)+length(v0-v2)+length(v1-v2);
        float dist0 = min( min( length(v0-l0), length(v0-l1) ), length(v0-l2) );
        float dist1 = min( min( length(v1-l0), length(v1-l1) ), length(v1-l2) );
        float dist2 = min( min( length(v2-l0), length(v2-l1) ), length(v2-l2) );
        float maxDist = max( max( dist0, dist1 ), dist2 );
        if( maxDist > (scale * 0.03 + 0.03) ) // edge1 & edge2 are stored in fp16
        {
            DebugPrint( "v0-{0} 1-{1} 2-{2} : l0-{3} 1-{4} 2-{5} : s:{6}, md {7} ", v0, v1, v2, l0, l1, l2, scale, maxDist );
            return false;
        }

        float normDotNorm = dot(faceNormal, light.normal);
        if( normDotNorm < 0.98 && maxDist > 0.01 ) // light normals are not correct for tiny triangles due to fp16 packing errors
        {
            DebugPrint( "v0-{0} 1-{1} 2-{2} : l0-{3} 1-{4} 2-{5} : s:{6}, md {7} ", v0, v1, v2, l0, l1, l2, scale, maxDist );
            DebugPrint( "d-{0} 1-{1} 2-{2}", normDotNorm, faceNormal, light.normal );
            return false;
        }
        
        return true;
    }
    // Check if the index is matching the triangle - for debugging only!!!
    bool ValidateEnvironmentLightIndex(uint lightIndex, float3 worldDirection)
    {
        #if POLYLIGHT_QT_ENV_ENABLE
        if ( lightIndex >= ControlBuffer[0].TotalLightCount )
        {
            DebugPrint( "Bad light index {0}", lightIndex );
            return false;
        }
        PolymorphicLightInfoFull lightPacked = LoadLight(lightIndex);
        
        if ( PolymorphicLight::DecodeType(lightPacked) != PolymorphicLightType::kEnvironmentQuad )
        {
            DebugPrint( "Good light index {0}, bad light type (not kEnvironmentQuad)", lightIndex );
            return false;
        }

        EnvironmentQuadLight light = EnvironmentQuadLight::Create(lightPacked);

        const float eps = 1e-7f;
        float2 subTexelPosMin = float2( ((float)light.NodeX+0-eps) / (float)light.NodeDim, ((float)light.NodeY+0-eps) / (float)light.NodeDim );
        float2 subTexelPosMax = float2( ((float)light.NodeX+1+eps) / (float)light.NodeDim, ((float)light.NodeY+1+eps) / (float)light.NodeDim );

        float3 localDir = EnvironmentQuadLight::ToLocal(worldDirection);
        float2 uv = ndir_to_oct_equal_area_unorm(localDir);

        if ( !( all(subTexelPosMin <= uv) && all(subTexelPosMax >= uv) ) )
        {
            // DebugPrint( "UV {0} of range {1}-{2}", uv, subTexelPosMin, subTexelPosMax );
            // There will be rare genuine errors at the borders - this is unresolved but hidden with the eps, except when UVs are very close to 0 or 1 (these aren't hidden)
            return false;
        }
        return true;
        #endif // #if POLYLIGHT_QT_ENV_ENABLE
        return false;
    }
#endif

};

inline void DebugDrawLight(PolymorphicLightInfo lightInfo, float size, float3 color)
{
    // TODO: draw actual triangle or whatever the light is
    DebugCross( lightInfo.Center, size, float4(color, 1.0f) );
}

#endif // #define __LIGHT_SAMPLER_HLSLI__