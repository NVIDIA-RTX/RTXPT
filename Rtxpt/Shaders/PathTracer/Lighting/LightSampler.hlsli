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

#include "LightingTypes.hlsli"
#include "../Utils/Utils.hlsli"
#include "LightingConfig.h"
#include "PolymorphicLightPTConfig.h"
#include "PolymorphicLight.hlsli"
#include "../Utils/Sampling/Sampling.hlsli"
#include "LightingAlgorithms.hlsli"

#define RTXPT_NEE_MIS_HEURISTIC      MISHeuristic::Balance  // MISHeuristic::PowerTwo

// Note: make sure to check IsEmpty() for case where there are no lights. Sampling when 'IsEmpty( ) == true' will result in undefined behaviour (NaNs and etc.)
struct LightSampler
{
    StructuredBuffer<LightingControlData>       ControlBuffer;              ///< control buffer containts constants like numbers of lights and importance sampling stuff; could be converted to ConstantBuffer
    StructuredBuffer<PolymorphicLightInfo>      LightsBuffer;               ///< all scene lights, encoded; NOTE: some can be unused light slots with uninitialized/old data, do not sample directly!
    StructuredBuffer<PolymorphicLightInfoEx>    LightsExBuffer;
    Buffer<uint>                                ProxyCounters;              ///< per light sampling proxy counters
    Buffer<uint>                                ProxyIndices;               ///< indices for proxies pointing to LightsBuffer, sorted 
    Buffer<uint>                                LocalSamplingBuffer;
    Texture2D<uint>                             EnvLookupMap;
    RWTexture2D<float>                          FeedbackTotalWeight;
    RWTexture2D<uint>  FeedbackCandidates;

    lpuint2                                     PixelPos;                   ///< screen pixel being lit (relevant for feedback loop and local sampling)
    uint                                        LocalSamplingTilePos;       ///< tile coord, jitter included, in 2D
    bool                                        IsScreenSpaceCoherent;

    static bool IsScreenSpaceCoherentHeuristic( StructuredBuffer<LightingControlData> controlBuffer, float rayConeWidth, float totalPathLength )
    {
        float rayConeWidthOverTotalPathTravel = rayConeWidth / totalPathLength;
        return rayConeWidthOverTotalPathTravel < controlBuffer[0].ScreenSpaceVsWorldSpaceThreshold;
    }

    static LightSampler make(
          StructuredBuffer<LightingControlData>         controlBuffer
          // ConstantBuffer<LightingControlData>       constants            // there seems to be a compiler error when using this approach
        , StructuredBuffer<PolymorphicLightInfo>        lightsBuffer
        , StructuredBuffer<PolymorphicLightInfoEx>      lightsExBuffer
        , Buffer<uint>                                  proxyCounters
        , Buffer<uint>                                  proxyIndices
        , Buffer<uint>                                  localSamplingBuffer
        , Texture2D<uint>                               envLookupMap
        , RWTexture2D<float>                            feedbackTotalWeight
        , RWTexture2D<uint>                             feedbackCandidates
        , uint2                                         pixelPos
        , bool                                          isScreenSpaceCoherent ) 
    {
        LightSampler lightSampler;

        lightSampler.IsScreenSpaceCoherent  = isScreenSpaceCoherent;

        lightSampler.ControlBuffer          = controlBuffer;
        lightSampler.LightsBuffer           = lightsBuffer;
        lightSampler.LightsExBuffer         = lightsExBuffer;
        lightSampler.ProxyCounters          = proxyCounters;
        lightSampler.ProxyIndices           = proxyIndices;
        lightSampler.LocalSamplingBuffer    = localSamplingBuffer;
        lightSampler.EnvLookupMap           = envLookupMap;
        lightSampler.FeedbackTotalWeight    = feedbackTotalWeight;
        lightSampler.FeedbackCandidates     = feedbackCandidates;
        lightSampler.PixelPos               = (lpuint2)pixelPos;

        uint2 jitteredTilePos = (pixelPos+controlBuffer[0].LocalSamplingTileJitter) / RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE.xx;
        lightSampler.LocalSamplingTilePos   = LLSB_ComputeBaseAddress(jitteredTilePos, controlBuffer[0].LocalSamplingResolution);

        // // storage for world space sits in somewhere else
        // if ( !lightSampler.IsScreenSpaceCoherent )
        // {
        //     lightSampler.PixelPos.y                += (lpuint)controlBuffer[0].FeedbackBufferHeight;
        //     lightSampler.LocalSamplingTilePos.y    += (lpuint)controlBuffer[0].TileBufferHeight;
        // }

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

    LightFeedbackReservoir GetTemporalFeedbackReservoir()
    {
        return LightFeedbackReservoir::make( PixelPos, FeedbackTotalWeight, FeedbackCandidates );
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

    void ReadLocal(uint localIndex, out uint lightIndex, out uint proxyCount)
    {
        UnpackMiniListLightAndCount( LocalSamplingBuffer[ LocalSamplingTilePos + localIndex ], lightIndex, proxyCount );
    }

    uint SampleLocal(const float rnd, out float pdf)
    {
        uint localProxyCount = RTXPT_LIGHTING_LOCAL_PROXY_COUNT;
        uint indexInIndex = clamp( uint(rnd * localProxyCount), 0, localProxyCount-1 );    // when rnd guaranteed to be [0, 1), clamp is unnecessary
        
        uint lightIndex; uint proxyCount;
        ReadLocal(indexInIndex, lightIndex, proxyCount);

        pdf = float(proxyCount) / float(localProxyCount);

        // note: app must ensure no bad lights are in the sampling buffer - out of range indices will cause a TDR
        // lightIndex = clamp( lightIndex, 0, ControlBuffer[0].TotalLightCount-1 );

        return lightIndex;
    }

    float SampleGlobalPDF(uint lightIndex)
    {
        float proxyCountPerLight = (float)ProxyCounters[lightIndex];
        return proxyCountPerLight / float(float(ControlBuffer[0].SamplingProxyCount));
    }

    float SampleLocalPDF(uint lightIndex)
    {
        const uint localProxyCount = RTXPT_LIGHTING_LOCAL_PROXY_COUNT;

        uint packedValue = LocalLightBinarySearch( LocalSamplingBuffer, LocalSamplingTilePos, lightIndex, localProxyCount, RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS );

        #if 0 // validation
        for ( int localIndex = 0; localIndex < localProxyCount; localIndex++ )
        {
            uint lightIndexR; uint proxyCountR;
            ReadLocal(localIndex, lightIndexR, proxyCountR);
            if( lightIndex == lightIndexR )
            {
                if( packedValue == RTXPT_INVALID_LIGHT_INDEX )
                    DebugPrint("Sort validation failed (not found in binary search but exists)");
                else
                    if( UnpackMiniListLight(packedValue) != lightIndexR )
                    {
                        DebugPrint("Sort validation failed (different found)");
                        // return float(proxyCountR) / float(localProxyCount);
                        break;
                    }
            }
        }
        #endif
        
        if ( packedValue == RTXPT_INVALID_LIGHT_INDEX )
            return 0.0f;

        uint lightIndexR; uint proxyCountR;
        UnpackMiniListLightAndCount(packedValue, lightIndexR, proxyCountR);
        return float(proxyCountR) / float(localProxyCount);
    }

    void InsertFeedbackFromNEE(const uint lightIndex, const float pixelRadianceContributionAvg, const float randomValue)
    {
        LightFeedbackReservoir feedbackReservoir = GetTemporalFeedbackReservoir();

        float feedbackWeight = pixelRadianceContributionAvg;
                
        // could be user option - weight power; add slider, 0.5-2.0?
        // feedbackWeight = pow(feedbackWeight, 0.8);

        // should be user option - give some positive bias [0, 1] for globally improbable lights; this (slightly) helps smaller screen regions feature more prominently in Local sampler (and Global but there it evens out)
        feedbackWeight /= pow( SampleGlobalPDF(lightIndex), 0.65 );

        if (IsScreenSpaceCoherent)
            feedbackWeight *= RTXPT_LIGHTING_SCREEN_SPACE_COHERENT_FEEDBACK_BIAS;

        feedbackReservoir.Add( randomValue, lightIndex, feedbackWeight, IsScreenSpaceCoherent );
        feedbackReservoir.CommitToStorage();
    }

    // This should probably just be removed as it only adds complexity to code
    void InsertFeedbackFromBSDF(const uint lightIndex, const float pixelRadianceContributionAvgWithoutBsdfMISWeight, const float bsdfMISWeight, const float randomNumber)
    {
#if RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK
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

    // Returns pdf-s for selecting the light given the already drawn sample in the sampler it was drawn from (thisPdf) and the other (otherPdf); 
    // Note: number of drawn samples is baked in but solidAnglePdf is not (but is already computed in lightSample.SolidAnglePdf)
    void ComputeLightSelectionPdfs(const LightSample lightSample, const uint localCandidateCount, const uint globalCandidateCount, out float thisPdf, out float otherPdf)
    {
        thisPdf       = lightSample.SelectionPdf;   // we already have this information, so no need to get it again
        [branch]if ( lightSample.FromLocalDistribution )
        {
            otherPdf    = SampleGlobalPDF(lightSample.LightIndex);
        }
        else
        {
            if( localCandidateCount != 0 )
                otherPdf = SampleLocalPDF(lightSample.LightIndex);
            else
                otherPdf = 0;
        }
    }
    void ComputeLightSelectionPdfs(const LightSample lightSample, const uint localCandidateCount, const uint globalCandidateCount, out float thisPdf, out float otherPdf, out float thisCount, out float otherCount)
    {
        thisPdf         = lightSample.SelectionPdf;   // we already have this information, so no need to get it again
        [branch]if ( lightSample.FromLocalDistribution )
        {
            otherPdf    = SampleGlobalPDF(lightSample.LightIndex);
            thisCount   = localCandidateCount;
            otherCount  = globalCandidateCount;
        }
        else
        {
            thisCount   = globalCandidateCount;
            if( localCandidateCount != 0 )
            {
                otherPdf = SampleLocalPDF(lightSample.LightIndex);
                otherCount = localCandidateCount;
            }
            else
            {   
                otherPdf = 0;
                otherCount = 0;
            }
        }
    }

//    clean up, optimize/tweak, make it part of UI, clean up so there's a ComputeLightVsBSDF_MIS_ForLight_Approx (and counterpart) so the choice is in user code

#define RTXPT_NEEAT_MIS_APPROXIMATION_LIGHTSAMPLER_BIAS 3

    float ComputeLightVsBSDF_MIS_ForLight_Approx(const LightSample lightSample, const uint candidateSampleCount, const uint fullSampleCount, float bsdfPdf)
    {
        return EvalMIS(RTXPT_NEE_MIS_HEURISTIC, fullSampleCount*candidateSampleCount, RTXPT_NEEAT_MIS_APPROXIMATION_LIGHTSAMPLER_BIAS, 1, lightSample.LightSampleableByBSDF?bsdfPdf:0);
    }
    float ComputeLightVsBSDF_MIS_ForBSDF_Approx(lpfloat bsdfPdf, const uint candidateSampleCount, const uint fullSampleCount)
    {
        return EvalMIS(RTXPT_NEE_MIS_HEURISTIC, 1, bsdfPdf, fullSampleCount*candidateSampleCount, RTXPT_NEEAT_MIS_APPROXIMATION_LIGHTSAMPLER_BIAS);
    }

    float ComputeLightVsBSDF_MIS_ForLight(const float3 surfacePosW, const LightSample lightSample, float thisPdf, float otherPdf, float thisCount, float otherCount, const uint candidateSampleCount, const uint fullSampleCount, float bsdfPdf)
    {
        float solidAnglePdf = lightSample.SolidAnglePdf;    //< both 'this' and 'other' are for the same light with same viewer and sample positions, so they will have same SolidAnglePdf

        solidAnglePdf *= LightSamplingMISBoost();           //< this could also be done by dividing bsdfPdf by LightSamplingMISBoost()

        //float lightAvgPdf = (thisPdf*sqrt(thisCount) + otherPdf*sqrt(otherCount))*sqrt(fullSampleCount);
        float lightAvgPdf = (thisPdf + otherPdf)*fullSampleCount;

        float thisMIS = EvalMIS(RTXPT_NEE_MIS_HEURISTIC, 1, lightAvgPdf*solidAnglePdf, 1, lightSample.LightSampleableByBSDF?bsdfPdf:0); // balance seems a lot less noisy than power

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
        return thisMIS; // the "/ thisCount" isn't technically part of MIS!! TODO: figure out better naming or pull out
    }

    float ComputeLightVsBSDF_MIS_ForBSDF(const uint lightIndex, lpfloat bsdfPdf, float solidAnglePdf, const uint candidateSampleCount, const uint fullSampleCount)
    {
        solidAnglePdf *= LightSamplingMISBoost();           //< this could also be done by dividing bsdfPdf by LightSamplingMISBoost()

        uint localCount, globalCount;
        GetCandidateSampleCounts(candidateSampleCount, localCount, globalCount);
        
        float globalPdf = SampleGlobalPDF(lightIndex);
        float localPdf = (localCount>0)?(SampleLocalPDF(lightIndex)):(0);

        //float lightAvgPdf = (localPdf*sqrt(localCount) + globalPdf*sqrt(globalCount))*sqrt(fullSampleCount);
        float lightAvgPdf = (localPdf + globalPdf)*fullSampleCount;

        return EvalMIS(RTXPT_NEE_MIS_HEURISTIC, 1, bsdfPdf, 1, lightAvgPdf*solidAnglePdf); // balance seems a lot less noisy than power
    }

    float ComputeBSDFMISForEmissiveTriangle(const uint emissiveTriangleLightIndex, lpfloat bsdfPdf, const float3 viewerPosition, const float3 lightSamplePosition, const uint candidateSampleCount, const uint fullSamples)
    {
        // 0 means delta lobe (zero roughness specular) - in that case LightSampling has zero chance of ever selecting a light, only BSDF can, so MIS is 1; no emissive light means missing data - that's fine, also rely on BSDF only
        if( bsdfPdf == 0 || emissiveTriangleLightIndex == RTXPT_INVALID_LIGHT_INDEX )
            return 1;

        PolymorphicLightInfoFull lightInfo = LoadLight(emissiveTriangleLightIndex);
        TriangleLight triangleLight = TriangleLight::Create(lightInfo);

        float solidAnglePdf = triangleLight.CalcSolidAnglePdfForMIS(viewerPosition, lightSamplePosition);
        return ComputeLightVsBSDF_MIS_ForBSDF(emissiveTriangleLightIndex, bsdfPdf, solidAnglePdf, candidateSampleCount, fullSamples);
    }

    float ComputeBSDFMISForEnvironmentQuad(const uint environmentQuadLightIndex, lpfloat bsdfPdf, const uint candidateSampleCount, const uint fullSamples)
    {
#if POLYLIGHT_QT_ENV_ENABLE
        // 0 means delta lobe (zero roughness specular) - in that case LightSampling has zero chance of ever selecting a light, only BSDF can, so MIS is 1; no emissive light means missing data - that's fine, also rely on BSDF only
        if( bsdfPdf == 0 || environmentQuadLightIndex == RTXPT_INVALID_LIGHT_INDEX )
            return 1;

        PolymorphicLightInfoFull lightInfo = LoadLight(environmentQuadLightIndex);
        EnvironmentQuadLight eqLight = EnvironmentQuadLight::Create(lightInfo);
        float solidAnglePdf = eqLight.CalcSolidAnglePdfForMIS(0, 0);
        return ComputeLightVsBSDF_MIS_ForBSDF(environmentQuadLightIndex, bsdfPdf, solidAnglePdf, candidateSampleCount, fullSamples);
#else
        return 1.0;
#endif
    }

    void ComputeAnalyticLightProxyContributionWithMIS(inout lpfloat3 surfaceEmission, const uint analyticLightIndex, lpfloat bsdfPdf, const float3 previousVertex, const float3 rayDir, const uint localCandidates, const uint fullSamples, const uniform bool useApproxMIS)
    {
        PolymorphicLightInfoFull lightInfo = LoadLight(analyticLightIndex);

        // only sphere lights supported here at the moment
        if( PolymorphicLight::DecodeType(lightInfo) != PolymorphicLightType::kSphere )
            return;

        SphereLight sphereLight = SphereLight::Create(lightInfo);

        float3 radiance;
        float3 lightSamplePosition;

        if( sphereLight.Eval(previousVertex, rayDir, radiance, lightSamplePosition) )
        {
            float mis = 1.0;
            if (bsdfPdf != 0) // <0 means delta lobe (zero roughness specular) - in that case LightSampling has zero chance of ever selecting a light, only BSDF can, so MIS is 1
            {
                if (useApproxMIS) // useApproxMIS must be compile time const for perf
                    mis = ComputeLightVsBSDF_MIS_ForBSDF_Approx(bsdfPdf, localCandidates, fullSamples);
                else
                {
                    float solidAnglePdf = sphereLight.CalcSolidAnglePdfForMIS(previousVertex, lightSamplePosition);
                    mis = ComputeLightVsBSDF_MIS_ForBSDF(analyticLightIndex, bsdfPdf, solidAnglePdf, localCandidates, fullSamples);
                }
            }

            surfaceEmission += lpfloat3(radiance * mis);
        }
    }

    // We can boost the MIS weights for the lighting at the expense of BSDF samples because NEE-AT is shadow-aware and material-aware; some boost is always beneficial but the amount depends on the scene
    float LightSamplingMISBoost()
    {
        return 1.0; //ControlBuffer[0].LightSampling_MIS_Boost;
    }

    PolymorphicLightInfoFull LoadLight(uint index)
    {
        PolymorphicLightInfo infoBase = LightsBuffer[index];  // no bounds checking here
        PolymorphicLightInfoEx infoExtended = (PolymorphicLightInfoEx)0;
        if ( infoBase.HasLightShaping() )
            infoExtended = LightsExBuffer[index];
        return PolymorphicLightInfoFull::make(infoBase, infoExtended);
    }

#define RTXPT_LIGHTING_NEEAT_ENABLE_WORLDSPACE_LOCAL_LAYER 0

    template<typename T>
    void GetCandidateSampleCounts(const uint totalCandidateSamples, out T localCount, out T globalCount)
    {
#if RTXPT_LIGHTING_NEEAT_ENABLE_WORLDSPACE_LOCAL_LAYER
        localCount = ::ComputeCandidateSampleLocalCount(ControlBuffer[0].LocalToGlobalSampleRatio, totalCandidateSamples);
#else
        localCount = IsScreenSpaceCoherent?(::ComputeCandidateSampleLocalCount(ControlBuffer[0].LocalToGlobalSampleRatio, totalCandidateSamples)):(0);
#endif
        globalCount = totalCandidateSamples - localCount;
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