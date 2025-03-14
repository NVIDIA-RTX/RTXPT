/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __POST_PROCESS_HLSL__
#define __POST_PROCESS_HLSL__

#define VIEWZ_SKY_MARKER        FLT_MAX             // for 16bit use HLF_MAX but make sure it's bigger than commonSettings.denoisingRange in NRD!

#if defined(STABLE_PLANES_DEBUG_VIZ)

#include "Bindings/ShaderResourceBindings.hlsli"
#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"
#include "PathTracer/Utils/Utils.hlsli"

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;
    // u_DebugVizOutput[pixelPos] = float4(1,0,1,1);
    // return;

    DebugContext debug; debug.Init( pixelPos, g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput );
    const Ray cameraRay = Bridge::computeCameraRay( pixelPos, 0 );
    StablePlanesContext stablePlanes = StablePlanesContext::make(pixelPos, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, u_SecondarySurfaceRadiance, g_Const.ptConsts);

#if ENABLE_DEBUG_VIZUALISATION
    debug.StablePlanesDebugViz(stablePlanes);
#endif
}

#endif

//

#if defined(DENOISER_PREPARE_INPUTS)

#include "Bindings/ShaderResourceBindings.hlsli"
#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"
#include "NRD/DenoiserNRD.hlsli"

float ComputeNeighbourDisocclusionRelaxation(const StablePlanesContext stablePlanes, const int2 pixelPos, const int2 imageSize, const uint stablePlaneIndex, const float3 rayDirC, const int2 offset)
{
    const float kEdge = 0.02;

    uint2 pixelPosN = clamp( int2(pixelPos)+offset, 0.xx, (imageSize-1.xx) );
    uint bidN = stablePlanes.GetBranchID(pixelPosN, stablePlaneIndex);
    if( bidN == cStablePlaneInvalidBranchID )
        return kEdge;
    uint spAddressN = stablePlanes.PixelToAddress( pixelPosN, stablePlaneIndex ); 
    float3 rayDirN = normalize( stablePlanes.StablePlanesUAV[spAddressN].RayDirSceneLength.xyz );
    return 1-dot(rayDirC, rayDirN);
}

float ComputeDisocclusionRelaxation(const StablePlanesContext stablePlanes, const uint2 pixelPos, const uint stablePlaneIndex, const uint spBranchID, const StablePlane sp)
{
    float disocclusionRelax = 0;

    const int2 imageSize = int2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight);
    const float3 rayDirC = normalize(sp.RayDirSceneLength.xyz);

    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(stablePlanes, pixelPos, imageSize, stablePlaneIndex, rayDirC, int2(-1, 0));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(stablePlanes, pixelPos, imageSize, stablePlaneIndex, rayDirC, int2( 1, 0));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(stablePlanes, pixelPos, imageSize, stablePlaneIndex, rayDirC, int2( 0,-1));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(stablePlanes, pixelPos, imageSize, stablePlaneIndex, rayDirC, int2( 0, 1));
#if 0 // add diagonals for more precision (at a cost!)
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(stablePlanes, pixelPos, imageSize, stablePlaneIndex, rayDirC, int2(-1,-1));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(stablePlanes, pixelPos, imageSize, stablePlaneIndex, rayDirC, int2( 1,-1));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(stablePlanes, pixelPos, imageSize, stablePlaneIndex, rayDirC, int2(-1, 1));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(stablePlanes, pixelPos, imageSize, stablePlaneIndex, rayDirC, int2( 1, 1));
    disocclusionRelax *= 0.5;
#endif
    return saturate( (disocclusionRelax-0.00002) * 25 );
}

void NRDRadianceClamp( inout float4 radianceHitT, const float rangeK )
{
    const float kClampMin = g_Const.ptConsts.preExposedGrayLuminance/rangeK;
    const float kClampMax = min( 255.0, g_Const.ptConsts.preExposedGrayLuminance*rangeK );  // using absolute max of 255 due to NRD internal overflow when using FP16 to store luminance squared

    const float lum = luminance( radianceHitT.xyz );
    //if (lum < kClampMin)
    //    radianceHitT.xyzw = 0.0.xxxx;
    //else
    if (lum > kClampMax)
        radianceHitT.xyz *= kClampMax / lum;
}

float3 ReinhardMax(float3 color)
{
    float luminance = max( 1e-7, max(max(color.x, color.y), color.z) ); // instead of luminance, use max - this ensures output is always [0, 1]
    float reinhard = luminance / (luminance + 1);
    return color * (reinhard / luminance);
}

#if defined(DENOISER_DLSS_RR)

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;

    float3 combinedRadiance = u_Output[pixelPos].rgb; // we already have direct (non-noisy, stable) radiance in here

#define MIX_STABLE_RADIANCE 1 // stable radiance comes from emitters including sky; it is good to have it as guidance
#define MIX_BY_THROUGHPUT   1 // mix guide buffers from layers based on layer throughputs
#define ALLOW_MIX_NORMALS   1


#if MIX_STABLE_RADIANCE
    float3 stableAlbedo = sqrt(ReinhardMax(combinedRadiance));
    float stableAlbedoAvg = average(stableAlbedo);
#endif

    DebugContext debug; debug.Init( pixelPos, g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput );
    const Ray cameraRay = Bridge::computeCameraRay( pixelPos, 0 );
    StablePlanesContext stablePlanes = StablePlanesContext::make(pixelPos, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, u_SecondarySurfaceRadiance, g_Const.ptConsts);

    uint dominantStablePlaneIndex = stablePlanes.LoadDominantIndexCenter();

    float3 guideNormals = float3(0, 0, 1e-6);
    float3 diffAlbedo = float3(0.0, 0.0, 0.0);
    float3 specAlbedo = float3(0.0, 0.0, 0.0);
    float  roughness  = 0.0;

    float spWeights[4] = { 1, 0, 0, 0 };
    for( uint stablePlaneIndex = 1; stablePlaneIndex < g_Const.ptConsts.activeStablePlaneCount; stablePlaneIndex++ )
    {
        uint spBranchID = stablePlanes.GetBranchIDCenter(stablePlaneIndex);
        if( spBranchID != cStablePlaneInvalidBranchID )
        {
            uint spAddress = GenericTSPixelToAddress(pixelPos, stablePlaneIndex, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);
            float3 throughput; float3 motionVectors;
            UnpackTwoFp32ToFp16(u_StablePlanesBuffer[spAddress].PackedThpAndMVs, throughput, motionVectors);

            float weight = saturate(average(throughput));
            spWeights[stablePlaneIndex] = weight;
            spWeights[0] = saturate( spWeights[0] - weight );
        }
    }
#if 1 // lift up those with very low values
    for( uint stablePlaneIndex = 0; stablePlaneIndex < g_Const.ptConsts.activeStablePlaneCount; stablePlaneIndex++ )
    {
        if (spWeights[stablePlaneIndex]==0) continue;
        spWeights[stablePlaneIndex] = saturate( pow(spWeights[stablePlaneIndex], 0.8) );
        //spWeights[stablePlaneIndex] *= (stablePlaneIndex == dominantStablePlaneIndex)?(1.5):(1.0);  // and a small boost for the dominant?
    }
#endif
    float spWeightTotal = 0;
    for( uint stablePlaneIndex = 0; stablePlaneIndex < g_Const.ptConsts.activeStablePlaneCount; stablePlaneIndex++ )
        spWeightTotal += spWeights[stablePlaneIndex];
    for( uint stablePlaneIndex = 0; stablePlaneIndex < g_Const.ptConsts.activeStablePlaneCount; stablePlaneIndex++ )
        spWeights[stablePlaneIndex] = saturate( spWeights[stablePlaneIndex] / spWeightTotal );

    //u_DebugVizOutput[pixelPos] = float4(spWeights[0], spWeights[1], spWeights[2], 1);

    for( uint stablePlaneIndex = 0; stablePlaneIndex < g_Const.ptConsts.activeStablePlaneCount; stablePlaneIndex++ )
    {
        uint spBranchID = stablePlanes.GetBranchIDCenter(stablePlaneIndex);
        if( spBranchID != cStablePlaneInvalidBranchID )
        {
            StablePlane sp = stablePlanes.LoadStablePlane(pixelPos, stablePlaneIndex);

            const HitInfo hit = HitInfo(sp.PackedHitInfo);
            bool hitSurface = hit.isValid(); // && hit.getType() == HitType::Triangle;
            if( hitSurface ) // skip sky!
            {
                // hasSurface = true;

                float4 denoiserDiffRadianceHitDist;
                float4 denoiserSpecRadianceHitDist;
                UnpackTwoFp32ToFp16(sp.DenoiserPackedRadianceHitDist, denoiserDiffRadianceHitDist, denoiserSpecRadianceHitDist);

                combinedRadiance += denoiserDiffRadianceHitDist.rgb;
                combinedRadiance += denoiserSpecRadianceHitDist.rgb;

#if MIX_BY_THROUGHPUT
                float3 throughput; float3 motionVectors;
                UnpackTwoFp32ToFp16(sp.PackedThpAndMVs, throughput, motionVectors);
                
                float weight = spWeights[stablePlaneIndex];
                if (weight>1e-6)
                {
                    float3 diffBSDFEstimate, specBSDFEstimate;
                    UnpackTwoFp32ToFp16(sp.DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate);

#if ALLOW_MIX_NORMALS
                    guideNormals    += weight * sp.DenoiserNormalRoughness.xyz;
#endif
                    roughness       += weight * sp.DenoiserNormalRoughness.w;
                    diffAlbedo      += weight * diffBSDFEstimate;
                    specAlbedo      += weight * specBSDFEstimate;
                }
                if( stablePlaneIndex == dominantStablePlaneIndex )
                {
#if !ALLOW_MIX_NORMALS
                    guideNormals    = sp.DenoiserNormalRoughness.xyz;
#endif
                }
#else
                if( stablePlaneIndex == dominantStablePlaneIndex )
                {
                    float3 diffBSDFEstimate, specBSDFEstimate;
                    UnpackTwoFp32ToFp16(sp.DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate);

                    guideNormals = sp.DenoiserNormalRoughness.xyz;
                    roughness = sp.DenoiserNormalRoughness.w;
                    diffAlbedo = diffBSDFEstimate;
                    specAlbedo = specBSDFEstimate;
                }
#endif
            }
        }
    }

#if MIX_STABLE_RADIANCE 
    diffAlbedo = lerp( diffAlbedo, stableAlbedo, stableAlbedoAvg / (average(diffAlbedo)+sqrt(stableAlbedoAvg)+1e-7) );
    specAlbedo = lerp( specAlbedo, stableAlbedo, stableAlbedoAvg / (average(specAlbedo)+sqrt(stableAlbedoAvg)+1e-7) );
#endif

    // must be in sane range
    float guideNormalsLength = length(guideNormals);
    if (guideNormalsLength < 1e-5)
        guideNormals = float3(0, 0, 1);
    else
        guideNormals /= guideNormalsLength;

    // avoid settings both diff and spec albedo guides to 0
    const float minAlbedo = 0.02;
    if (average(diffAlbedo+specAlbedo) < minAlbedo )
        diffAlbedo += minAlbedo;

    u_Output[pixelPos] = float4(combinedRadiance, 1.0);

    u_RRDiffuseAlbedo[pixelPos]         = float4(diffAlbedo, 1);
    u_RRSpecAlbedo[pixelPos]            = float4(specAlbedo, 1);
    u_RRNormalsAndRoughness[pixelPos]   = float4(guideNormals, roughness);
    u_RRSpecMotionVectors[pixelPos]     = u_MotionVectors[pixelPos].xy; // just copy for now?

    //u_DebugVizOutput[pixelPos] = float4(diffAlbedo.rgb, 1);
    //u_DebugVizOutput[pixelPos] = float4(specAlbedo.rgb, 1);
    //u_DebugVizOutput[pixelPos] = float4(DbgShowNormalSRGB(guideNormals), 1);
}

#else // !defined(DENOISER_DLSS_RR)  <- !RR is NRD

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint stablePlaneIndex = g_MiniConst.params[0];

    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;

    DebugContext debug; debug.Init( pixelPos, g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput );
    const Ray cameraRay = Bridge::computeCameraRay( pixelPos, 0 );
    StablePlanesContext stablePlanes = StablePlanesContext::make(pixelPos, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, u_SecondarySurfaceRadiance, g_Const.ptConsts);

    bool hasSurface = false;
    uint spBranchID = stablePlanes.GetBranchIDCenter(stablePlaneIndex);
    if( spBranchID != cStablePlaneInvalidBranchID )
    {
        StablePlane sp = stablePlanes.LoadStablePlane(pixelPos, stablePlaneIndex);

        const HitInfo hit = HitInfo(sp.PackedHitInfo);
        bool hitSurface = hit.isValid(); // && hit.getType() == HitType::Triangle;
        if( hitSurface ) // skip sky!
        {
            hasSurface = true;
            float3 diffBSDFEstimate, specBSDFEstimate;
            UnpackTwoFp32ToFp16(sp.DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate);
            //diffBSDFEstimate = 1.xxx; specBSDFEstimate = 1.xxx;

            float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * length(sp.RayDirSceneLength);
            float4 viewPos = mul(float4(/*bridgedData.shadingData.posW*/virtualWorldPos, 1), g_Const.view.matWorldToView);
            float virtualViewspaceZ = viewPos.z;

            float3 thp; float3 motionVectors;
            UnpackTwoFp32ToFp16(sp.PackedThpAndMVs, thp, motionVectors);

#if 0  // for testing correctness: compute first hit surface motion vector
            {
                float3 virtualWorldPos1 = cameraRay.origin + cameraRay.dir * stablePlanes.LoadFirstHitRayLength(pixelPos);
                motionVectors = Bridge::computeMotionVector(virtualWorldPos1, virtualWorldPos1);
            }
#endif

            // See if possible to get rid of these copies - or compress them better!
            u_DenoiserViewspaceZ[pixelPos]          = virtualViewspaceZ;
            u_DenoiserMotionVectors[pixelPos]       = float4(motionVectors, 0);
            float finalRoughness = sp.DenoiserNormalRoughness.w;         

            float disocclusionRelax = 0.0;
            float aliasingDampen = 0.0;

            float specularSuppressionMul = 1.0; // this applies 
            if (stablePlaneIndex == 0 && g_Const.ptConsts.stablePlanesSuppressPrimaryIndirectSpecularK != 0.0 && g_Const.ptConsts.activeStablePlaneCount > 1 )
            {   // only apply suppression on sp 0, and only if more than 1 stable plane enabled, and only if other stable planes are in use (so they captured some of specular radiance)
                bool shouldSuppress = true;
                for (int i = 1; i < g_Const.ptConsts.activeStablePlaneCount; i++ )
                    shouldSuppress &= stablePlanes.GetBranchIDCenter(i) != cStablePlaneInvalidBranchID;
                // (optional, experimental, for future: also don't apply suppression if rough specular)
                float roughnessModifiedSuppression = g_Const.ptConsts.stablePlanesSuppressPrimaryIndirectSpecularK; // * saturate(1 - (finalRoughness - g_Const.ptConsts.stablePlanesMinRoughness)*5);
                specularSuppressionMul = shouldSuppress?saturate(1-roughnessModifiedSuppression):specularSuppressionMul;
            }

            int vertexIndex = StablePlanesVertexIndexFromBranchID( spBranchID );
            if (vertexIndex > 1)
                disocclusionRelax = ComputeDisocclusionRelaxation(stablePlanes, pixelPos, stablePlaneIndex, spBranchID, sp);
            u_DenoiserDisocclusionThresholdMix[pixelPos] = disocclusionRelax;

            // adjust for thp and map to [0,1]
            u_CombinedHistoryClampRelax[pixelPos] = saturate(u_CombinedHistoryClampRelax[pixelPos] + disocclusionRelax * saturate(luminance(thp)) );
            
            finalRoughness = saturate( finalRoughness + disocclusionRelax );
            
            float4 denoiserDiffRadianceHitDist;
            float4 denoiserSpecRadianceHitDist;
            UnpackTwoFp32ToFp16(sp.DenoiserPackedRadianceHitDist, denoiserDiffRadianceHitDist, denoiserSpecRadianceHitDist);

            float fallthroughToBasePlane = saturate(disocclusionRelax-1.0+g_Const.ptConsts.stablePlanesAntiAliasingFallthrough);
            if (stablePlaneIndex > 0 && fallthroughToBasePlane > 0)
            {
                uint sp0Address = stablePlanes.PixelToAddress( pixelPos, 0 ); 

#if 0 // this will adjust hit length so that the fallthrough is added - but I couldn't notice any quality difference so leaving it out for now since it' not free
                float p0SceneLength = length(stablePlanes.StablePlanesUAV[sp0Address].RayDirSceneLength);
                float addedHitTLength = max( 0, length(sp.RayDirSceneLength) - p0SceneLength );
#else
                float addedHitTLength = 0;
#endif

                float4 currentDiff, currentSpec; 
                UnpackTwoFp32ToFp16(stablePlanes.StablePlanesUAV[sp0Address].DenoiserPackedRadianceHitDist, currentDiff, currentSpec);
                currentDiff.xyzw = StablePlaneCombineWithHitTCompensation(currentDiff, denoiserDiffRadianceHitDist.xyz * fallthroughToBasePlane, denoiserDiffRadianceHitDist.w+addedHitTLength);
                currentSpec.xyzw = StablePlaneCombineWithHitTCompensation(currentSpec, denoiserSpecRadianceHitDist.xyz * fallthroughToBasePlane, denoiserSpecRadianceHitDist.w+addedHitTLength);
                stablePlanes.StablePlanesUAV[sp0Address].DenoiserPackedRadianceHitDist = PackTwoFp32ToFp16(currentDiff, currentSpec);
                denoiserDiffRadianceHitDist.xyz *= (1-fallthroughToBasePlane);
                denoiserSpecRadianceHitDist.xyz *= (1-fallthroughToBasePlane);

#if 0   // debug viz
                u_DebugVizOutput[pixelPos].a = 1;
                if( stablePlaneIndex == 1 ) u_DebugVizOutput[pixelPos].x = fallthroughToBasePlane;
                if( stablePlaneIndex == 2 ) u_DebugVizOutput[pixelPos].y = fallthroughToBasePlane;
#endif
            }

            // demodulate
            denoiserDiffRadianceHitDist.xyz /= diffBSDFEstimate.xyz;
            denoiserSpecRadianceHitDist.xyz /= specBSDFEstimate.xyz;

            // apply suppression if any
            denoiserSpecRadianceHitDist.xyz *= specularSuppressionMul;

            u_DenoiserNormalRoughness[pixelPos]     = NRD_FrontEnd_PackNormalAndRoughness( sp.DenoiserNormalRoughness.xyz, finalRoughness, 0 );

            // Clamp the inputs to be within sensible range.
            NRDRadianceClamp( denoiserDiffRadianceHitDist, g_Const.ptConsts.denoiserRadianceClampK*16 );
            NRDRadianceClamp( denoiserSpecRadianceHitDist, g_Const.ptConsts.denoiserRadianceClampK*16 );

    #if USE_RELAX
            u_DenoiserDiffRadianceHitDist[pixelPos] = RELAX_FrontEnd_PackRadianceAndHitDist( denoiserDiffRadianceHitDist.xyz, denoiserDiffRadianceHitDist.w, true );
            u_DenoiserSpecRadianceHitDist[pixelPos] = RELAX_FrontEnd_PackRadianceAndHitDist( denoiserSpecRadianceHitDist.xyz, denoiserSpecRadianceHitDist.w, true );
    #else
            float4 hitParams = g_Const.denoisingHitParamConsts;
            float diffNormHitDistance = REBLUR_FrontEnd_GetNormHitDist( denoiserDiffRadianceHitDist.w, virtualViewspaceZ, hitParams, 1);
            u_DenoiserDiffRadianceHitDist[pixelPos] = REBLUR_FrontEnd_PackRadianceAndNormHitDist( denoiserDiffRadianceHitDist.xyz, diffNormHitDistance, true );
            float specNormHitDistance = REBLUR_FrontEnd_GetNormHitDist( denoiserSpecRadianceHitDist.w, virtualViewspaceZ, hitParams, sp.DenoiserNormalRoughness.w);
            u_DenoiserSpecRadianceHitDist[pixelPos] = REBLUR_FrontEnd_PackRadianceAndNormHitDist( denoiserSpecRadianceHitDist.xyz, specNormHitDistance, true );
    #endif
        }
    }
    
    // if no surface (sky or no data) mark the pixel for NRD as unused; all the other inputs will be ignored
    if( !hasSurface )
        u_DenoiserViewspaceZ[pixelPos]          = VIEWZ_SKY_MARKER;

    // // manual debug viz, just in case
    if( stablePlaneIndex == 2 )
    {
    //    u_DebugVizOutput[pixelPos] = float4( 0.5 + u_DenoiserMotionVectors[pixelPos] * float3(0.2, 0.2, 10), 1 );
    //        u_DebugVizOutput[pixelPos] = float4( frac(u_DenoiserViewspaceZ[pixelPos].xxx), 1 );
    //    //u_DebugVizOutput[pixelPos] = float4( DbgShowNormalSRGB(u_DenoiserNormalRoughness[pixelPos].xyz), 1 );
    //    u_DebugVizOutput[pixelPos] = float4( u_DenoiserNormalRoughness[pixelPos].www, 1 );
    //    //u_DebugVizOutput[pixelPos] = float4( u_DenoiserDiffRadianceHitDist[pixelPos].xyz, 1 );
    //    //u_DebugVizOutput[pixelPos] = float4( u_DenoiserSpecRadianceHitDist[pixelPos].xyz, 1 );
    //    //u_DebugVizOutput[pixelPos] = float4( u_DenoiserDiffRadianceHitDist[pixelPos].www / 100.0, 1 );
    //    //u_DebugVizOutput[pixelPos] = float4( u_DenoiserSpecRadianceHitDist[pixelPos].www / 100.0, 1 );
    }
}

#endif // #if defined(DENOISER_DLSS_RR)

#endif

//

#if defined(DENOISER_FINAL_MERGE)

#pragma pack_matrix(row_major)
#include <donut/shaders/binding_helpers.hlsli>
#include "SampleConstantBuffer.h"
#include "NRD/DenoiserNRD.hlsli"
#include "PathTracer/StablePlanes.hlsli"

ConstantBuffer<SampleConstants>         g_Const             : register(b0);
VK_PUSH_CONSTANT ConstantBuffer<SampleMiniConstants>     g_MiniConst         : register(b1);

RWTexture2D<float4>     u_InputOutput                           : register(u0);
RWTexture2D<float4>     u_DebugVizOutput                        : register(u1);
Texture2D<float4>       t_DiffRadiance                          : register(t2);
Texture2D<float4>       t_SpecRadiance                          : register(t3);
Texture2D<float4>       t_DenoiserValidation                    : register(t5);
Texture2D<float>        t_DenoiserViewspaceZ                    : register(t6);
Texture2D<float>        t_DenoiserDisocclusionThresholdMix      : register(t7);
StructuredBuffer<StablePlane> t_StablePlanesBuffer              : register(t10);

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint stablePlaneIndex = g_MiniConst.params.x;

    uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ))
        return;

    float4 diffRadiance = 0.0.xxxx;
    float4 specRadiance = 0.0.xxxx;
    float relaxedDisocclusion = 0; 

    bool hasSurface = t_DenoiserViewspaceZ[pixelPos] != VIEWZ_SKY_MARKER;

    uint spAddress = GenericTSPixelToAddress(pixelPos, stablePlaneIndex, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);

    // skip sky!
    if (hasSurface)
    {
        float3 diffBSDFEstimate, specBSDFEstimate;
        UnpackTwoFp32ToFp16(t_StablePlanesBuffer[spAddress].DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate);
        //diffBSDFEstimate = 1.xxx; specBSDFEstimate = 1.xxx;

        relaxedDisocclusion = t_DenoiserDisocclusionThresholdMix[pixelPos];
    #if 1 // classic
        diffRadiance = t_DiffRadiance[pixelPos];
        specRadiance = t_SpecRadiance[pixelPos];
    #else // re-jitter! requires edge-aware filter to actually work correctly
        float2 pixelSize = 1.0.xx / (float2)g_Const.ptConsts.camera.viewportSize;
        float2 samplingUV = (pixelPos.xy + float2(0.5, 0.5) + g_Const.ptConsts.camera.jitter) * pixelSize;
        diffRadiance = t_DiffRadiance.SampleLevel( g_Sampler, samplingUV, 0 );
        specRadiance = t_SpecRadiance.SampleLevel( g_Sampler, samplingUV, 0 );
    #endif

        DenoiserNRD::PostDenoiseProcess(diffBSDFEstimate, specBSDFEstimate, diffRadiance, specRadiance);
    }

#if ENABLE_DEBUG_VIZUALISATION
    if (g_Const.debug.debugViewType >= (int)DebugViewType::StablePlaneRelaxedDisocclusion && g_Const.debug.debugViewType <= ((int)DebugViewType::StablePlaneDenoiserValidation))
    {
        bool debugThisPlane = g_Const.debug.debugViewStablePlaneIndex == stablePlaneIndex;
        uint2 outDebugPixelPos = pixelPos;
        const uint2 screenSize = uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight);
        const uint2 halfSize = screenSize / 2;
        // figure out where we are in the small quad view
        if (g_Const.debug.debugViewStablePlaneIndex == -1)
        {
            const uint2 quadrant = uint2(stablePlaneIndex%2, stablePlaneIndex/2);
            debugThisPlane = true; 
            outDebugPixelPos = quadrant * halfSize + pixelPos / 2;
        }

        // draw checkerboard pattern for unused stable planes
        if (g_Const.debug.debugViewStablePlaneIndex == -1 && stablePlaneIndex == 0)
        {
            uint quadPlaneIndex = (pixelPos.x >= halfSize.x) + 2 * (pixelPos.y >= halfSize.y);
            if (quadPlaneIndex >= g_Const.ptConsts.activeStablePlaneCount)
                u_DebugVizOutput[pixelPos] = float4( ((pixelPos.x+pixelPos.y)%2).xxx, 1 );
        }
        
        if (debugThisPlane)
        {
            float viewZ = t_DenoiserViewspaceZ[pixelPos].x;
            float4 validation = t_DenoiserValidation[pixelPos].rgba;

            float3 throughput; float3 motionVectors;
            UnpackTwoFp32ToFp16(t_StablePlanesBuffer[spAddress].PackedThpAndMVs, throughput, motionVectors);

            switch (g_Const.debug.debugViewType)
            {
            // note: sqrt there is a cheap debug tonemapper :D
            case ((int)DebugViewType::StablePlaneRelaxedDisocclusion):      u_DebugVizOutput[outDebugPixelPos] = float4( sqrt(relaxedDisocclusion), 0, 0, 1 ); break;
            case ((int)DebugViewType::StablePlaneDiffRadianceDenoised):     u_DebugVizOutput[outDebugPixelPos] = float4( sqrt(diffRadiance.rgb), 1 ); break;
            case ((int)DebugViewType::StablePlaneSpecRadianceDenoised):     u_DebugVizOutput[outDebugPixelPos] = float4( sqrt(specRadiance.rgb), 1 ); break;
            case ((int)DebugViewType::StablePlaneCombinedRadianceDenoised): u_DebugVizOutput[outDebugPixelPos] = float4( sqrt(diffRadiance.rgb + specRadiance.rgb), 1 ); break;
            case ((int)DebugViewType::StablePlaneViewZ):                    u_DebugVizOutput[outDebugPixelPos] = float4( viewZ/10, frac(viewZ), 0, 1 ); break;
            case ((int)DebugViewType::StablePlaneThroughput):               u_DebugVizOutput[outDebugPixelPos] = float4( throughput, 1 ); break;
            case ((int)DebugViewType::StablePlaneDenoiserValidation):       
                if( validation.a > 0 ) 
                    u_DebugVizOutput[outDebugPixelPos] = float4( validation.rgb, 1 ); 
                else
                    u_DebugVizOutput[outDebugPixelPos] = float4( sqrt(diffRadiance.rgb + specRadiance.rgb), 1 );
                break;
            default: break;
            }
        }
    }
#endif // #if ENABLE_DEBUG_VIZUALISATION

    if (hasSurface)
        u_InputOutput[pixelPos.xy].xyz += (diffRadiance.rgb + specRadiance.rgb);
    //else
    //    u_InputOutput[pixelPos.xy].xyz = float3(1,0,0);
}
#endif

//

#if defined(DUMMY_PLACEHOLDER_EFFECT) || defined(__INTELLISENSE__)
RWBuffer<float>     u_CaptureTarget         : register(u8);
Texture2D<float>    t_CaptureSource         : register(t0);

[numthreads(1, 1, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    uint dummy0, dummy1, mipLevels; t_CaptureSource.GetDimensions(0,dummy0,dummy1,mipLevels); 
    float avgLum = t_CaptureSource.Load( int3(0, 0, mipLevels-1) );
    u_CaptureTarget[0] = avgLum;
}
#endif

#endif // __POST_PROCESS_HLSL__