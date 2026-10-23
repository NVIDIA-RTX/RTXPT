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

#pragma pack_matrix(row_major)

#include "../Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl"

#define VIEWZ_SKY_MARKER        FLT_MAX             // for 16bit use HLF_MAX but make sure it's bigger than commonSettings.denoisingRange in NRD!

// Per-denoiser reflectance floor. This used to be a single clamp applied when the denoising layer was written
// (DenoisingLayers.hlsli), which forced both denoisers to share one value; the estimates are now stored unfloored and each
// denoiser applies its own - see DLSSRR_MIN_ALBEDO in DenoisingGuidesBaker.hlsl for the DLSS-RR side.
// NRD_MIN_REFLECTANCE guards the demodulate (radiance /= estimate) / re-modulate (radiance *= estimate) round trip. Both
// sides MUST use the identical value or the pair stops cancelling and energy is lost on dark surfaces.
#define NRD_MIN_REFLECTANCE     0.04

#if defined(DENOISING_LAYERS_DEBUG_VIZ)

#define NON_PATH_TRACING_PASS 1

#include "../Shaders/Bindings/ShaderResourceBindings.hlsli"
#include "../Shaders/PathTracerBridgeDonut.hlsli"
#include "../Shaders/PathTracer/PathTracer.hlsli"
#include "../Shaders/PathTracer/Utils/Utils.hlsli"

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;
    // u_DebugVizOutput[pixelPos] = float4(1,0,1,1);
    // return;

    DebugContext debug; debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack );
    const Ray cameraRay = Bridge::computeCameraRay( pixelPos );
    DenoisingLayersContext denoisingLayers = DenoisingLayersContext::make(u_DenoisingLayersHeader, u_DenoisingLayersBuffer, u_StableRadiance, g_Const.ptConsts);

#if ENABLE_DEBUG_VIZUALISATIONS
    debug.DenoisingLayersDebugViz(pixelPos, denoisingLayers);
#endif
}

#endif

//

#if defined(DENOISER_PREPARE_INPUTS)

#define NON_PATH_TRACING_PASS 1

#include "../Shaders/Bindings/ShaderResourceBindings.hlsli"
#include "../Shaders/PathTracerBridgeDonut.hlsli"
#include "../Shaders/PathTracer/PathTracer.hlsli"
#include "../NRD/DenoiserNRD.hlsli"

float ComputeNeighbourDisocclusionRelaxation(const DenoisingLayersContext denoisingLayers, const int2 pixelPos, const int2 imageSize, const uint denoisingLayerIndex, const float3 rayDirC, const int2 offset)
{
    const float kEdge = 0.02;

    uint2 pixelPosN = clamp( int2(pixelPos)+offset, 0.xx, (imageSize-1.xx) );
    uint bidN = denoisingLayers.GetBranchID(pixelPosN, denoisingLayerIndex);
    if( bidN == cDenoisingLayerInvalidBranchID )
        return kEdge;
    uint spAddressN = denoisingLayers.PixelToAddress( pixelPosN, denoisingLayerIndex ); 
    float3 rayDirN = denoisingLayers.DenoisingLayersUAV[spAddressN].GetNormal();
    return 1-dot(rayDirC, rayDirN);
}

float ComputeDisocclusionRelaxation(const DenoisingLayersContext denoisingLayers, const uint2 pixelPos, const uint denoisingLayerIndex, const uint spBranchID, const DenoisingLayer sp)
{
    float disocclusionRelax = 0;

    const int2 imageSize = int2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight);
    const float3 rayDirC = sp.GetNormal();

    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(denoisingLayers, pixelPos, imageSize, denoisingLayerIndex, rayDirC, int2(-1, 0));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(denoisingLayers, pixelPos, imageSize, denoisingLayerIndex, rayDirC, int2( 1, 0));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(denoisingLayers, pixelPos, imageSize, denoisingLayerIndex, rayDirC, int2( 0,-1));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(denoisingLayers, pixelPos, imageSize, denoisingLayerIndex, rayDirC, int2( 0, 1));
#if 0 // add diagonals for more precision (at a cost!)
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(denoisingLayers, pixelPos, imageSize, denoisingLayerIndex, rayDirC, int2(-1,-1));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(denoisingLayers, pixelPos, imageSize, denoisingLayerIndex, rayDirC, int2( 1,-1));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(denoisingLayers, pixelPos, imageSize, denoisingLayerIndex, rayDirC, int2(-1, 1));
    disocclusionRelax += ComputeNeighbourDisocclusionRelaxation(denoisingLayers, pixelPos, imageSize, denoisingLayerIndex, rayDirC, int2( 1, 1));
    disocclusionRelax *= 0.5;
#endif
    return saturate( (disocclusionRelax-0.00002) * 25 );
}


void NRDRadianceClamp( inout float3 radiance, const float rangeK )
{
    const float kClampMin = g_Const.ptConsts.preExposedGrayLuminance/rangeK;
    const float kClampMax = min( 255.0, g_Const.ptConsts.preExposedGrayLuminance*rangeK );  // using absolute max of 255 due to NRD internal overflow when using FP16 to store luminance squared

    const float lum = Luminance( radiance.xyz );
    //if (lum < kClampMin)
    //    radiance.xyzw = 0.0.xxxx;
    //else
    if (lum > kClampMax)
        radiance.xyz *= kClampMax / lum;
}


[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint denoisingLayerIndex         = g_MiniConst.params[0];
    const bool initWithStableRadiance   = g_MiniConst.params[1];

    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;

    DebugContext debug; debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack );
    const Ray cameraRay = Bridge::computeCameraRay( pixelPos );
    DenoisingLayersContext denoisingLayers = DenoisingLayersContext::make(u_DenoisingLayersHeader, u_DenoisingLayersBuffer, u_StableRadiance, g_Const.ptConsts);

    if (initWithStableRadiance)
        u_OutputColor[pixelPos] = float4( denoisingLayers.LoadStableRadiance(pixelPos), 1 );

    bool hasSurface = false;
    uint spBranchID = denoisingLayers.GetBranchID(pixelPos, denoisingLayerIndex);
    if( spBranchID != cDenoisingLayerInvalidBranchID )
    {
        DenoisingLayer sp = denoisingLayers.LoadDenoisingLayer(pixelPos, denoisingLayerIndex);

        bool hitSurface = isfinite(sp.SceneLength); 
        if( hitSurface ) // skip sky!
        {
            hasSurface = true;
            float3 diffBSDFEstimate, specBSDFEstimate;
            UnpackTwoFp32ToFp16(sp.DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate);
            //diffBSDFEstimate = 1.xxx; specBSDFEstimate = 1.xxx;

            // NRD reflectance floor - must match the re-modulate site in the resolve pass below, see NRD_MIN_REFLECTANCE
            diffBSDFEstimate = max( diffBSDFEstimate, NRD_MIN_REFLECTANCE.xxx );
            specBSDFEstimate = max( specBSDFEstimate, NRD_MIN_REFLECTANCE.xxx );

            float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sp.SceneLength;
            float4 viewPos = mul(float4(/*bridgedData.shadingData.posW*/virtualWorldPos, 1), g_Const.view.matWorldToView);
            float virtualViewspaceZ = viewPos.z;

            float3 thp; float3 toPrevPosW;
            sp.GetThpAndWorldMotionVector(thp, toPrevPosW);

            float3 virtualWorldMotion = toPrevPosW;
            float3 motionVectors = ComputeMotionVector( g_Const.view, g_Const.previousView, virtualWorldPos, virtualWorldPos+virtualWorldMotion );

            // float3 motionVectors = u_MotionVectors[pixelPos].rgb;

            // See if possible to get rid of these copies - or compress them better!
            u_DenoiserViewspaceZ[pixelPos]          = virtualViewspaceZ;
            u_DenoiserMotionVectors[pixelPos]       = float4(motionVectors, 0);

            const float kMinRoughness = 0.2;
            float finalRoughness = max( kMinRoughness, sp.GetRoughness() );

            float disocclusionRelax = 0.0;
            float aliasingDampen = 0.0;

            float specularSuppressionMul = 1.0; // this applies 
            if (denoisingLayerIndex == 0 && g_Const.ptConsts.denoisingLayersSuppressPrimaryIndirectSpecularK != 0.0 && g_Const.ptConsts.GetActiveDenoisingLayerCount() > 1 )
            {   // only apply suppression on sp 0, and only if more than 1 denoising layer enabled, and only if other denoising layers are in use (so they captured some of specular radiance)
                bool shouldSuppress = true;
                for (int i = 1; i < g_Const.ptConsts.GetActiveDenoisingLayerCount(); i++ )
                    shouldSuppress &= denoisingLayers.GetBranchID(pixelPos, i) != cDenoisingLayerInvalidBranchID;
                // (optional, experimental, for future: also don't apply suppression if rough specular)
                float roughnessModifiedSuppression = g_Const.ptConsts.denoisingLayersSuppressPrimaryIndirectSpecularK; // * saturate(1 - (finalRoughness - g_Const.ptConsts.denoisingLayersMinRoughness)*5);
                specularSuppressionMul = shouldSuppress?saturate(1-roughnessModifiedSuppression):specularSuppressionMul;
            }

            int vertexIndex = DenoisingLayersVertexIndexFromBranchID( spBranchID );
            if (vertexIndex > 1)
                disocclusionRelax = ComputeDisocclusionRelaxation(denoisingLayers, pixelPos, denoisingLayerIndex, spBranchID, sp);
            u_DenoiserDisocclusionThresholdMix[pixelPos] = disocclusionRelax;

            // adjust for thp and map to [0,1]
            u_CombinedHistoryClampRelax[pixelPos] = saturate(u_CombinedHistoryClampRelax[pixelPos] + disocclusionRelax * saturate(Luminance(thp)) );
            
            finalRoughness = saturate( finalRoughness + disocclusionRelax );
            
            float3 denoiserDiffRadiance = sp.GetNoisyDiffRadiance();
            float3 denoiserSpecRadiance = sp.GetNoisySpecRadiance();

            // demodulate
            denoiserDiffRadiance.xyz /= diffBSDFEstimate.xyz;
            denoiserSpecRadiance.xyz /= specBSDFEstimate.xyz;

            // apply suppression if any
            denoiserSpecRadiance.xyz *= specularSuppressionMul;

            u_DenoiserNormalRoughness[pixelPos]     = NRD_FrontEnd_PackNormalAndRoughness( sp.GetNormal(), finalRoughness, 0 );

            // Clamp the inputs to be within sensible range.
            NRDRadianceClamp( denoiserDiffRadiance, g_Const.ptConsts.denoiserRadianceClampK*16 );
            NRDRadianceClamp( denoiserSpecRadiance, g_Const.ptConsts.denoiserRadianceClampK*16 );

#if 1 // new specHitT
            float specHitT = 0;
            uint dominantDenoisingLayerIndex = denoisingLayers.LoadDominantIndex(pixelPos);
            if (dominantDenoisingLayerIndex == denoisingLayerIndex)
            {
                specHitT = u_SpecularHitT[pixelPos];
                // DebugPixel( pixelPos, float4( specHitT / 100.0, 1 ) );
            }
#endif


    #if USE_RELAX
            u_DenoiserDiffRadianceHitDist[pixelPos] = RELAX_FrontEnd_PackRadianceAndHitDist( denoiserDiffRadiance.xyz, 0, true );
            u_DenoiserSpecRadianceHitDist[pixelPos] = RELAX_FrontEnd_PackRadianceAndHitDist( denoiserSpecRadiance.xyz, specHitT, true );
    #else
            float3 hitParams = g_Const.denoisingHitParamConsts;
            float diffNormHitDistance = REBLUR_FrontEnd_GetNormHitDist( 0, virtualViewspaceZ, hitParams, 1);
            u_DenoiserDiffRadianceHitDist[pixelPos] = REBLUR_FrontEnd_PackRadianceAndNormHitDist( denoiserDiffRadiance.xyz, /*diffNormHitDistance*/0, true );
            float specNormHitDistance = REBLUR_FrontEnd_GetNormHitDist( specHitT, virtualViewspaceZ, hitParams, sp.GetRoughness());
            u_DenoiserSpecRadianceHitDist[pixelPos] = REBLUR_FrontEnd_PackRadianceAndNormHitDist( denoiserSpecRadiance.xyz, specNormHitDistance, true );
    #endif
        }
    }
    
    // if no surface (sky or no data) mark the pixel for NRD as unused; all the other inputs will be ignored
    if( !hasSurface )
        u_DenoiserViewspaceZ[pixelPos]          = VIEWZ_SKY_MARKER;

    // // manual debug viz, just in case
    if( denoisingLayerIndex == 2 )
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


#endif

//

#if defined(DENOISER_FINAL_MERGE)

#include <donut/shaders/binding_helpers.hlsli>
#include "../Shaders/SampleConstantBuffer.h"
#include "../NRD/DenoiserNRD.hlsli"
#include "../Shaders/PathTracer/DenoisingLayers.hlsli"

ConstantBuffer<SampleConstants>         g_Const             : register(b0);
VK_PUSH_CONSTANT ConstantBuffer<SampleMiniConstants>     g_MiniConst         : register(b1);

RWTexture2D<float4>     u_InputOutput                           : register(u0);
Texture2D<float4>       t_DiffRadiance                          : register(t2);
Texture2D<float4>       t_SpecRadiance                          : register(t3);
Texture2D<float4>       t_DenoiserValidation                    : register(t5);
Texture2D<float>        t_DenoiserViewspaceZ                    : register(t6);
Texture2D<float>        t_DenoiserDisocclusionThresholdMix      : register(t7);
StructuredBuffer<DenoisingLayer> t_DenoisingLayersBuffer              : register(t10);

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint denoisingLayerIndex = g_MiniConst.params.x;

    uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ))
        return;

    float4 diffRadiance = 0.0.xxxx;
    float4 specRadiance = 0.0.xxxx;
    float relaxedDisocclusion = 0; 

    bool hasSurface = t_DenoiserViewspaceZ[pixelPos] != VIEWZ_SKY_MARKER;

    uint spAddress = GenericTSPixelToAddress(pixelPos, denoisingLayerIndex, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);

    // skip sky!
    if (hasSurface)
    {
        float3 diffBSDFEstimate, specBSDFEstimate;
        UnpackTwoFp32ToFp16(t_DenoisingLayersBuffer[spAddress].DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate);
        //diffBSDFEstimate = 1.xxx; specBSDFEstimate = 1.xxx;

        // Re-modulate side of the NRD demodulation round trip - must use exactly the same floor as the demodulate above
        diffBSDFEstimate = max( diffBSDFEstimate, NRD_MIN_REFLECTANCE.xxx );
        specBSDFEstimate = max( specBSDFEstimate, NRD_MIN_REFLECTANCE.xxx );

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

#if ENABLE_DEBUG_VIZUALISATIONS
    if (g_Const.debug.debugViewType >= (int)DebugViewType::DenoisingLayer_RelaxedDisocclusion && g_Const.debug.debugViewType <= ((int)DebugViewType::DenoisingLayer_DenoiserValidation))
    {
        bool debugThisPlane = g_Const.debug.debugViewDenoisingLayerIndex == denoisingLayerIndex;
        uint2 outDebugPixelPos = pixelPos;
        const uint2 screenSize = uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight);
        const uint2 halfSize = screenSize / 2;
        // figure out where we are in the small quad view
        if (g_Const.debug.debugViewDenoisingLayerIndex == -1)
        {
            const uint2 quadrant = uint2(denoisingLayerIndex%2, denoisingLayerIndex/2);
            debugThisPlane = true; 
            outDebugPixelPos = quadrant * halfSize + pixelPos / 2;
        }

        // draw checkerboard pattern for unused denoising layers
        if (g_Const.debug.debugViewDenoisingLayerIndex == -1 && denoisingLayerIndex == 0)
        {
            uint quadPlaneIndex = (pixelPos.x >= halfSize.x) + 2 * (pixelPos.y >= halfSize.y);
            if (quadPlaneIndex >= g_Const.ptConsts.GetActiveDenoisingLayerCount())
                DebugPixel( pixelPos, float4( ((pixelPos.x+pixelPos.y)%2).xxx, 1 ) );
        }
        
        if (debugThisPlane)
        {
            float viewZ = t_DenoiserViewspaceZ[pixelPos].x;
            float4 validation = t_DenoiserValidation[pixelPos].rgba;

            float3 throughput; float3 dummy;
            UnpackTwoFp32ToFp16(t_DenoisingLayersBuffer[spAddress].PackedThpAndMVs, throughput, dummy);
            // TODO: use GetThp()

            switch (g_Const.debug.debugViewType)
            {
            // note: sqrt there is a cheap debug tonemapper :D
            case ((int)DebugViewType::DenoisingLayer_RelaxedDisocclusion):      DebugPixel( outDebugPixelPos, float4( sqrt(relaxedDisocclusion), 0, 0, 1 ) ); break;
            case ((int)DebugViewType::DenoisingLayer_DiffRadianceDenoised):     DebugPixel( outDebugPixelPos, float4( sqrt(diffRadiance.rgb), 1 ) ); break;
            case ((int)DebugViewType::DenoisingLayer_SpecRadianceDenoised):     DebugPixel( outDebugPixelPos, float4( sqrt(specRadiance.rgb), 1 ) ); break;
            case ((int)DebugViewType::DenoisingLayer_CombinedRadianceDenoised): DebugPixel( outDebugPixelPos, float4( sqrt(diffRadiance.rgb + specRadiance.rgb), 1 ) ); break;
            case ((int)DebugViewType::DenoisingLayer_ViewZ):                    DebugPixel( outDebugPixelPos, float4( viewZ/10, frac(viewZ), 0, 1 ) ); break;
            case ((int)DebugViewType::DenoisingLayer_Throughput):               DebugPixel( outDebugPixelPos, float4( throughput, 1 ) ); break;
            case ((int)DebugViewType::DenoisingLayer_DenoiserValidation):       
                if( validation.a > 0 ) 
                    DebugPixel( pixelPos, float4( validation.rgb, 1 ) ); 
                else
                    DebugPixel( pixelPos, float4( sqrt(diffRadiance.rgb + specRadiance.rgb), 1 ) );
                break;
            default: break;
            }
        }
    }
#endif // #if ENABLE_DEBUG_VIZUALISATIONS

    if (hasSurface)
        u_InputOutput[pixelPos.xy].xyz += max(0, (diffRadiance.rgb + specRadiance.rgb));
    //else
    //    u_InputOutput[pixelPos.xy].xyz = float3(1,0,0);
}
#endif

#if defined(NO_DENOISER_FINAL_MERGE)

#include "../Shaders/Bindings/ShaderResourceBindings.hlsli"
#include "../Shaders/PathTracerBridgeDonut.hlsli"

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    uint2 pixelPos = dispatchThreadID.xy;
    if (any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ))
        return;

    const Ray cameraRay = Bridge::computeCameraRay( pixelPos );
    DenoisingLayersContext denoisingLayers = DenoisingLayersContext::make(u_DenoisingLayersHeader, u_DenoisingLayersBuffer, u_StableRadiance, g_Const.ptConsts);

    u_OutputColor[pixelPos] = float4(denoisingLayers.GetAllRadiance(pixelPos), 1.0);
}
#endif // NO_DENOISER_FINAL_MERGE

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