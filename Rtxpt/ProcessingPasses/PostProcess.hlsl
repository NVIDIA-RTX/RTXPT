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

#if defined(STABLE_PLANES_DEBUG_VIZ)

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
    StablePlanesContext stablePlanes = StablePlanesContext::make(u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);

#if ENABLE_DEBUG_VIZUALISATIONS
    debug.StablePlanesDebugViz(pixelPos, stablePlanes);
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

float ComputeNeighbourDisocclusionRelaxation(const StablePlanesContext stablePlanes, const int2 pixelPos, const int2 imageSize, const uint stablePlaneIndex, const float3 rayDirC, const int2 offset)
{
    const float kEdge = 0.02;

    uint2 pixelPosN = clamp( int2(pixelPos)+offset, 0.xx, (imageSize-1.xx) );
    uint bidN = stablePlanes.GetBranchID(pixelPosN, stablePlaneIndex);
    if( bidN == cStablePlaneInvalidBranchID )
        return kEdge;
    uint spAddressN = stablePlanes.PixelToAddress( pixelPosN, stablePlaneIndex ); 
    float3 rayDirN = stablePlanes.StablePlanesUAV[spAddressN].GetNormal();
    return 1-dot(rayDirC, rayDirN);
}

float ComputeDisocclusionRelaxation(const StablePlanesContext stablePlanes, const uint2 pixelPos, const uint stablePlaneIndex, const uint spBranchID, const StablePlane sp)
{
    float disocclusionRelax = 0;

    const int2 imageSize = int2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight);
    const float3 rayDirC = sp.GetNormal();

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

float2 ComputeSpecularMotionVector(
    float3 primaryHitPosWorld,
    float3 primaryRayDirWorld,    // Ie. normalized view vector.
    float3 primaryHitNormalWorld, // Assumed to be normalized.
    float3 reflectionRayWorld, // Secondary hit position - primary hit position.
    float4x4 worldToClipMatrix,
    float4x4 prevWorldToClipMatrix )
{
    // Assuming a planar reflector, compute the image of reflection ray hit
    // point (ie. mirror it behind the plane), and transform the image to
    // previous frame normalized device coordinates. Note that this only takes
    // camera motion, not object motion into account, and is not accurate for
    // curved reflectors.
    float3 principalAxis = primaryHitNormalWorld;
    float zObj = dot( principalAxis, reflectionRayWorld );
    if ( zObj < 0.0f ) {
        principalAxis = -principalAxis;
        zObj = -zObj;
    }
    zObj = max( zObj, 1e-5f );

    // Constuct an orthonormal basis for the reflector plane.
    // From Duff et al "Building an Orthonormal Basis, Revisited"
    // http://jcgt.org/published/0006/01/01/paper-lowres.pdf
    // Improved from the original paper by Frisvad "Building an orthonormal
    // basis from a 3D unit vector without normalization"
    // http://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
    float3 n = principalAxis;
    float s = n.z < 0.0f ? -1.0f : 1.0f;
    float a = -1.0f / ( s + n.z );
    float b = n.x * n.y * a;
    float3 xAxis = float3( 1.0f + s * n.x * n.x * a, s * b, -s * n.x );
    float3 yAxis = float3( b, s + n.y * n.y * a, -n.y );

    // Coordinates of the reflection hit point in the reflector basis.
    float xObj = dot( xAxis, reflectionRayWorld );
    float yObj = dot( yAxis, reflectionRayWorld );
    float3 imagePosInReflectorSpace = float3( xObj, yObj, -zObj ); // Mirror z.

    // Image position in world space.
    float3 imageInWorld = primaryHitPosWorld +
        primaryRayDirWorld * length( imagePosInReflectorSpace );

    // Transform to previous frame normalized device coordinates (NDC).
    float4 prev_clip_pos = mul( float4( imageInWorld, 1.0f ), prevWorldToClipMatrix );
    float2 prev_ndc = float2( prev_clip_pos.x / prev_clip_pos.w,
                              prev_clip_pos.y / prev_clip_pos.w );

    // Transform to current frame normalized device coordinates (NDC).
    float4 curr_clip_pos = mul( float4( imageInWorld, 1.0f ), worldToClipMatrix );
    float2 curr_ndc = float2( curr_clip_pos.x / curr_clip_pos.w,
                              curr_clip_pos.y / curr_clip_pos.w );

    // By DLSS convention, mvecs point from current frame to previous frame.
    float2 specularVelocity = prev_ndc - curr_ndc;

#if 0 // original code
    // Scale from [-1,1] to [0,1].
    specularVelocity *= 0.5f;
    // Flip y if y-axis of clip space and screen space point to different
    // directions (game specific).
    specularVelocity.y = -specularVelocity.y;
#else
    specularVelocity = (specularVelocity) * g_Const.view.clipToWindowScale;
#endif
    return specularVelocity;
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

#if defined(DENOISER_DLSS_RR)

float3 SampleAvgLayerRadiance(uint2 pixelPos)
{
#if 0 // no filter
    uint2 halfResPos = pixelPos / 2;
    float3 ret = u_DenoisingAvgLayerRadiance[halfResPos].rgb;
#else // 3x3 filter
    uint halfWidth, halfHeight;
    u_DenoisingAvgLayerRadiance.GetDimensions(halfWidth, halfHeight);
    int2 halfResPos = pixelPos / 2;
    float3 ret = 0;
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++)
        {
            int2 samplePos = clamp(halfResPos + int2(x, y), int2(0, 0), int2(halfWidth - 1, halfHeight - 1));
            ret += u_DenoisingAvgLayerRadiance[samplePos].rgb;
        }
    ret /= 9.0;
#endif
    return ret;
}

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;

    DebugContext debug; debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack );
    const Ray cameraRay = Bridge::computeCameraRay( pixelPos );
    StablePlanesContext stablePlanes = StablePlanesContext::make(u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);
    uint dominantStablePlaneIndex = stablePlanes.LoadDominantIndex(pixelPos);

    float3 combinedRadiance = stablePlanes.LoadStableRadiance(pixelPos);

#define MIX_STABLE_RADIANCE 1 // stable radiance comes from emitters including sky; it is good to have it as guidance
#define MIX_BY_THROUGHPUT   1 // mix guide buffers from layers based on layer throughputs
#define ALLOW_MIX_NORMALS   1

#if MIX_STABLE_RADIANCE
    float3 stableAlbedo = sqrt(ReinhardMax(combinedRadiance));
    float stableAlbedoAvg = Average(stableAlbedo);
#endif

    float3 guideNormals = float3(0, 0, 1e-6);
    float3 diffAlbedo = float3(0.0, 0.0, 0.0);
    float3 specAlbedo = float3(0.0, 0.0, 0.0);
    float  roughness  = 0.0;

    uint primaryLayer = dominantStablePlaneIndex;

    float3 spWeights = float3(0,0,0);
    {
        // disabled, experimental, see below: const float kRW = 1.0;  // radiance weight
        const float kTW = 0.2;  // throughput weight
        const float kNW = 0.01; // equalization weight
        const float kDW = 0.05;  // user-selected dominant SP weight

        #if 0
        {
            float3 radianceWeight = SampleAvgLayerRadiance(pixelPos).xyz;
            float avgTotal = radianceWeight.x + radianceWeight.y + radianceWeight.z + 1e-5;
            radianceWeight = saturate(radianceWeight/avgTotal);
            spWeights = radianceWeight * kRW;
        }
        #endif
        
        float3 spAvailable = float3(1,0,0);
        {
            float3 thpWeights = { 1, 0, 0 };
            for( uint stablePlaneIndex = 1; stablePlaneIndex < g_Const.ptConsts.GetActiveStablePlaneCount(); stablePlaneIndex++ )
            {
                uint spBranchID = stablePlanes.GetBranchID(pixelPos, stablePlaneIndex);
                if( spBranchID != cStablePlaneInvalidBranchID )
                {
                    uint spAddress = GenericTSPixelToAddress(pixelPos, stablePlaneIndex, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);
                    float3 throughput; float3 motionVectors;
                    UnpackTwoFp32ToFp16(u_StablePlanesBuffer[spAddress].PackedThpAndMVs, throughput, motionVectors);

                    float weight = saturate(Average(throughput));
                    thpWeights[stablePlaneIndex] = weight;
                    thpWeights[0] = saturate( thpWeights[0] - weight );
                    spAvailable[stablePlaneIndex] = 1;
                }
            }
            spWeights += thpWeights * kTW;
        }

        spWeights += float3( 1, 1, 1 ) * kNW;       // equalization

        spWeights[dominantStablePlaneIndex] += kDW;

        spWeights *= spAvailable;                   // explicitly remove unavailable layers

        // normalize
        spWeights /= spWeights.x+spWeights.y+spWeights.z; 

        if (spWeights.x>=max(spWeights.y, spWeights.y))
            primaryLayer = 0;
        else
            primaryLayer =  (spWeights.y > spWeights.z)?(1):(2);
    }

    // DebugPixel( pixelPos, float4(spWeights[0], spWeights[1], spWeights[2], 1) );

    for( uint stablePlaneIndex = 0; stablePlaneIndex < g_Const.ptConsts.GetActiveStablePlaneCount(); stablePlaneIndex++ )
    {
        uint spBranchID = stablePlanes.GetBranchID(pixelPos, stablePlaneIndex);
        if( spBranchID != cStablePlaneInvalidBranchID )
        {
            StablePlane sp = stablePlanes.LoadStablePlane(pixelPos, stablePlaneIndex);

            bool hitSurface = isfinite(sp.SceneLength); 
            if( hitSurface ) // skip sky!
            {
                // hasSurface = true;

                combinedRadiance += sp.GetNoisyRadiance();

#if MIX_BY_THROUGHPUT
                float3 throughput; float3 motionVectors;
                UnpackTwoFp32ToFp16(sp.PackedThpAndMVs, throughput, motionVectors);
                
                float weight = spWeights[stablePlaneIndex];
                if (weight>1e-6)
                {
                    float3 diffBSDFEstimate, specBSDFEstimate;
                    UnpackTwoFp32ToFp16(sp.DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate);

#if ALLOW_MIX_NORMALS
                    guideNormals    += weight * sp.GetNormal();
#endif
                    roughness       += weight * sp.GetRoughness();
                    diffAlbedo      += weight * diffBSDFEstimate;
                    specAlbedo      += weight * specBSDFEstimate;
                }
                if( stablePlaneIndex == dominantStablePlaneIndex )
                {
#if !ALLOW_MIX_NORMALS
                    guideNormals    = sp.GetNormal();
#endif
                }
#else
                if( stablePlaneIndex == dominantStablePlaneIndex )
                {
                    float3 diffBSDFEstimate, specBSDFEstimate;
                    UnpackTwoFp32ToFp16(sp.DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate);

                    guideNormals = sp.GetNormal();
                    roughness = sp.GetRoughness();
                    diffAlbedo = diffBSDFEstimate;
                    specAlbedo = specBSDFEstimate;
                }
#endif
            }
        }
    }

#if MIX_STABLE_RADIANCE 
    float3 stableAlbedoGreyMix = lerp(stableAlbedo, 0.5.xxx, 0.2);  // make sure guidance is never 0
    diffAlbedo = lerp( diffAlbedo, stableAlbedoGreyMix, stableAlbedoAvg / (Average(diffAlbedo)+sqrt(stableAlbedoAvg)+1e-7) );
    //specAlbedo = lerp( specAlbedo, stableAlbedoGreyMix, stableAlbedoAvg / (Average(specAlbedo)+sqrt(stableAlbedoAvg)+1e-7) );
#endif

    // must be in sane range
    float guideNormalsLength = length(guideNormals);
    if (guideNormalsLength < 1e-5)
        guideNormals = float3(0, 0, 1);
    else
        guideNormals /= guideNormalsLength;

    // avoid settings both diff and spec albedo guides to 0
    const float minAlbedo = 0.05;
    if (Average(diffAlbedo+specAlbedo) < minAlbedo )
        diffAlbedo += minAlbedo;

    float maxRadiance = max3( combinedRadiance );
    if (maxRadiance > g_Const.ptConsts.DLSSRRBrightnessClampK )
        combinedRadiance *= g_Const.ptConsts.DLSSRRBrightnessClampK / maxRadiance;

    // we feed this as the main input into denoiser

    u_OutputColor[pixelPos] = float4(combinedRadiance, 1.0);

#if 0 // remove guide buffers (but not motion vectors and depth!)
    diffAlbedo = 0.5.xxx;
    specAlbedo = 0.5.xxx;
    roughness = 0.5;
    guideNormals = float3(0, 1, 0);
#endif

    u_RRDiffuseAlbedo[pixelPos]         = float4(diffAlbedo, 1);
    u_RRSpecAlbedo[pixelPos]            = float4(specAlbedo, 1);
    u_RRNormalsAndRoughness[pixelPos]   = float4(guideNormals, roughness);
    //u_RRTransparencyLayer[pixelPos]     = float4(0,0,0,0);

#if 1 // compute specular MVs from hitT
    {
        uint mirrorPlaneIndex = 0; //stablePlanes.LoadDominantIndex(pixelPos); // <- not sure this actually works in non-primary hits
	    StablePlane sp = stablePlanes.LoadStablePlane(pixelPos, mirrorPlaneIndex);
        float mirrorRoughness = sp.GetRoughness();

        float3 primaryHitPosWorld = sp.RayOrigin + sp.RayDir * sp.SceneLength;
        //DebugPixel( pixelPos, float4(frac(primaryHitPosWorld), 1));

        float3 primaryRayDirWorld = sp.RayDir;
        //DebugPixel( pixelPos, float4(DbgShowNormalSRGB(primaryRayDirWorld), 1));

        float3 primaryHitNormalWorld = sp.GetNormal();
        //DebugPixel( pixelPos, float4(DbgShowNormalSRGB(primaryHitNormalWorld), 1));

        float3 reflectionRayWorld = reflect( primaryRayDirWorld, primaryHitNormalWorld );
        //DebugPixel( pixelPos, float4(DbgShowNormalSRGB(reflectionRayWorld), 1));

        float specHitT = u_SpecularHitT[pixelPos];
        //DebugPixel( pixelPos, float4(frac(specHitT.xxx), 1));

        reflectionRayWorld *= specHitT;
        //DebugPixel( pixelPos, float4(DbgShowNormalSRGB(reflectionRayWorld), 1));

        float2 specMotionVector = u_MotionVectors[pixelPos].xy; // or should we default to zero?
        if (specHitT > 1e-3 && roughness < PathTracer::kSpecularRoughnessThreshold ) // have to be careful with these, they're pretty noisy otherwise
            specMotionVector = ComputeSpecularMotionVector( 
                primaryHitPosWorld, primaryRayDirWorld, primaryHitNormalWorld, reflectionRayWorld, 
                g_Const.view.matWorldToClipNoOffset, g_Const.previousView.matWorldToClipNoOffset );
        // ^^ HAD A BUG HERE with clip to screen ^^

        // DebugPixel( pixelPos, float4(0.5 + specMotionVector, 0, 1));

        u_RRSpecMotionVectors[pixelPos] = specMotionVector;
    }

#else // we default to zero - or should we just copy main MVs?
    u_RRSpecMotionVectors[pixelPos] = u_MotionVectors[pixelPos].xy;
#endif

    #if 0// debug draw motion vectors as lines
    if ( ((pixelPos.x % 20) == 0) && ((pixelPos.y % 20) == 0) )
        DebugLine( float2(pixelPos), float2(pixelPos)+u_MotionVectors[pixelPos].xy, float4(1, 0, 0, 1) );
    #endif

    //DebugPixel( pixelPos, float4(diffAlbedo.rgb, 1) );
    //DebugPixel( pixelPos, float4(specAlbedo.rgb, 1) );
    //DebugPixel( pixelPos, float4(DbgShowNormalSRGB(guideNormals), 1) );


#if ENABLE_DEBUG_VIZUALISATIONS
    switch (g_Const.debug.debugViewType)
    {
        case ((int)DebugViewType::DenoiserGuide_Depth):             DebugPixel( pixelPos, float4( u_Depth[pixelPos].xxx * 100.0, 1 ) ); break;
        case ((int)DebugViewType::DenoiserGuide_Roughness):         DebugPixel( pixelPos, float4( u_RRNormalsAndRoughness[pixelPos].www, 1.0 ) ); break;
        case ((int)DebugViewType::DenoiserGuide_Albedo):            DebugPixel( pixelPos, float4( u_RRDiffuseAlbedo[pixelPos].rgb, 1.0 ) ); break;
        case ((int)DebugViewType::DenoiserGuide_SpecAlbedo):        DebugPixel( pixelPos, float4( u_RRSpecAlbedo[pixelPos].rgb, 1.0 ) ); break;
        case ((int)DebugViewType::DenoiserGuide_Normal):            DebugPixel( pixelPos, float4( DbgShowNormalSRGB(u_RRNormalsAndRoughness[pixelPos].xyz), 1.0) ); break;
        case ((int)DebugViewType::DenoiserGuide_MotionVectors):     DebugPixel( pixelPos, float4( 0.5.xx + u_MotionVectors[pixelPos].xy * float2(0.2, 0.2), 0, 1 ) ); break;
        case ((int)DebugViewType::DenoiserGuide_SpecMotionVectors): DebugPixel( pixelPos, float4( 0.5.xx + u_RRSpecMotionVectors[pixelPos].xy * float2(0.2, 0.2), 0, 1 ) ); break;
        case ((int)DebugViewType::DenoiserGuide_SpecHitT):          DebugPixel( pixelPos, float4( GradientHeatMap( u_SpecularHitT[pixelPos] / 50.0 ), 1 ) ); break;
        case ((int)DebugViewType::DenoiserGuide_LayerWeights):      DebugPixel( pixelPos, float4( spWeights/*ReinhardMax(SampleAvgLayerRadiance(pixelPos))*/, 1 ) ); break;
        case ((int)DebugViewType::DenoiserGuide_PrimaryLayer):      DebugPixel( pixelPos, float4( primaryLayer == 0, primaryLayer == 1, primaryLayer == 2, 1 ) ); break;
        default: break;
    }

#endif // ENABLE_DEBUG_VIZUALISATIONS
}

#else // !defined(DENOISER_DLSS_RR)  <- !RR is NRD

[numthreads(NUM_COMPUTE_THREADS_PER_DIM, NUM_COMPUTE_THREADS_PER_DIM, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint stablePlaneIndex         = g_MiniConst.params[0];
    const bool initWithStableRadiance   = g_MiniConst.params[1];

    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;

    DebugContext debug; debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack );
    const Ray cameraRay = Bridge::computeCameraRay( pixelPos );
    StablePlanesContext stablePlanes = StablePlanesContext::make(u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);

    if (initWithStableRadiance)
        u_OutputColor[pixelPos] = float4( stablePlanes.LoadStableRadiance(pixelPos), 1 );

    bool hasSurface = false;
    uint spBranchID = stablePlanes.GetBranchID(pixelPos, stablePlaneIndex);
    if( spBranchID != cStablePlaneInvalidBranchID )
    {
        StablePlane sp = stablePlanes.LoadStablePlane(pixelPos, stablePlaneIndex);

        bool hitSurface = isfinite(sp.SceneLength); 
        if( hitSurface ) // skip sky!
        {
            hasSurface = true;
            float3 diffBSDFEstimate, specBSDFEstimate;
            UnpackTwoFp32ToFp16(sp.DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate);
            //diffBSDFEstimate = 1.xxx; specBSDFEstimate = 1.xxx;

            float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sp.SceneLength;
            float4 viewPos = mul(float4(/*bridgedData.shadingData.posW*/virtualWorldPos, 1), g_Const.view.matWorldToView);
            float virtualViewspaceZ = viewPos.z;

            float3 thp; float3 motionVectors;
            UnpackTwoFp32ToFp16(sp.PackedThpAndMVs, thp, motionVectors);

            // See if possible to get rid of these copies - or compress them better!
            u_DenoiserViewspaceZ[pixelPos]          = virtualViewspaceZ;
            u_DenoiserMotionVectors[pixelPos]       = float4(motionVectors, 0);

            const float kMinRoughness = 0.2;
            float finalRoughness = max( kMinRoughness, sp.GetRoughness() );

            float disocclusionRelax = 0.0;
            float aliasingDampen = 0.0;

            float specularSuppressionMul = 1.0; // this applies 
            if (stablePlaneIndex == 0 && g_Const.ptConsts.stablePlanesSuppressPrimaryIndirectSpecularK != 0.0 && g_Const.ptConsts.GetActiveStablePlaneCount() > 1 )
            {   // only apply suppression on sp 0, and only if more than 1 stable plane enabled, and only if other stable planes are in use (so they captured some of specular radiance)
                bool shouldSuppress = true;
                for (int i = 1; i < g_Const.ptConsts.GetActiveStablePlaneCount(); i++ )
                    shouldSuppress &= stablePlanes.GetBranchID(pixelPos, i) != cStablePlaneInvalidBranchID;
                // (optional, experimental, for future: also don't apply suppression if rough specular)
                float roughnessModifiedSuppression = g_Const.ptConsts.stablePlanesSuppressPrimaryIndirectSpecularK; // * saturate(1 - (finalRoughness - g_Const.ptConsts.stablePlanesMinRoughness)*5);
                specularSuppressionMul = shouldSuppress?saturate(1-roughnessModifiedSuppression):specularSuppressionMul;
            }

            int vertexIndex = StablePlanesVertexIndexFromBranchID( spBranchID );
            if (vertexIndex > 1)
                disocclusionRelax = ComputeDisocclusionRelaxation(stablePlanes, pixelPos, stablePlaneIndex, spBranchID, sp);
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
            uint dominantStablePlaneIndex = stablePlanes.LoadDominantIndex(pixelPos);
            if (dominantStablePlaneIndex == stablePlaneIndex)
            {
                specHitT = u_SpecularHitT[pixelPos];
                // DebugPixel( pixelPos, float4( specHitT / 100.0, 1 ) );
            }
#endif


    #if USE_RELAX
            u_DenoiserDiffRadianceHitDist[pixelPos] = RELAX_FrontEnd_PackRadianceAndHitDist( denoiserDiffRadiance.xyz, 0, true );
            u_DenoiserSpecRadianceHitDist[pixelPos] = RELAX_FrontEnd_PackRadianceAndHitDist( denoiserSpecRadiance.xyz, specHitT, true );
    #else
            float4 hitParams = g_Const.denoisingHitParamConsts;
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

#include <donut/shaders/binding_helpers.hlsli>
#include "../Shaders/SampleConstantBuffer.h"
#include "../NRD/DenoiserNRD.hlsli"
#include "../Shaders/PathTracer/StablePlanes.hlsli"

ConstantBuffer<SampleConstants>         g_Const             : register(b0);
VK_PUSH_CONSTANT ConstantBuffer<SampleMiniConstants>     g_MiniConst         : register(b1);

RWTexture2D<float4>     u_InputOutput                           : register(u0);
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

#if ENABLE_DEBUG_VIZUALISATIONS
    if (g_Const.debug.debugViewType >= (int)DebugViewType::StablePlane_RelaxedDisocclusion && g_Const.debug.debugViewType <= ((int)DebugViewType::StablePlane_DenoiserValidation))
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
            if (quadPlaneIndex >= g_Const.ptConsts.GetActiveStablePlaneCount())
                DebugPixel( pixelPos, float4( ((pixelPos.x+pixelPos.y)%2).xxx, 1 ) );
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
            case ((int)DebugViewType::StablePlane_RelaxedDisocclusion):      DebugPixel( outDebugPixelPos, float4( sqrt(relaxedDisocclusion), 0, 0, 1 ) ); break;
            case ((int)DebugViewType::StablePlane_DiffRadianceDenoised):     DebugPixel( outDebugPixelPos, float4( sqrt(diffRadiance.rgb), 1 ) ); break;
            case ((int)DebugViewType::StablePlane_SpecRadianceDenoised):     DebugPixel( outDebugPixelPos, float4( sqrt(specRadiance.rgb), 1 ) ); break;
            case ((int)DebugViewType::StablePlane_CombinedRadianceDenoised): DebugPixel( outDebugPixelPos, float4( sqrt(diffRadiance.rgb + specRadiance.rgb), 1 ) ); break;
            case ((int)DebugViewType::StablePlane_ViewZ):                    DebugPixel( outDebugPixelPos, float4( viewZ/10, frac(viewZ), 0, 1 ) ); break;
            case ((int)DebugViewType::StablePlane_Throughput):               DebugPixel( outDebugPixelPos, float4( throughput, 1 ) ); break;
            case ((int)DebugViewType::StablePlane_DenoiserValidation):       
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
    StablePlanesContext stablePlanes = StablePlanesContext::make(u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);

    u_OutputColor[pixelPos] = float4(stablePlanes.GetAllRadiance(pixelPos), 1.0);
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