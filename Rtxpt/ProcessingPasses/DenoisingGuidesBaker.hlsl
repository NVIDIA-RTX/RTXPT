/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __DENOISING_GUIDES_BAKER__
#define __DENOISING_GUIDES_BAKER__

struct DenoisingGuidesBakerConstants
{
    uint2           RenderResolution;
    uint2           DisplayResolution;

    int             DebugView;
    uint            Ping;
    uint            _padding1;
    uint            _padding2;

    uint4           _padding3;
    uint4           _padding4;
};

#define DGB_1D_THREADGROUP_SIZE 256
#define DGB_2D_THREADGROUP_SIZE 8

#if !defined(__cplusplus)

#include "../Shaders/Libraries/ShaderDebug/ShaderDebug.hlsl"

#define NON_PATH_TRACING_PASS 1

#include "../Shaders/Bindings/ShaderResourceBindings.hlsli"
#include "../Shaders/PathTracerBridgeDonut.hlsli"
#include "../Shaders/PathTracer/PathTracer.hlsli"

// This pass is dispatched with the path tracer's global binding layout and shares its b1 push constant slot. Only one
// VK_PUSH_CONSTANT declaration is legal on Vulkan and ShaderResourceBindings.hlsli owns it (as g_MiniConst), so this file
// cannot declare a second one at b1 - which is why it used to hand-declare a handful of UAVs instead of including the
// bindings header at all. DenoisingGuidesBakerConstants is layout-compatible with SampleMiniConstants (see the
// static_assert in DenoisingGuidesBaker.cpp), so the host still fills the friendlier named struct and we unpack it back
// out of g_MiniConst here. Field offsets must stay in lockstep with the struct declared above.
DenoisingGuidesBakerConstants GetDenoisingGuidesBakerConstants()
{
    DenoisingGuidesBakerConstants ret;
    ret.RenderResolution    = g_MiniConst.params.xy;    // offset 0
    ret.DisplayResolution   = g_MiniConst.params.zw;    // offset 8
    ret.DebugView           = (int)g_MiniConst.params1.x;   // offset 16
    ret.Ping                = g_MiniConst.params1.y;        // offset 20
    ret._padding1           = g_MiniConst.params1.z;
    ret._padding2           = g_MiniConst.params1.w;
    ret._padding3           = g_MiniConst.params2;
    ret._padding4           = g_MiniConst.params3;
    return ret;
}

float SpecHitTNeighbourhood( RWTexture2D<float> texSrc, int2 pixelPos )
{
    float centerD = u_Depth[pixelPos];
    float prevHitT = max( 0, texSrc[pixelPos] );

    float minSpecHitT = 5e-2f;  // depends on the storage & computation precision - this seems to make sense with "normal" scene scales
    if (prevHitT < minSpecHitT)
        prevHitT = 0;

    int neigS = 2;
    uint width, height; texSrc.GetDimensions(width, height); 
    
    float vAvg = prevHitT;
    float sumW = prevHitT > 0;
    for( int x = -neigS; x <= +neigS; x++ )
        for( int y = -neigS; y <= +neigS; y++ )
        {
            if (x == 0 && y == 0)
                continue;
            int2 npixelPos = (int2)pixelPos + int2(x, y);
            if (npixelPos.x >= 0 && npixelPos.y >= 0 && npixelPos.x < width && npixelPos.y < height)
            {
                float v = min( texSrc[npixelPos], HLF_MAX );
                float d = max( 0, u_Depth[npixelPos] );

                const float depthThreshold = 0.025;

                float weight = v>0;
                weight *= abs(d-centerD) <= (d+centerD+1e-5f) * depthThreshold;

                if (weight>0)
                {
                    vAvg += v * weight;
                    sumW += weight;
                }
            }
        }

    if (sumW==0)
        return prevHitT;

    vAvg /= sumW;

    return (prevHitT<=0)?(vAvg):(min(prevHitT*1.5 + 0.5, vAvg));
}

[numthreads(DGB_2D_THREADGROUP_SIZE, DGB_2D_THREADGROUP_SIZE, 1)]
void DenoiseSpecHitT( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const DenoisingGuidesBakerConstants consts = GetDenoisingGuidesBakerConstants();

    const int2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(consts.RenderResolution.x, consts.RenderResolution.y) ) )
        return;

    const bool isPing = consts.Ping;

    #define MAIN_BUFFER     u_SpecularHitT
    #define SCRATCH_BUFFER  u_ScratchFloat1

    if (isPing)
        SCRATCH_BUFFER[pixelPos]    = SpecHitTNeighbourhood(MAIN_BUFFER, pixelPos);
    else
        MAIN_BUFFER[pixelPos]       = SpecHitTNeighbourhood(SCRATCH_BUFFER, pixelPos);
}

[numthreads(DGB_2D_THREADGROUP_SIZE, DGB_2D_THREADGROUP_SIZE, 1)]
void DebugViz( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const DenoisingGuidesBakerConstants consts = GetDenoisingGuidesBakerConstants();

    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(consts.RenderResolution.x, consts.RenderResolution.y) ) )
        return;

    // TODO: move debugging here

    // if (consts.DebugView == (int)DebugViewType::DenoiserGuide_SpecHitT)
    // {
    //     float specHitT = u_SpecularHitT[pixelPos];
    //     DebugPixel( pixelPos, float4( GradientHeatMap( specHitT / 50.0 ), 1 ) );
    // }
    // if ( (pixelPos.x/10+pixelPos.y/10)%2==0 )
    //     DebugPixel( pixelPos, float4( pixelPos.x%2, pixelPos.y%2, (pixelPos.x+pixelPos.y)%2, 1.0 ) );
}

[numthreads(DGB_2D_THREADGROUP_SIZE, DGB_2D_THREADGROUP_SIZE, 1)]
void ComputeAvgLayerRadiance( uint2 dispatchThreadID : SV_DispatchThreadID )
{
#if 0 // code below is experimental, it is not going to work in current configuration
    const DenoisingGuidesBakerConstants consts = GetDenoisingGuidesBakerConstants();
    const uint2 halfResPos = dispatchThreadID.xy;
    const uint2 halfRes = uint2( (consts.RenderResolution.x + 1) / 2, (consts.RenderResolution.y + 1) / 2 );
    if( any(halfResPos >= halfRes) )
        return;

    DenoisingLayersContext denoisingLayers = DenoisingLayersContext::make(u_DenoisingLayersHeader, u_DenoisingLayersBuffer, u_StableRadiance, g_Const.ptConsts);

    float exponentialFalloffK = saturate(0.05);

    float2 screenSpaceMotion = u_MotionVectors[halfResPos*2 + (halfResPos % 2)].xy;
    uint2 historicHalfResPos = int2(float2(halfResPos*2) + screenSpaceMotion.xy + 0.5.xx) / 2;
    float4 outVal = clamp( u_DenoisingAvgLayerRadiance[historicHalfResPos], 0, HLF_MAX) * (1-exponentialFalloffK);

    uint renderWidth, renderHeight;
    u_MotionVectors.GetDimensions(renderWidth, renderHeight);

    for( uint denoisingLayerIndex = 0; denoisingLayerIndex < g_Const.ptConsts.GetActiveDenoisingLayerCount(); denoisingLayerIndex++ )
    {
        float avg = 0;
        float k = 1e-7;
        for( int x = 0; x < 2; x++ )
            for( int y = 0; y < 2; y++ )
            {
                int2 pixelPos = halfResPos*2 + int2(x,y);
                uint spBranchID = denoisingLayers.GetBranchID(clamp(pixelPos, int2(0,0), int2(renderWidth-1, renderHeight-1)), denoisingLayerIndex);
                if (spBranchID != cDenoisingLayerInvalidBranchID)
                {
                    DenoisingLayer sp = denoisingLayers.LoadDenoisingLayer(pixelPos, denoisingLayerIndex);
                    float rad = max( 1e-5, Average(sp.GetNoisyRadiance()) );
                    rad = min( rad, g_Const.ptConsts.preExposedGrayLuminance*2);    // clamp - this is a very poor denoiser so add some ff filter
                    avg += log(rad+1);
                    //avg += rad;
                    k += 1.0;
                }
            }
        avg = exp(avg/k)-1;
        //avg = avg/k;
        outVal[denoisingLayerIndex] += Average(avg) * exponentialFalloffK;
    }
    outVal.w = outVal.x+outVal.y+outVal.z;
    u_DenoisingAvgLayerRadiance[halfResPos] = outVal;
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// DLSS-RR guide preparation - moved here from PostProcess.hlsl, which is where it used to live for historical reasons only.
// ---------------------------------------------------------------------------------------------------------------------

// DLSS-RR tolerates near-black albedo far better than NRD (which needs a much higher floor to keep its demodulation divide
// stable - see NRD_MIN_REFLECTANCE in PostProcess.hlsl), so this floor exists only to keep guides strictly positive.
#define DLSSRR_MIN_ALBEDO       0.0001
#if 0 // New variant that works with previous position
float2 ComputeSpecularMotionVector(
    float3 primaryHitPosWorld,
    float3 primaryRayDirWorld,            // Normalized direction from camera toward reflector.
    float3 primaryHitNormalWorld,         // Normalized reflector normal.
    float3 reflectionRayWorld,            // Current secondary hit - primary hit.
    float3 reflectionRayWorldPrevious,    // Previous secondary hit - primary hit.
    float4x4 worldToClipMatrix,
    float4x4 prevWorldToClipMatrix )
{
    float3 n = primaryHitNormalWorld;

    // Mirror a displacement vector through the reflector plane:
    //
    //     mirrored = v - 2 * n * dot(n, v)
    //
    // The normal orientation does not matter because negating n produces
    // the same reflected vector.
    float3 currentImageOffset = reflectionRayWorld - 2.0f * n * dot(n, reflectionRayWorld);
    float3 previousImageOffset = reflectionRayWorldPrevious - 2.0f * n * dot(n, reflectionRayWorldPrevious);
    float3 currentImageWorld = primaryHitPosWorld + currentImageOffset;
    float3 previousImageWorld = primaryHitPosWorld + previousImageOffset;

    // Optionally use the primary-ray representation for the current image
    // position. For a perfect planar reflection this should be equivalent:
    //
    // currentImageWorld =
    //     primaryHitPosWorld +
    //     primaryRayDirWorld * length(reflectionRayWorld);

    float4 currentClipPos = mul( worldToClipMatrix, float4(currentImageWorld, 1.0f));
    float4 previousClipPos = mul( prevWorldToClipMatrix, float4(previousImageWorld, 1.0f));

    float2 currentNdc = currentClipPos.xy / currentClipPos.w;
    float2 previousNdc = previousClipPos.xy / previousClipPos.w;

    // Kept consistent with your existing SL-specific convention:
    // returned value is previous-to-current in NDC.
    return currentNdc - previousNdc;
}
#endif


float2 ComputeSpecularMotionVector(
    float3 primaryHitPosWorld,
    float3 primaryHitWorldMotionVector,
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
    float4 prev_clip_pos = mul( float4( imageInWorld+primaryHitWorldMotionVector, 1.0f ), prevWorldToClipMatrix );
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

[numthreads(DGB_2D_THREADGROUP_SIZE, DGB_2D_THREADGROUP_SIZE, 1)]
void PrepareDLSSRRGuides( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) ) )
        return;

    DebugContext debug; debug.Init( g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack );
    const Ray cameraRay = Bridge::computeCameraRay( pixelPos );
    DenoisingLayersContext denoisingLayers = DenoisingLayersContext::make(u_DenoisingLayersHeader, u_DenoisingLayersBuffer, u_StableRadiance, g_Const.ptConsts);
    uint dominantDenoisingLayerIndex = denoisingLayers.LoadDominantIndex(pixelPos);

    float3 combinedRadiance = denoisingLayers.LoadStableRadiance(pixelPos);

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

    uint primaryLayer = dominantDenoisingLayerIndex;

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
            for( uint denoisingLayerIndex = 1; denoisingLayerIndex < g_Const.ptConsts.GetActiveDenoisingLayerCount(); denoisingLayerIndex++ )
            {
                uint spBranchID = denoisingLayers.GetBranchID(pixelPos, denoisingLayerIndex);
                if( spBranchID != cDenoisingLayerInvalidBranchID )
                {
                    uint spAddress = GenericTSPixelToAddress(pixelPos, denoisingLayerIndex, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);
                    float3 throughput; float3 motionVectors;
                    UnpackTwoFp32ToFp16(u_DenoisingLayersBuffer[spAddress].PackedThpAndMVs, throughput, motionVectors);
                    // TODO: use .GetThp()

                    float weight = saturate(Average(throughput));
                    thpWeights[denoisingLayerIndex] = weight;
                    thpWeights[0] = saturate( thpWeights[0] - weight );
                    spAvailable[denoisingLayerIndex] = 1;
                }
            }
            spWeights += thpWeights * kTW;
        }

        spWeights += float3( 1, 1, 1 ) * kNW;       // equalization

        spWeights[dominantDenoisingLayerIndex] += kDW;

        spWeights *= spAvailable;                   // explicitly remove unavailable layers

        // normalize
        spWeights /= spWeights.x+spWeights.y+spWeights.z; 

        if (spWeights.x>=max(spWeights.y, spWeights.z))
            primaryLayer = 0;
        else
            primaryLayer =  (spWeights.y > spWeights.z)?(1):(2);
    }

    // DebugPixel( pixelPos, float4(spWeights[0], spWeights[1], spWeights[2], 1) );

    for( uint denoisingLayerIndex = 0; denoisingLayerIndex < g_Const.ptConsts.GetActiveDenoisingLayerCount(); denoisingLayerIndex++ )
    {
        uint spBranchID = denoisingLayers.GetBranchID(pixelPos, denoisingLayerIndex);
        if( spBranchID != cDenoisingLayerInvalidBranchID )
        {
            DenoisingLayer sp = denoisingLayers.LoadDenoisingLayer(pixelPos, denoisingLayerIndex);

            bool hitSurface = isfinite(sp.SceneLength); 
            if( hitSurface ) // skip sky!
            {
                // hasSurface = true;

                combinedRadiance += sp.GetNoisyRadiance();

#if MIX_BY_THROUGHPUT
                float3 throughput; float3 dummy;
                UnpackTwoFp32ToFp16(sp.PackedThpAndMVs, throughput, dummy);
                // TODO: use GetThp()
                
                float weight = spWeights[denoisingLayerIndex];
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
                if( denoisingLayerIndex == dominantDenoisingLayerIndex )
                {
#if !ALLOW_MIX_NORMALS
                    guideNormals    = sp.GetNormal();
#endif
                }
#else
                if( denoisingLayerIndex == dominantDenoisingLayerIndex )
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

    // DLSS-RR's own reflectance floor - the stored BSDF estimates are no longer floored at write time (see DenoisingLayers.hlsli)
    diffAlbedo = max( diffAlbedo, 0 );
    specAlbedo = max( specAlbedo, 0 );

    // avoid settings both diff and spec albedo guides to 0
    const float minAlbedo = DLSSRR_MIN_ALBEDO;
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
        uint mirrorPlaneIndex = 0; //denoisingLayers.LoadDominantIndex(pixelPos); // <- not sure this actually works in non-primary hits
	    DenoisingLayer sp = denoisingLayers.LoadDenoisingLayer(pixelPos, mirrorPlaneIndex);
        float mirrorRoughness = sp.GetRoughness();

        float3 primaryHitPosWorld = sp.RayOrigin + sp.RayDir * sp.SceneLength;
        //DebugPixel( pixelPos, float4(frac(primaryHitPosWorld), 1));

        float3 primaryHitWorldMotionVector = sp.GetWorldMotionVector();
        //DebugPixel( pixelPos, float4(frac(primaryHitWorldMotionVector), 1));

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
                primaryHitPosWorld, primaryHitWorldMotionVector, primaryRayDirWorld, primaryHitNormalWorld, reflectionRayWorld, 
                g_Const.view.matWorldToClipNoOffset, g_Const.previousView.matWorldToClipNoOffset );

        //DebugPixel( pixelPos, float4(0.5 + specMotionVector, 0, 1));

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

#endif

#endif // __DENOISING_GUIDES_BAKER__