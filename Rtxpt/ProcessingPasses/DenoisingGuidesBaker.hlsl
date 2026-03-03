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

#define NON_PATH_TRACING_PASS 1

//<<<<<<< HEAD
//#include "../Shaders/Bindings/ShaderResourceBindings.hlsli"
////#include "../Shaders/PathTracerBridgeDonut.hlsli"
////#include "../Shaders/PathTracer/PathTracer.hlsli"
////#include "../Shaders/Libraries/ShaderDebug/ShaderDebug.hlsli"
//=======
#include "../Shaders/SampleConstantBuffer.h"
#include <donut/shaders/binding_helpers.hlsli>

// Only the UAV bindings needed by this shader (avoids including ShaderResourceBindings.hlsli
// which declares its own VK_PUSH_CONSTANT, conflicting with g_denoisingConstants on Vulkan)
RWTexture2D<float>                      u_Depth                         : register(u6);
RWTexture2D<float>                      u_SpecularHitT                  : register(u7);
RWTexture2D<float>                      u_ScratchFloat1                 : register(u8);
//>>>>>>> remotes/origin/dev

VK_PUSH_CONSTANT ConstantBuffer<DenoisingGuidesBakerConstants>  g_denoisingConstants : register(b1);

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
    const int2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_denoisingConstants.RenderResolution.x, g_denoisingConstants.RenderResolution.y) ) )
        return;

    const bool isPing = g_denoisingConstants.Ping;

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
    const uint2 pixelPos = dispatchThreadID.xy;
    if( any(pixelPos >= uint2(g_denoisingConstants.RenderResolution.x, g_denoisingConstants.RenderResolution.y) ) )
        return;

    // TODO: move debugging here

    // if (g_denoisingConstants.DebugView == (int)DebugViewType::DenoiserGuide_SpecHitT)
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
    const uint2 halfResPos = dispatchThreadID.xy;
    const uint2 halfRes = uint2( (g_denoisingConstants.RenderResolution.x + 1) / 2, (g_denoisingConstants.RenderResolution.y + 1) / 2 );
    if( any(halfResPos >= halfRes) )
        return;

    StablePlanesContext stablePlanes = StablePlanesContext::make(u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);

    float exponentialFalloffK = saturate(0.05);

    float2 screenSpaceMotion = u_MotionVectors[halfResPos*2 + (halfResPos % 2)].xy;
    uint2 historicHalfResPos = int2(float2(halfResPos*2) + screenSpaceMotion.xy + 0.5.xx) / 2;
    float4 outVal = clamp( u_DenoisingAvgLayerRadiance[historicHalfResPos], 0, HLF_MAX) * (1-exponentialFalloffK);

    uint renderWidth, renderHeight;
    u_MotionVectors.GetDimensions(renderWidth, renderHeight);

    for( uint stablePlaneIndex = 0; stablePlaneIndex < g_Const.ptConsts.GetActiveStablePlaneCount(); stablePlaneIndex++ )
    {
        float avg = 0;
        float k = 1e-7;
        for( int x = 0; x < 2; x++ )
            for( int y = 0; y < 2; y++ )
            {
                int2 pixelPos = halfResPos*2 + int2(x,y);
                uint spBranchID = stablePlanes.GetBranchID(clamp(pixelPos, int2(0,0), int2(renderWidth-1, renderHeight-1)), stablePlaneIndex);
                if (spBranchID != cStablePlaneInvalidBranchID)
                {
                    StablePlane sp = stablePlanes.LoadStablePlane(pixelPos, stablePlaneIndex);
                    float rad = max( 1e-5, Average(sp.GetNoisyRadiance()) );
                    rad = min( rad, g_Const.ptConsts.preExposedGrayLuminance*2);    // clamp - this is a very poor denoiser so add some ff filter
                    avg += log(rad+1);
                    //avg += rad;
                    k += 1.0;
                }
            }
        avg = exp(avg/k)-1;
        //avg = avg/k;
        outVal[stablePlaneIndex] += Average(avg) * exponentialFalloffK;
    }
    outVal.w = outVal.x+outVal.y+outVal.z;
    u_DenoisingAvgLayerRadiance[halfResPos] = outVal;
#endif
}
#endif

#endif // __DENOISING_GUIDES_BAKER__