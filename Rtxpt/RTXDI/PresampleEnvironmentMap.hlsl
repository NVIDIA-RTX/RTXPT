/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "RtxdiApplicationBridge.hlsli"

#include <rtxdi/LightSampling/PresamplingFunctions.hlsli>

[numthreads(RTXDI_PRESAMPLING_GROUP_SIZE, 1, 1)]    // dispatch size will be [.tileSize / RTXDI_PRESAMPLING_GROUP_SIZE, .tileCount, 1]
void main(uint2 GlobalIndex : SV_DispatchThreadID) 
{
    RAB_RandomSamplerState rng = RAB_InitRandomSampler(GlobalIndex.xy, 0);

    RTXDI_RISBufferSegmentParameters envSegParams = g_RtxdiBridgeConst.environmentLightRISBufferSegmentParams;

    // this makes no sense since this pdf texture isn't used anymore but it's the way dispatch is setup so please bear with me for now
    const uint presamplingTotalCount = envSegParams.tileSize * RTXDI_PRESAMPLING_GROUP_SIZE * envSegParams.tileCount; // g_RtxdiBridgeConst.environmentPdfTextureSize.x * g_RtxdiBridgeConst.environmentPdfTextureSize.y;
    const uint presampledIndex = GlobalIndex.x + GlobalIndex.y * envSegParams.tileSize; 

    SampleGenerator sampleGenerator = SampleGenerator::make( SampleGeneratorVertexBase::make( uint2(0,0), 0, Bridge::getSampleIndex() * presamplingTotalCount + presampledIndex ) );
    EnvMapSampler envMapSampler =  Bridge::CreateEnvMapImportanceSampler( );

    float2 rnd = sampleNext2D(sampleGenerator);
    DistantLightSample result = envMapSampler.MIPDescentSample(rnd);

    float2 encodedDir = RAB_GetEnvironmentMapRandXYFromDir(result.Dir);
    uint packedEncodedDir = uint(saturate(encodedDir.x) * 0xffff) | (uint(saturate(encodedDir.y) * 0xffff) << 16);

    // see RTXDI_LightBrdfMisWeight where this and pdf from PolymorphicLight.hlsli EnvironmentLight::CalcSample are used - I think this is correct with regards to MIS
    result.Pdf = 1.0;

    float invSourcePdf = (result.Pdf > 0) ? (1.0 / result.Pdf) : 0;

    // Store the result
    uint risBufferPtr = envSegParams.bufferOffset + GlobalIndex.x + GlobalIndex.y * envSegParams.tileSize;
    RTXDI_RIS_BUFFER[risBufferPtr] = uint2(packedEncodedDir, asuint(invSourcePdf));

    // Note, original code used to use RTXDI built in importance sampling for this:
    // #if RTXDI_ENABLE_PRESAMPLING
    //     RTXDI_PresampleEnvironmentMap(
    //         rng,
    //         t_EnvironmentPdfTexture,
    //         g_RtxdiBridgeConst.environmentPdfTextureSize,
    //         GlobalIndex.y,
    //         GlobalIndex.x,
    //         g_RtxdiBridgeConst.environmentLightRISBufferSegmentParams);
    // #endif
}