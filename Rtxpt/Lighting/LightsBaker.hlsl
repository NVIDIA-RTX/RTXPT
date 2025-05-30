/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __LIGHTS_BAKER_HLSL__
#define __LIGHTS_BAKER_HLSL__

#define LLB_NUM_COMPUTE_THREADS         128
#define LLB_LOCAL_BLOCK_SIZE            32

#define LLB_SCRATCH_BUFFER_SIZE         (48*1024*1024)

#define RTXPT_LIGHTING_CPJ_BLOCKSIZE    1024

#define LLB_MAX_TRIANGLES_PER_TASK      32
#define LLB_MAX_PROC_TASKS              (RTXPT_LIGHTING_MAX_LIGHTS / LLB_MAX_TRIANGLES_PER_TASK * 2)
struct EmissiveTrianglesProcTask
{
    uint InstanceIndex; 
    uint GeometryIndex;
    uint TriangleIndexFrom;
    uint TriangleIndexTo;
    uint DestinationBufferOffset;
    uint HistoricBufferOffset;
    uint EmissiveLightMappingOffset;
    uint Padding0;
};

#define LLB_MAX_PROXIES_PER_TASK        32
#define LLB_MAX_PROXY_PROC_TASKS        (RTXPT_LIGHTING_MAX_LIGHTS+(RTXPT_LIGHTING_MAX_SAMPLING_PROXIES+LLB_MAX_PROXIES_PER_TASK-1) / LLB_MAX_PROXIES_PER_TASK)
struct SamplingProxyBuildProcTask
{
    uint LightIndex;            // <- index into u_lightsBuffer
    uint ProxyIndexBase;        // useful for figuring out sampling proxy index within its own proxies
    uint FillProxyIndexFrom;    // this task needs to fill from this index                                  
    uint FillProxyIndexTo;      // this task needs to fill to this index                                    
};

#if defined(__cplusplus)
static_assert( sizeof(EmissiveTrianglesProcTask) * LLB_MAX_PROC_TASKS <= LLB_SCRATCH_BUFFER_SIZE ); // does it fit
static_assert( sizeof(SamplingProxyBuildProcTask) * LLB_MAX_PROXY_PROC_TASKS <= LLB_SCRATCH_BUFFER_SIZE ); // does it fit
static_assert( (RTXPT_LIGHTING_MAX_LIGHTS / LLB_MAX_TRIANGLES_PER_TASK * 2) <= LLB_MAX_PROC_TASKS );
#endif


#if !defined(__cplusplus)

#pragma pack_matrix(row_major)

#define LLB_ENABLE_VALIDATION 0

#define NON_PATH_TRACING_PASS 1

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>

#include "../Shaders/SubInstanceData.h"
#include "../Shaders/PathTracer/Materials/MaterialPT.h"

#include "../Shaders/ShaderDebug.hlsli"
#include "../Shaders/PathTracer/Utils/Math/MathHelpers.hlsli"
#include "../Shaders/PathTracer/Lighting/LightingTypes.h"
#include "../Shaders/PathTracer/Lighting/LightingConfig.h"
#include "../Shaders/PathTracer/Lighting/PolymorphicLight.hlsli"
#include "../Shaders/PathTracer/Lighting/LightingAlgorithms.hlsli"
#include "../Shaders/PathTracer/Utils/NoiseAndSequences.hlsli"
#include "../Shaders/PathTracer/Utils/SampleGenerators.hlsli"

ConstantBuffer<LightsBakerConstants>        g_const                         : register(b0);

RWStructuredBuffer<LightingControlData>     u_controlBuffer                 : register(u0);

RWStructuredBuffer<PolymorphicLightInfo>    u_lightsBuffer                  : register(u1);
RWStructuredBuffer<PolymorphicLightInfoEx>  u_lightsExBuffer                : register(u2);

RWByteAddressBuffer                         u_scratchBuffer                 : register(u3);
RWBuffer<uint>                              u_scratchList                   : register(u4);

RWBuffer<float>                             u_lightWeights                  : register(u5);
RWBuffer<uint>                              u_historyRemapCurrentToPast     : register(u6);
RWBuffer<uint>                              u_historyRemapPastToCurrent     : register(u7);
RWBuffer<uint>                              u_perLightProxyCounters         : register(u8);
RWBuffer<uint>                              u_lightSamplingProxies          : register(u9);
RWTexture2D<uint>                           u_envLightLookupMap             : register(u10);

RWTexture2D<uint2>                          u_feedbackReservoirBuffer       : register(u12);    //  this is the raw "reservoirs" coming from previous frame - historic light indices that need remapping!
RWTexture2D<uint>                           u_processedFeedbackBuffer       : register(u13);    //  this is the partially processed feedback data, with old light indices matched to new and holes filled but NOT yet reprojected
RWTexture2D<uint2>                          u_reprojectedFeedbackBuffer     : register(u14);    //  this is u_reprojectedFeedbackBuffer but reprojected, plus .y contains historic samples that bypass NEE
RWTexture2D<uint>                           u_reprojectedLRFeedbackBuffer   : register(u15);
RWTexture3D<uint>                           u_narrowSamplingBuffer          : register(u16);

#if RTXPT_LIGHTING_NEEAT_ENABLE_RESERVOIR_HISTORY
RWTexture2D<uint2>                          u_feedbackReservoirBufferScratch: register(u17);
#endif

Texture2D<float>                            t_depthBuffer                   : register(t10);     //< unused, we have very simple reprojection that relies on motion vectors only, errors are tolerated
Texture2D<float3>                           t_motionVectors                 : register(t11);
Texture2D<float4>                           t_envRadianceAndImportanceMap   : register(t12);

StructuredBuffer<SubInstanceData>   t_SubInstanceData                       : register(t1);
StructuredBuffer<InstanceData>      t_InstanceData                          : register(t2);
StructuredBuffer<GeometryData>      t_GeometryData                          : register(t3);
StructuredBuffer<PTMaterialData>    t_PTMaterialData                        : register(t5);

VK_BINDING(0, 1) ByteAddressBuffer t_BindlessBuffers[]                  : register(t0, space1);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[]                         : register(t0, space2);

SamplerState                                    s_point                 : register(s0);
SamplerState                                    s_linear                : register(s1);
SamplerState                                    s_materialSampler       : register(s2);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// debugging viz
void DebugDrawLightSphere(const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor);
void DebugDrawLightPoint(const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor);
void DebugDrawLightTriangle(const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor);
void DebugDrawLightDirectional(const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor);
void DebugDrawLightEnvironment(const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor);
void DebugDrawLightEnvironmentQuad(const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor);
void DebugDrawLight(const PolymorphicLightInfoFull lightInfo, float alpha);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void ResetPastToCurrentHistory( uint lightIndex : SV_DispatchThreadID )
{
    const LightingControlData controlInfo = u_controlBuffer[0];
    uint totalCount = max(controlInfo.HistoricTotalLightCount, g_const.TotalLightCount);
    if( lightIndex >= totalCount )
        return;
    u_historyRemapPastToCurrent[lightIndex] = RTXPT_INVALID_LIGHT_INDEX;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ENVMAP SECTION
///
float4 EnvironmentComputeRadianceAndWeight( uint dim, uint x, uint y )
{
    int dimLog2 = firstbithigh(dim); //(uint)log2( (float)dim );
    uint mipLevel = g_const.EnvMapImportanceMapMIPCount - dimLog2 - 1;
    float areaMul = 1u << (mipLevel*2); //pow(4.0,mipLevel);
    float4 value = t_envRadianceAndImportanceMap.Load( int3( x, y, mipLevel ) ).rgba;
    float weight = areaMul * max( 0, value.a * average(g_const.EnvMapParams.ColorMultiplier) * g_const.DistantVsLocalRelativeImportance );
    return float4( value.rgb * g_const.EnvMapParams.ColorMultiplier, weight );
};
//
#define PACK_20F_12UI(_value, _index)   ((min(uint(FastSqrt(_value)*100+0.5), 0x000FFFFF) << 12) | uint(_index))   // 20 bits for value, 12 bits for index (not overflow clamped)
#define UNPACK_20F(_packed)             (sq(float(uint(_packed) >> 12) / 100.0))
#define UNPACK_12UI(_packed)            (_packed&0xFFF)
#define MIN_PACK_WEIGHT                 (sq(1.0 / 100.0))
uint EnvironmentComputeWeightForQTBuild( uint dim, uint x, uint y, uint lightIndex, uint depthLimit )
{
    int dimLog2 = firstbithigh(dim);//(uint)log2( (float)dim );
    uint mipLevel = g_const.EnvMapImportanceMapMIPCount - dimLog2 - 1;
    float areaMul = 1u << (mipLevel*2); //pow(4.0,mipLevel);
    float radiance = t_envRadianceAndImportanceMap.Load( int3( x, y, mipLevel ) ).w;
    // if (depthLimit!=0)  // tweak subdivision for base layer only
    //     areaMul = pow(areaMul, 1.0);  
    float ret = areaMul * radiance;
    
    // this is purely so we can pack to non-zero
    ret = max(MIN_PACK_WEIGHT * mipLevel, ret);

    // this is our termination criteria: mark final node with (near-)zero weight; this prevents them from being subdivided.
    ret *= (mipLevel > depthLimit);  
    
    return PACK_20F_12UI(ret, lightIndex);
}
//
float3 EnvironmentQuadLight::ToWorld(float3 localDir)  // Transform direction from local to world space.
{
    return mul(localDir, (float3x3)g_const.EnvMapParams.Transform);
}
//
float3 EnvironmentQuadLight::ToLocal(float3 worldDir)  // Transform direction from world to local space.
{
    return mul(worldDir, (float3x3)g_const.EnvMapParams.InvTransform);
}
////
//float3 EnvironmentQuadLight::SampleLocalSpace(float3 localDir)
//{
//    return float3(0,0,0); // not needed here - in case needed for debugging - add!
//}
//
uint EQTNodePack( uint dim, uint x, uint y )
{ 
    uint dimLog2 = (uint)log2( (float)dim );
    return (dimLog2<<(uint)28) | (x<<(uint)14) | (y);
}
//
void EQTNodeUnpack( const uint packed, out uint dim, out uint x, out uint y )
{ 
    uint dimLog2    = packed >> (uint)28;
    dim             = ((uint)1U<<dimLog2);
    x               = (packed>>(uint)14) & (uint)0x3FFF;
    y               = packed & (uint)0x3FFF;
}
//
EnvironmentQuadLight LoadEnvironmentQuadLight( uint lightIndex )
{
    EnvironmentQuadLight light; light.NodeDim = 0; light.NodeX = 0; light.NodeY = 0;

    PolymorphicLightInfoFull packedLightInfo = PolymorphicLightInfoFull::make(u_lightsBuffer[lightIndex]);
#if LLB_ENABLE_VALIDATION
    if( PolymorphicLight::DecodeType(packedLightInfo) != PolymorphicLightType::kEnvironmentQuad )
    {
        DebugPrint("Error in LoadEnvironmentQuadLight({0})", lightIndex);
        // DebugPrint("", packedLightInfo.Base.Center, packedLightInfo.Base.ColorTypeAndFlags, packedLightInfo.Base.Direction1, packedLightInfo.Base.Direction2, packedLightInfo.Base.Scalars, packedLightInfo.Base.LogRadiance);
    }
    else
#endif
        light = EnvironmentQuadLight::Create(packedLightInfo);
    return light;
}
//
[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]       // dispatch is (FEIS_TARGET_QUADTREE_NODE_COUNT, 1, 1)
void EnvLightsBackupPast( uint lightIndex : SV_DispatchThreadID )
{
    if( lightIndex >= RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT )
        return;

    const LightingControlData controlInfo = u_controlBuffer[0];
    uint value = 0;
    if( controlInfo.LastFrameTemporalFeedbackAvailable )
    {
        EnvironmentQuadLight light = LoadEnvironmentQuadLight(lightIndex);
        value = EQTNodePack(light.NodeDim, light.NodeX, light.NodeY);
    }
    u_scratchList[lightIndex] = value;        // history is backed up in RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT
}
//
#define ENV_LIGHTS_BAKE_THREADS 128
#define SUBDIVISION_MAX_NODES max(RTXPT_LIGHTING_ENVMAP_QT_UNBOOSTED_NODE_COUNT, RTXPT_LIGHTING_ENVMAP_QT_BOOST_NODES_MULT)
groupshared uint    g_nodes[SUBDIVISION_MAX_NODES];
groupshared uint    g_nodePackedWeights[SUBDIVISION_MAX_NODES];
groupshared uint    g_findMaxPacked;
[numthreads(ENV_LIGHTS_BAKE_THREADS, 1, 1)] // note, Dispatch size is (1, 1, 1)
void EnvLightsSubdivideBase( uint groupThreadID : SV_GroupThreadId )
{
    const uint baseNodeCount = RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION*RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION;

    // Init base nodes
    for( int i = 0; i < (baseNodeCount+ENV_LIGHTS_BAKE_THREADS-1)/ENV_LIGHTS_BAKE_THREADS; i++ )
    {
        uint lightIndex = i * ENV_LIGHTS_BAKE_THREADS + groupThreadID;

        if( lightIndex < baseNodeCount )
        {
            uint nodeDim    = RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION;
            uint nodeX      = lightIndex / RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION;
            uint nodeY      = lightIndex % RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION;
            
            g_nodes[lightIndex]             = EQTNodePack(nodeDim, nodeX, nodeY);
            g_nodePackedWeights[lightIndex] = EnvironmentComputeWeightForQTBuild(nodeDim, nodeX, nodeY, lightIndex, RTXPT_LIGHTING_ENVMAP_QT_BOOST_SUBDIVISION_DPT);
        }
    }
    
    if( groupThreadID == 0 )
        g_findMaxPacked = 0;

    // Quad tree build 
    GroupMemoryBarrierWithGroupSync(); // g_nodes/g_nodeWeights were touched, have to sync
    uint nodeCount = baseNodeCount; // every thread keeps their node count
    for( int si = 0; si < RTXPT_LIGHTING_ENVMAP_QT_SUBDIVISIONS; si++ ) // we know exactly how many subdivisions we'll make
    {
        // uint nodeCount = baseNodeCount + si * 3; // we could also do this - makes no difference
        // find the max value
        const uint itemsPerThread = (nodeCount + ENV_LIGHTS_BAKE_THREADS - 1) / ENV_LIGHTS_BAKE_THREADS;
        uint indexFrom = groupThreadID * itemsPerThread;
        uint indexTo = min( indexFrom + itemsPerThread, nodeCount );
        uint localMax = (indexFrom < nodeCount)?(g_nodePackedWeights[indexFrom]):(0);
        for( uint index = indexFrom+1; index < indexTo; index++ )
            localMax = max( localMax, g_nodePackedWeights[index] );

        uint waveMax = WaveActiveMax(localMax);
        if ( WaveIsFirstLane() )
            InterlockedMax(g_findMaxPacked, waveMax);

        // make sure latest g_findMaxPacked is available to all threads
        GroupMemoryBarrierWithGroupSync();
        uint packed = g_findMaxPacked;
        int globalMaxIndex = UNPACK_12UI(packed);

        // if (packed == 0)
        //     DebugPrint("Shouldn't ever happen");

        uint nodeDim; uint nodeX; uint nodeY;
        EQTNodeUnpack( g_nodes[globalMaxIndex], nodeDim, nodeX, nodeY );

        GroupMemoryBarrierWithGroupSync(); // this is due to reading from g_nodes[] above, as we'll be modifying it
        
        if( groupThreadID == 0 )
            g_findMaxPacked = 0;

        // use 4 threads to handle splitting - better than serializing;
        if( groupThreadID < 4 )
        {
            nodeDim *= 2; // resolution of the layer - increases by 2 with every subdivision! confusingly, more subdivided (smaller) nodes have higher dim
            nodeX = nodeX*2+(groupThreadID%2);
            nodeY = nodeY*2+(groupThreadID/2);
            uint newNodeIndex = (groupThreadID==0)?(globalMaxIndex):(nodeCount+groupThreadID-1);  // reusing the existing node's storage in the first thread, allocating new for remaining 3

            g_nodes[newNodeIndex]         = EQTNodePack( nodeDim, nodeX, nodeY );
            g_nodePackedWeights[newNodeIndex] = EnvironmentComputeWeightForQTBuild(nodeDim, nodeX, nodeY, newNodeIndex, RTXPT_LIGHTING_ENVMAP_QT_BOOST_SUBDIVISION_DPT);
        }

        GroupMemoryBarrierWithGroupSync(); // since we've just modified g_nodes and g_nodePackedWeights, we must sync up
        nodeCount += 3; // we're always adding 4 new nodes, one in the place of the old one and 3 new ones, so update the count
    }

    if( nodeCount != RTXPT_LIGHTING_ENVMAP_QT_UNBOOSTED_NODE_COUNT )
        DebugPrint("Node number overflow/underflow");

    for( int i = 0; i < (RTXPT_LIGHTING_ENVMAP_QT_UNBOOSTED_NODE_COUNT+ENV_LIGHTS_BAKE_THREADS-1)/ENV_LIGHTS_BAKE_THREADS; i++ )
    {
        uint lightIndex = i * ENV_LIGHTS_BAKE_THREADS + groupThreadID;
        if (lightIndex < RTXPT_LIGHTING_ENVMAP_QT_UNBOOSTED_NODE_COUNT)
            u_scratchList[RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT+lightIndex*RTXPT_LIGHTING_ENVMAP_QT_BOOST_NODES_MULT] = g_nodes[lightIndex]; // spread out our "seed" nodes with RTXPT_LIGHTING_ENVMAP_QT_BOOST_NODES_MULT space between them
    }
}

[numthreads(ENV_LIGHTS_BAKE_THREADS, 1, 1)] // note, Dispatch size is (RTXPT_LIGHTING_ENVMAP_QT_UNBOOSTED_NODE_COUNT, 1, 1)
void EnvLightsSubdivideBoost( uint groupThreadID : SV_GroupThreadId, uint groupID : SV_GroupID )
{
    const uint baseNodeCount = 1;

    const uint globalNodeBaseIndex     = groupID * RTXPT_LIGHTING_ENVMAP_QT_BOOST_NODES_MULT;

    // Init base node
    uint baseNode = u_scratchList[RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT + globalNodeBaseIndex];
    uint nodeDim; uint nodeX; uint nodeY;
    EQTNodeUnpack( baseNode, nodeDim, nodeX, nodeY );

    g_nodes[0]             = baseNode; //EQTNodePack(nodeDim, nodeX, nodeY);
    g_nodePackedWeights[0] = EnvironmentComputeWeightForQTBuild(nodeDim, nodeX, nodeY, 0, 0);

    // DebugPrint("", groupID, nodeDim, nodeX, nodeY, g_nodePackedWeights[0]);
    
    if( groupThreadID == 0 )
        g_findMaxPacked = 0;

    // Quad tree build 
    GroupMemoryBarrierWithGroupSync(); // g_nodes/g_nodeWeights were touched, have to sync
    uint nodeCount = baseNodeCount; // every thread keeps their node count
    for( int si = 0; si < RTXPT_LIGHTING_ENVMAP_QT_BOOST_SUBDIVISION; si++ ) // we know exactly how many subdivisions we'll make
    {
        // find the max value
        const uint itemsPerThread = (nodeCount + ENV_LIGHTS_BAKE_THREADS - 1) / ENV_LIGHTS_BAKE_THREADS;
        uint indexFrom = groupThreadID * itemsPerThread;
        uint indexTo = min( indexFrom + itemsPerThread, nodeCount );
        uint localMax = (indexFrom < nodeCount)?(g_nodePackedWeights[indexFrom]):(0);
        for( uint index = indexFrom+1; index < indexTo; index++ )
            localMax = max( localMax, g_nodePackedWeights[index] );

        uint waveMax = WaveActiveMax(localMax);
        if ( WaveIsFirstLane() )
            InterlockedMax(g_findMaxPacked, waveMax);

        // make sure latest g_findMaxPacked is available to all threads
        GroupMemoryBarrierWithGroupSync();
        uint packed = g_findMaxPacked;
        int globalMaxIndex = UNPACK_12UI(packed);

        // if (packed == 0)
        //     DebugPrint("Shouldn't ever happen");

        uint nodeDim; uint nodeX; uint nodeY;
        EQTNodeUnpack( g_nodes[globalMaxIndex], nodeDim, nodeX, nodeY );

        GroupMemoryBarrierWithGroupSync(); // this is due to reading from g_nodes[] above, as we'll be modifying it
        
        if( groupThreadID == 0 )
            g_findMaxPacked = 0;

        // use 4 threads to handle splitting - better than serializing;
        if( groupThreadID < 4 )
        {
            nodeDim *= 2; // resolution of the layer - increases by 2 with every subdivision! confusingly, more subdivided (smaller) nodes have higher dim
            nodeX = nodeX*2+(groupThreadID%2);
            nodeY = nodeY*2+(groupThreadID/2);
            uint newNodeIndex = (groupThreadID==0)?(globalMaxIndex):(nodeCount+groupThreadID-1);  // reusing the existing node's storage in the first thread, allocating new for remaining 3

            g_nodes[newNodeIndex]         = EQTNodePack( nodeDim, nodeX, nodeY );
            g_nodePackedWeights[newNodeIndex] = EnvironmentComputeWeightForQTBuild(nodeDim, nodeX, nodeY, newNodeIndex, 0);
        }

        GroupMemoryBarrierWithGroupSync(); // since we've just modified g_nodes and g_nodePackedWeights, we must sync up
        nodeCount += 3; // we're always adding 4 new nodes, one in the place of the old one and 3 new ones, so update the count
    }

    if( nodeCount != RTXPT_LIGHTING_ENVMAP_QT_BOOST_NODES_MULT )
        DebugPrint("Node number overflow/underflow (boost)");

    for( int i = 0; i < (RTXPT_LIGHTING_ENVMAP_QT_BOOST_NODES_MULT+ENV_LIGHTS_BAKE_THREADS-1)/ENV_LIGHTS_BAKE_THREADS; i++ )
    {
        uint lightIndex = i * ENV_LIGHTS_BAKE_THREADS + groupThreadID;
        if (lightIndex < RTXPT_LIGHTING_ENVMAP_QT_BOOST_NODES_MULT)
        {
            // u_scratchList[RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT+globalNodeBaseIndex+lightIndex] = g_nodes[lightIndex];

            // bake in-place!
            uint outLightIndex = globalNodeBaseIndex+lightIndex;
            EnvironmentQuadLight envLight;
            EQTNodeUnpack( g_nodes[lightIndex], envLight.NodeDim, envLight.NodeX, envLight.NodeY );
    
            float4 radianceAndWeight = EnvironmentComputeRadianceAndWeight(envLight.NodeDim, envLight.NodeX, envLight.NodeY);
            envLight.Radiance = radianceAndWeight.rgb;
            envLight.Weight = radianceAndWeight.a;
            //DebugPrint("", envLight.Weight);

            uint uniqueID = Hash32CombineSimple( Hash32CombineSimple(Hash32(envLight.NodeX), Hash32(envLight.NodeY)), Hash32(envLight.NodeDim) );

            PolymorphicLightInfoFull lightFull = envLight.Store(uniqueID);
        #if 1       // figure out our "world location" and patch it into the lightInfo; used for debugging only - feel free to remove in production code!
            float2 subTexelPos = float2( ((float)envLight.NodeX+0.5) / (float)envLight.NodeDim, ((float)envLight.NodeY+0.5) / (float)envLight.NodeDim );
            float3 localDir = oct_to_ndir_equal_area_unorm(subTexelPos);
            float3 worldDir = EnvironmentQuadLight::ToWorld(localDir);
            lightFull.Base.Center = worldDir * DISTANT_LIGHT_DISTANCE;
            //DebugPrint("", lightFull.Base.Center );
        #endif

            u_lightsBuffer[outLightIndex] = lightFull.Base;
            u_lightsExBuffer[outLightIndex] = lightFull.Extended;

            // figure out our past frame's counterpart if any
            uint historicIndex = RTXPT_INVALID_LIGHT_INDEX;
            if( u_controlBuffer[0].LastFrameTemporalFeedbackAvailable )
            {
                uint dimScale = g_const.EnvMapImportanceMapResolution / envLight.NodeDim;
                uint cx = envLight.NodeX * dimScale;
                uint cy = envLight.NodeY * dimScale;
                historicIndex = u_envLightLookupMap[ uint2(cx, cy) ];   //< Note: at this stage this is still old envLightLookupMap
                // Note: we can't map past to current here because mapping might not be 1<->1
            }
            u_historyRemapCurrentToPast[outLightIndex] = historicIndex;
        }
    }
}

//
// This uses 1 group per node and then splits per-node processing to 8x8 threadgroup. Each node might require outputting just 1 value or 
// up to dimScale^2, i.e. 1000s of elements. There were no attempts to further optimize this approach in any way so far as it's not that costly compared to other parts.
#define FILL_THREAD_COUNT   8
[numthreads(FILL_THREAD_COUNT, FILL_THREAD_COUNT, 1)]
void EnvLightsFillLookupMap( uint lightIndex : SV_GroupID, uint2 threadID : SV_GroupThreadID )
{
    if( lightIndex >= RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT )
        return;

    EnvironmentQuadLight light = LoadEnvironmentQuadLight(lightIndex);

#if 0
    if ( threadID.x == 0 )
        DebugPrint("envLight index {0}: ", lightIndex, light.NodeDim, light.NodeX, light.NodeY );
#endif

    uint dimScale = g_const.EnvMapImportanceMapResolution / light.NodeDim; //assert( dimScale >= 1 );
    for( uint x = 0; (x+threadID.x) < dimScale; x += FILL_THREAD_COUNT )
        for( uint y = 0; (y+threadID.y) < dimScale; y += FILL_THREAD_COUNT )
        {
            uint cx = light.NodeX * dimScale + (x+threadID.x);
            uint cy = light.NodeY * dimScale + (y+threadID.y);
            u_envLightLookupMap[ uint2(cx, cy) ] = lightIndex;
        }
}
//
#if LLB_ENABLE_VALIDATION
bool EnvLightNodeIsInside( uint nodeDim_A, uint nodeX_A, uint nodeY_A, uint nodeDim_B, uint nodeX_B, uint nodeY_B )
{
    while( nodeDim_A != nodeDim_B )
    {
        if( nodeDim_B > nodeDim_A )
        {
            nodeDim_B /= 2;
            nodeX_B /= 2;
            nodeY_B /= 2;
        }
        else
        {
            nodeDim_B *= 2;
            nodeX_B *= 2;
            nodeY_B *= 2;
        }
    }
    return (nodeX_B >= nodeX_A) && (nodeX_B < (nodeX_A+nodeDim_A)) && (nodeY_B >= nodeY_A) && (nodeY_B < (nodeY_A+nodeDim_A)) && (nodeDim_B != 0);
}
#endif
//
[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]       // dispatch is (FEIS_TARGET_QUADTREE_NODE_COUNT, 1, 1)
void EnvLightsMapPastToCurrent( uint historicIndex : SV_DispatchThreadID )
{
    if( historicIndex >= RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT )
        return;

    const LightingControlData controlInfo = u_controlBuffer[0];

    uint presentIndex = RTXPT_INVALID_LIGHT_INDEX;
    if( controlInfo.LastFrameTemporalFeedbackAvailable )
    {
        uint nodeDim, nodeX, nodeY;
        EQTNodeUnpack(u_scratchList[historicIndex], nodeDim, nodeX, nodeY);  // Note: these are the old nodes backed up in the first pass; u_scratchList no longer used after this!
        uint dimScale = g_const.EnvMapImportanceMapResolution / nodeDim;
        uint cx = nodeX * dimScale;
        uint cy = nodeY * dimScale;
        presentIndex = u_envLightLookupMap[ uint2(cx, cy) ];   //< Note: at this stage this is the current envLightLookupMap!

#if LLB_ENABLE_VALIDATION
        EnvironmentQuadLight light = LoadEnvironmentQuadLight(presentIndex);
        if ( !EnvLightNodeIsInside( light.NodeDim, light.NodeX, light.NodeY, nodeDim, nodeX, nodeY ) )
            DebugPrint("Error mapping envmap node historicIndex {0} to presentIndex {1}", historicIndex, presentIndex, nodeDim, nodeX, nodeY, light.NodeDim, light.NodeX, light.NodeY ); //, light.NodeDim, light.NodeX, light.NodeY );
#endif
    }
    u_historyRemapPastToCurrent[historicIndex] = presentIndex;
}
///
/// END OF ENVMAP SECTION
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


[numthreads(8*LLB_MAX_TRIANGLES_PER_TASK, 1, 1)]
void BakeEmissiveTriangles( uint dispatchThreadID : SV_DispatchThreadID, uint groupThreadID : SV_GroupThreadID ) // note, this is adding triangle lights only - analytic lights have been added on the CPU side already
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    if( dispatchThreadID.x/LLB_MAX_TRIANGLES_PER_TASK >= g_const.TriangleLightTaskCount )
        return;

    EmissiveTrianglesProcTask task = u_scratchBuffer.Load<EmissiveTrianglesProcTask>((dispatchThreadID.x/LLB_MAX_TRIANGLES_PER_TASK) * sizeof(EmissiveTrianglesProcTask));

    InstanceData instance = t_InstanceData[task.InstanceIndex];
    //uint geometryInstanceIndex = instance.firstGeometryIndex + task.geometryIndex;
    GeometryData geometry = t_GeometryData[instance.firstGeometryIndex + task.GeometryIndex];   // <- can precompute this into task.geometryIndex

    uint materialIndex = t_SubInstanceData[instance.firstGeometryInstanceIndex + task.GeometryIndex].GlobalGeometryIndex_PTMaterialDataIndex & 0xFFFF;
    PTMaterialData material = t_PTMaterialData[materialIndex];

    //DebugPrint( "tID {0}; fgii {1}, fgi {2}, ng {3}", dispatchThreadID, instance.firstGeometryInstanceIndex, instance.firstGeometryIndex, instance.numGeometries  );

    // if( task.EmissiveLightMappingOffset != (instance.firstGeometryInstanceIndex + task.GeometryIndex) )
    //     DebugPrint( "ELMO {0}, FGII {1}, GI{2}", task.EmissiveLightMappingOffset, instance.firstGeometryIndex, task.GeometryIndex );

    int triangleCount = task.TriangleIndexTo-task.TriangleIndexFrom;

    // culling removed unfortunately to maintain fixed memory allocation and track it from the CPU side
    uint subIndex = dispatchThreadID.x%LLB_MAX_TRIANGLES_PER_TASK;

    ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.indexBufferIndex)];
    ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.vertexBufferIndex)];

    //for( uint triangleIdx = task.TriangleIndexFrom; triangleIdx < task.TriangleIndexTo; triangleIdx++ )
    uint triangleIdx = task.TriangleIndexFrom+subIndex;
    if ( triangleIdx < task.TriangleIndexTo )
    {
        // DebugPrint( "NEW: ii {0}; gi {1}, gii {2}, ti {3}, T0{4}, T1{5}, T2{6}", task.instanceIndex, task.geometryIndex, geometryInstanceIndex, triangleIdx, instance.transform[0], instance.transform[1], instance.transform[2] );

        uint3 indices = indexBuffer.Load3(geometry.indexOffset + triangleIdx * c_SizeOfTriangleIndices);

        float3 positions[3];

        positions[0] = asfloat(vertexBuffer.Load3(geometry.positionOffset + indices[0] * c_SizeOfPosition));
        positions[1] = asfloat(vertexBuffer.Load3(geometry.positionOffset + indices[1] * c_SizeOfPosition));
        positions[2] = asfloat(vertexBuffer.Load3(geometry.positionOffset + indices[2] * c_SizeOfPosition));

        // DebugTriangle( positions[0], positions[1], positions[2], float4( 1, 0, 0, 0.1 ) );

        positions[0] = mul(instance.transform, float4(positions[0], 1)).xyz;
        positions[1] = mul(instance.transform, float4(positions[1], 1)).xyz;
        positions[2] = mul(instance.transform, float4(positions[2], 1)).xyz;

        float3 radiance = material.EmissiveColor;

        if ((material.EmissiveTextureIndex != 0xFFFFFFFF) && (geometry.texCoord1Offset != ~0u) && ((material.Flags & PTMaterialFlags_UseEmissiveTexture) != 0))
        {
            Texture2D emissiveTexture = t_BindlessTextures[NonUniformResourceIndex(material.EmissiveTextureIndex & 0xFFFF)];

            // Load the vertex UVs
            float2 uvs[3];
            uvs[0] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[0] * c_SizeOfTexcoord));
            uvs[1] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[1] * c_SizeOfTexcoord));
            uvs[2] = asfloat(vertexBuffer.Load2(geometry.texCoord1Offset + indices[2] * c_SizeOfTexcoord));

            // Calculate the triangle edges and edge lengths in UV space
            float2 edges[3];
            edges[0] = uvs[1] - uvs[0];
            edges[1] = uvs[2] - uvs[1];
            edges[2] = uvs[0] - uvs[2];

            float3 edgeLengths;
            edgeLengths[0] = length(edges[0]);
            edgeLengths[1] = length(edges[1]);
            edgeLengths[2] = length(edges[2]);

            // Find the shortest edge and the other two (longer) edges
            float2 shortEdge;
            float2 longEdge1;
            float2 longEdge2;

            if (edgeLengths[0] < edgeLengths[1] && edgeLengths[0] < edgeLengths[2])
            {
                shortEdge = edges[0];
                longEdge1 = edges[1];
                longEdge2 = edges[2];
            }
            else if (edgeLengths[1] < edgeLengths[2])
            {
                shortEdge = edges[1];
                longEdge1 = edges[2];
                longEdge2 = edges[0];
            }
            else
            {
                shortEdge = edges[2];
                longEdge1 = edges[0];
                longEdge2 = edges[1];
            }

            // Use anisotropic sampling with the sample ellipse axes parallel to the short edge
            // and the median from the opposite vertex to the short edge.
            // This ellipse is roughly inscribed into the triangle and approximates long or skinny
            // triangles with highly anisotropic sampling, and is mostly round for usual triangles.
            float2 shortGradient = shortEdge * (2.0 / 3.0);
            float2 longGradient = (longEdge1 + longEdge2) / 3.0;

            // Sample
            float2 centerUV = (uvs[0] + uvs[1] + uvs[2]) / 3.0;
            float3 emissiveMask = emissiveTexture.SampleGrad(s_materialSampler, centerUV, shortGradient, longGradient).rgb;

            radiance *= emissiveMask;
        }

        radiance.rgb = max(0, radiance.rgb);

        // radiance.rgb *= 0;

        // Check if the transform flips the coordinate system handedness (its determinant is negative).
        float3x3 transform;
        transform._m00_m01_m02 = (instance.transform._m00_m01_m02);
        transform._m10_m11_m12 = (instance.transform._m10_m11_m12);
        transform._m20_m21_m22 = (instance.transform._m20_m21_m22);

        bool isFlipped = determinant(transform) < 0.f;

        TriangleLight triLight;
        triLight.base = positions[0];
        if (!isFlipped)
        {
            triLight.edge1 = positions[1] - positions[0];
            triLight.edge2 = positions[2] - positions[0];
        }
        else
        {
            triLight.edge1 = positions[2] - positions[0];
            triLight.edge2 = positions[1] - positions[0];
        }

        float maxR = max(radiance.x, max(radiance.y, radiance.z));
        if( maxR < 1e-7f )
        {
            radiance = float3(0,0,0);
            maxR = 0;
        }

        triLight.radiance = radiance;

        // debugging        
        // if( dispatchThreadID.x % 10 == 0 )
        // {
        //     //DebugPrint( "tID {0}; base {1}, radiance: {2}", dispatchThreadID, triLight.base, triLight.radiance );
        //     // DebugTriangle( triLight.base, triLight.base+float3(0.5, 0.0, 0.0), triLight.base+float3(0.0, 0.5, 0.5), float4( 1, 0, 0, 1 ) );
        //     // DebugTriangle( triLight.base, triLight.base+float3(0.0, 0.5, 0.0), triLight.base+float3(0.5, 0.0, 0.5), float4( 0, 1, 0, 1 ) );
        //     // DebugTriangle( triLight.base, triLight.base+float3(0.0, 0.0, 0.5), triLight.base+float3(0.5, 0.5, 0.0), float4( 0, 0, 1, 1 ) );
        // }
        // 
        // //DebugTriangle( triLight.base, triLight.base+triLight.edge1, triLight.base+triLight.edge2, float4( triLight.radiance, 1 ) );

        // float emissiveFlux = PolymorphicLight::GetPower(triLight.Store());

        uint uniqueID = Hash32CombineSimple( Hash32CombineSimple(Hash32(subIndex), Hash32(task.InstanceIndex)), Hash32(task.GeometryIndex) );

        uint lightIndex = task.DestinationBufferOffset+subIndex;

        PolymorphicLightInfoFull lightFull = triLight.Store(uniqueID);
        u_lightsBuffer[lightIndex] = lightFull.Base;
        u_lightsExBuffer[lightIndex] = lightFull.Extended;

        uint historicIndex = RTXPT_INVALID_LIGHT_INDEX;
        if( task.HistoricBufferOffset != RTXPT_INVALID_LIGHT_INDEX )
        {
            historicIndex = task.HistoricBufferOffset+subIndex;
            u_historyRemapPastToCurrent[historicIndex] = lightIndex;
        }

        u_historyRemapCurrentToPast[lightIndex] = historicIndex;

        subIndex++;
    }

    // this is how we used to do it, but introduces non-determinism in the order of lights and messes up ordering
    // uint outLightIndex;
    // InterlockedAdd(u_controlBuffer[0].TotalLightCount, collectedLightCount, outLightIndex);   
}

// from https://www.gamedev.net/forums/topic/613648-dx11-interlockedadd-on-floats-in-pixel-shader-workaround/
void InterlockedAddFloat_WeightSum( float value ) // Works perfectly! <- original comment, I won't remove because it inspires confidence
{ 
   uint i_val = asuint(value);
   uint tmp0 = 0;
   uint tmp1;

#ifndef SPIRV
   [allow_uav_condition]
#endif
   while (true)
   {
      InterlockedCompareExchange( u_controlBuffer[0].WeightsSumUINT, tmp0, i_val, tmp1);
      if (tmp1 == tmp0)
         break;
      tmp0 = tmp1;
      i_val = asuint(value + asfloat(tmp1));
   }
}

float ComputeWeight( const LightingControlData controlInfo, const PolymorphicLightInfoFull light )
{
    // Calculate the total flux
    // We do not have to check light types as GetPower handles directional and environment lights (returns zero)
    float emissiveFlux = PolymorphicLight::GetPower(light);
        
    float weight = emissiveFlux; // weight is just emissive flux now - could be scaled by LOD like distance to camera 
    //float weight = sqrt(emissiveFlux); // alternative: weight is the square root of emissive flux - actually works better in some cases

    if( weight < RTXPT_LIGHTING_MIN_WEIGHT_THRESHOLD )
        weight = 0;

    return weight;
}

[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void ResetLightProxyCounters( uint dispatchThreadID : SV_DispatchThreadID, uint groupThreadID : SV_GroupThreadId )
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    const uint lightIndex = dispatchThreadID;
    const uint lightCount = g_const.TotalLightCount; //controlInfo.TotalLightCount;
    if( lightIndex > lightCount ) // also zero out last element, because that's where we store invalid light count - that's why it's `>` and not `>=`
        return;

    u_perLightProxyCounters[lightIndex] = 0;
}

// Needed only for dynamic resolution (where viewport is resizing dynamically)
float3 ConvertMotionVectorToPixelSpace( int2 pixelPosition, float3 motionVector)
{
    float2 currentPixelCenter = float2(pixelPosition.xy) + 0.5;
    float2 previousPosition = currentPixelCenter + motionVector.xy;
    previousPosition *= g_const.PrevOverCurrentViewportSize;
    motionVector.xy = previousPosition - currentPixelCenter;
    return motionVector;
}

[numthreads(8, 8, 1)]
void ClearFeedbackHistory( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    uint2 pixelPos = dispatchThreadID;

    bool indirect = false; // prevent direct and indirect parts, which are stacked one on the other, bleeding into each other
    if( pixelPos.y >= g_const.FeedbackResolution.y )
    {
        indirect = true;
        pixelPos.y -= g_const.FeedbackResolution.y;
    }

    if( pixelPos.x >= g_const.FeedbackResolution.x || pixelPos.y >= g_const.FeedbackResolution.y )
        return;

#if RTXPT_LIGHTING_NEEAT_ENABLE_RESERVOIR_HISTORY
    if( controlInfo.LastFrameTemporalFeedbackAvailable )
    {
        float3 screenSpaceMotion = ConvertMotionVectorToPixelSpace( pixelPos, t_motionVectors[pixelPos] );
        int2 prevPixelPos = int2( float2(pixelPos) + screenSpaceMotion.xy + 0.5.xx /* + sampleNext2D(sampleGenerator)*/ );

        // if wrong/missing motion vectors, use current as backup
        if( !(all(prevPixelPos >= 0.xx) && all(prevPixelPos < g_const.FeedbackResolution.xy)) )
            prevPixelPos = pixelPos;

        LightFeedbackReservoir pastReservoir = LightFeedbackReservoir::Unpack8Byte(u_feedbackReservoirBufferScratch[ (uint2)prevPixelPos.xy + uint2(0, indirect?g_const.FeedbackResolution.y:0) ]);
        pastReservoir.Scale(0.05);
        if( !(f16tof32(f32tof16(pastReservoir.CandidateWeight)) > 0 && pastReservoir.TotalWeight > pastReservoir.CandidateWeight ) )
            pastReservoir = LightFeedbackReservoir::make();
        u_feedbackReservoirBuffer[(uint2)pixelPos.xy + uint2(0, indirect?g_const.FeedbackResolution.y:0)] = pastReservoir.Pack8Byte();
    }
    else
#endif
    {
        LightFeedbackReservoir emptyReservoir = LightFeedbackReservoir::make();
        u_feedbackReservoirBuffer[dispatchThreadID.xy] = emptyReservoir.Pack8Byte();
    }


    if( g_const.DebugDrawType == (int)LightingDebugViewType::FeedbackReadyForNew && g_const.DebugDrawDirect == !indirect )
    {
        LightFeedbackReservoir reservoir = LightFeedbackReservoir::Unpack8Byte( u_feedbackReservoirBuffer[ (uint2)pixelPos.xy + uint2(0, indirect?g_const.FeedbackResolution.y:0) ] );
        
        uint dbgLightIndex = reservoir.CandidateIndex;

        DebugPixel( pixelPos.xy, float4( ColorFromHash(Hash32(dbgLightIndex)), 0.95) );
    }
}

PolymorphicLightInfoFull LoadLight(uint lightIndex) // used to facilitate sort mapping with "return u_lightsBuffer[u_lightSortIndices[lightIndex]]"
{
    return PolymorphicLightInfoFull::make( u_lightsBuffer[lightIndex], u_lightsExBuffer[lightIndex] );
}

groupshared float g_blockWeightSums[LLB_NUM_COMPUTE_THREADS]; // these contain per-block (LLB_LOCAL_BLOCK_SIZE) sums
[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void ComputeWeights( uint dispatchThreadID : SV_DispatchThreadID, uint groupThreadID : SV_GroupThreadId )
{
    const LightingControlData controlInfo = u_controlBuffer[0];
    // if( dispatchThreadID == 0 )
    //    u_controlBuffer[0].SamplingProxyCount = 0; // controlInfo.TotalLightCount; <- init to zero

    const int from = dispatchThreadID.x * LLB_LOCAL_BLOCK_SIZE;
    const int to = min( from + LLB_LOCAL_BLOCK_SIZE, g_const.TotalLightCount );

    // this breaks stuff - something to do with group memory barrier sync
    // if( from >= controlInfo.TotalLightCount )
    //     return;

    float blockWeightSum = 0.0;
    for( int lightIndex = from; lightIndex < to; lightIndex ++ )
    {
        if( lightIndex >= g_const.TotalLightCount )
            DebugPrint( "Danger, overflow", groupThreadID, from, to );

        PolymorphicLightInfoFull packedLightInfo = LoadLight( lightIndex );

        float weight = ComputeWeight(controlInfo, packedLightInfo);
        u_lightWeights[ lightIndex ] = weight;
        blockWeightSum += weight;
    }
    
    g_blockWeightSums[groupThreadID] = blockWeightSum;

    GroupMemoryBarrierWithGroupSync();

    float total = 0.0;
    if( groupThreadID == 0 )
    {
        for( int i = 0; i < LLB_NUM_COMPUTE_THREADS; i++ )
            total += g_blockWeightSums[i];

        // Note, due to precision issues we could, in theory, under-sum the total here. This could, in theory, result in an overflow of required number of proxies
        // This needs to be accounted for at some point.
        InterlockedAddFloat_WeightSum(total);
    }
}

[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void ComputeProxyCounts( uint dispatchThreadID : SV_DispatchThreadID, uint groupThreadID : SV_GroupThreadId )
{
#if 0
    if( dispatchThreadID == 0 )
    {
        float testSum = 0;
        const LightingControlData controlInfo = u_controlBuffer[0];
        for( int lightIndex = 0; lightIndex < g_const.TotalLightCount; lightIndex ++ )
            testSum += u_lightWeights[ lightIndex ];

        if( !RelativelyEqual( controlInfo.WeightsSum(), testSum, 5e-5f ) )
            DebugPrint( "Compute weight sum {0}, test: {1}", controlInfo.WeightsSum(), testSum );
    }
#endif

    const LightingControlData controlInfo = u_controlBuffer[0];

    const uint lightIndex = dispatchThreadID;
    const uint lightCount = g_const.TotalLightCount;
    if( lightIndex >= lightCount )
        return;

    const uint cTotalSamplingProxiesBudget = RTXPT_LIGHTING_SAMPLING_PROXY_RATIO*(max( g_const.TotalLightCount, RTXPT_LIGHTING_MAX_LIGHTS/20 ) );    // Sampling proxies budget is based on current total lights or 5% of max supported lights, whichever is greater. This allows small number of lights to benefit from better balancing, without adding too much to the overall cost.
    const float weightSum = asfloat(controlInfo.WeightsSumUINT);

    // this is what comes from past frame's feedback on light usage
    uint validFeedbackCount = g_const.TotalFeedbackCount - u_perLightProxyCounters[g_const.TotalLightCount];
    //if (validFeedbackCount != controlInfo.ValidFeedbackCount)
    //    DebugPrint("Error in valid feedback count", validFeedbackCount, controlInfo.ValidFeedbackCount);
    const float feedbackWeight = (float)u_perLightProxyCounters[lightIndex] * weightSum / (float)max( 1.0, validFeedbackCount );

    // if( dispatchThreadID==0 )
    //     DebugPrint("Valid count {0} ", validFeedbackCount );


    // combine computed light weights with historical usage-based feedback weight
    const float lightWeight = lerp( u_lightWeights[ lightIndex ], feedbackWeight, g_const.GlobalFeedbackUseRatio );

    uint lightSamplingProxies = 0;
    if( lightWeight > 0 )
        // if controlInfo.ImportanceSamplingType==0, we use 1 proxy per light - all this is unnecessary but kept in to reduce code complexity as "uniform" mode is for reference/testing only anyway
        lightSamplingProxies = (controlInfo.ImportanceSamplingType==0)?(1):(uint( ceil( (float(cTotalSamplingProxiesBudget-g_const.TotalLightCount) * lightWeight) / weightSum ) ));

    // limit the boost offered by proxies - possibly unnecessary limitation, but would in theory allow us to pack it to 16bits for sampling
    lightSamplingProxies = min( lightSamplingProxies, RTXPT_LIGHTING_MAX_SAMPLING_PROXIES_PER_LIGHT );

    // store! this is used by sampling
    u_perLightProxyCounters[lightIndex] = lightSamplingProxies;

    AllMemoryBarrierWithGroupSync();

    // NOTE: 
    //  * we still don't use u_lightSamplingProxies so use them to save base offsets

    uint total = 0;
    if( groupThreadID == 0 )
    {
        for( int i = lightIndex; i < min(lightIndex+LLB_NUM_COMPUTE_THREADS, lightCount); i++ )
        {
            u_scratchList[i] = total;      // this is where local - in [i*LLB_NUM_COMPUTE_THREADS, (i+1)LLB_NUM_COMPUTE_THREADS) range - offsets are stored
            total += u_perLightProxyCounters[i];
        }
        u_lightSamplingProxies[lightIndex/LLB_NUM_COMPUTE_THREADS+1] = total; // this is where total counts for each LLB_NUM_COMPUTE_THREADS are stored

        InterlockedAdd( u_controlBuffer[0].SamplingProxyCount, total );
    }
}

#if 0
[numthreads(1, 1, 1)]
void ComputeProxyBaselineOffsets( uint dispatchThreadID : SV_DispatchThreadID, uint groupThreadID : SV_GroupThreadId )
{
    const LightingControlData controlInfo = u_controlBuffer[0];
    const uint lightCount = g_const.TotalLightCount;

    u_lightSamplingProxies[0] = 0;

    uint counter = 0;
    int lightIndex = LLB_NUM_COMPUTE_THREADS;
    for( lightIndex = LLB_NUM_COMPUTE_THREADS; lightIndex < lightCount; lightIndex += LLB_NUM_COMPUTE_THREADS )
    {
        counter += u_lightSamplingProxies[lightIndex/LLB_NUM_COMPUTE_THREADS];
        u_lightSamplingProxies[lightIndex/LLB_NUM_COMPUTE_THREADS] = counter;
        //DebugPrint( "BASE {0}, {1}", lightIndex/LLB_NUM_COMPUTE_THREADS, u_lightSamplingProxies[lightIndex/LLB_NUM_COMPUTE_THREADS] );
    }
    counter += u_lightSamplingProxies[lightIndex/LLB_NUM_COMPUTE_THREADS];

    if( counter != u_controlBuffer[0].SamplingProxyCount )
        DebugPrint( "Proxies count error {0} != {1}", counter, u_controlBuffer[0].SamplingProxyCount );
}
#else
[numthreads(32, 1, 1)]
void ComputeProxyBaselineOffsets( uint groupThreadID : SV_GroupThreadId )
{
    const LightingControlData controlInfo = u_controlBuffer[0];
    const uint lightCount = g_const.TotalLightCount;

    // if( groupThreadID == 0 )
    //     DebugPrint( "", lightCount );

    if (groupThreadID == 0)
        u_lightSamplingProxies[0] = 0;

    GroupMemoryBarrierWithGroupSync();

    uint counter = 0;
    int lightIndex = 0;
    for( ; lightIndex < (lightCount+LLB_NUM_COMPUTE_THREADS-1); lightIndex += LLB_NUM_COMPUTE_THREADS*32 )
    {
        int actualIndex = lightIndex + (groupThreadID*LLB_NUM_COMPUTE_THREADS);
        uint lastBlockCount = 0;
        if (actualIndex < (lightCount+LLB_NUM_COMPUTE_THREADS-1))
            lastBlockCount = u_lightSamplingProxies[actualIndex/LLB_NUM_COMPUTE_THREADS];
        
        GroupMemoryBarrierWithGroupSync();

        uint mySum      = WavePrefixSum(lastBlockCount)+lastBlockCount;
        uint totalSum   = WaveActiveSum(lastBlockCount);

        if (actualIndex < (lightCount+LLB_NUM_COMPUTE_THREADS-1))
            u_lightSamplingProxies[actualIndex/LLB_NUM_COMPUTE_THREADS] = counter + mySum;

        counter += totalSum;
    }

    if( groupThreadID == 0 && counter != u_controlBuffer[0].SamplingProxyCount )
        DebugPrint( "Proxies count error {0} != {1}", counter, u_controlBuffer[0].SamplingProxyCount );
}
#endif

[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void CreateProxyJobs( uint dispatchThreadID : SV_DispatchThreadID, uint groupThreadID : SV_GroupThreadId )
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    const uint lightIndex = dispatchThreadID;
    const uint lightCount = g_const.TotalLightCount;
    if( lightIndex >= lightCount )
        return;

    uint storageBaseIndex = u_scratchList[lightIndex] + u_lightSamplingProxies[lightIndex/LLB_NUM_COMPUTE_THREADS];

     // if( lightIndex > 100 && lightIndex < 150 )
     //     DebugPrint( "N {0}, {1}, {2}, -- {3}", lightIndex, storageBaseIndex, u_lightSamplingProxies[lightIndex/LLB_NUM_COMPUTE_THREADS], u_perLightProxyCounters[lightIndex] );

    uint lightSamplingProxies = u_perLightProxyCounters[lightIndex];

    // if( lightIndex < 15000 && lightSamplingProxies != (nextBaseIndex - storageBaseIndex) )
    //     DebugPrint( "M {0}, {1}, c{2}, n{3}", lightIndex, lightSamplingProxies, storageBaseIndex, nextBaseIndex );

    uint tasksRequired = (lightSamplingProxies + LLB_MAX_PROXIES_PER_TASK - 1) / LLB_MAX_PROXIES_PER_TASK;
    uint taskBaseIndex;
    InterlockedAdd( u_controlBuffer[0].ProxyBuildTaskCount, tasksRequired, taskBaseIndex );

    uint localTotal = 0;
    for( int i = 0; i < tasksRequired; i++ )
    {
        SamplingProxyBuildProcTask task;
        task.LightIndex = lightIndex;
        task.ProxyIndexBase = storageBaseIndex;

        task.FillProxyIndexFrom = storageBaseIndex + i * LLB_MAX_PROXIES_PER_TASK;
        task.FillProxyIndexTo   = storageBaseIndex + min( (i+1) * LLB_MAX_PROXIES_PER_TASK, lightSamplingProxies );
        u_scratchBuffer.Store<SamplingProxyBuildProcTask>( (taskBaseIndex+i) * sizeof(SamplingProxyBuildProcTask), task );

        localTotal += task.FillProxyIndexTo - task.FillProxyIndexFrom;
    }
#if 1
    if( localTotal != lightSamplingProxies )
        DebugPrint( "Danger, danger danger {0} != {1}", localTotal, lightSamplingProxies );
#endif
}

[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void ExecuteProxyJobs( uint dispatchThreadID : SV_DispatchThreadID)
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    const uint taskIndex = dispatchThreadID;
    const uint taskCount = controlInfo.ProxyBuildTaskCount;
    if( taskIndex >= taskCount )
        return;

    SamplingProxyBuildProcTask task = u_scratchBuffer.Load<SamplingProxyBuildProcTask>( taskIndex * sizeof(SamplingProxyBuildProcTask) );

    for( int proxyIndex = task.FillProxyIndexFrom; proxyIndex < task.FillProxyIndexTo; proxyIndex++ )
    {
        u_lightSamplingProxies[proxyIndex] = task.LightIndex;

        if( task.ProxyIndexBase > proxyIndex )
            DebugPrint( "Danger, strange mismatch - barrier missing?" );
    }
}

uint RemapPastToCurrent(const LightingControlData controlInfo, uint historicLightIndex)
{
    uint lightIndex = RTXPT_INVALID_LIGHT_INDEX;
    if (historicLightIndex != RTXPT_INVALID_LIGHT_INDEX)
    {
        // it's essential to bounds-check against controlInfo.HistoricTotalLightCount
        lightIndex = ( historicLightIndex < controlInfo.HistoricTotalLightCount )?(u_historyRemapPastToCurrent[historicLightIndex]):(RTXPT_INVALID_LIGHT_INDEX);

        if ( lightIndex != RTXPT_INVALID_LIGHT_INDEX )
        {
            if ( lightIndex >= g_const.TotalLightCount )
            {
                lightIndex = RTXPT_INVALID_LIGHT_INDEX;
                DebugPrint( "3 - Danger, overflow {0}", lightIndex );
            }
        }
        // else
        //     DebugPrint( "History not found for {0}", historicLightIndex );

        // if( historicLightIndex != lightIndex )
        //     DebugPrint( "{0} maps to {1}", historicLightIndex, lightIndex );
    }
    return lightIndex;
}

[numthreads(16, 16, 1)]
void UpdateFeedbackIndices( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    uint2 pixelCoord = dispatchThreadID.xy;

    // bool indirect = pixelCoord.y >= g_const.FeedbackResolution.y;

    LightFeedbackReservoir reservoir = LightFeedbackReservoir::Unpack8Byte( u_feedbackReservoirBuffer[ (uint2)pixelCoord.xy ] );
    
    reservoir.CandidateIndex = RemapPastToCurrent(controlInfo, reservoir.CandidateIndex);
   
    u_feedbackReservoirBuffer[ (uint2)pixelCoord.xy ] = reservoir.Pack8Byte();
}

LightFeedbackReservoir SampleFeedback(int2 coord, bool indirect)
{
    if( !(all(coord >= 0.xx) && all(coord < g_const.FeedbackResolution.xy)) )
        return LightFeedbackReservoir::make();

    LightFeedbackReservoir reservoir = LightFeedbackReservoir::Unpack8Byte( u_feedbackReservoirBuffer[ (uint2)coord.xy + uint2(0, indirect?g_const.FeedbackResolution.y:0) ] );

    uint lightIndex = reservoir.CandidateIndex;

    // Empty is completely valid
    if (lightIndex == RTXPT_INVALID_LIGHT_INDEX)
        return LightFeedbackReservoir::make();

    reservoir.CandidateIndex = lightIndex;
    return reservoir;
}

bool SampleFeedback(int2 coord, bool indirect, inout uint lightIndex)
{
    LightFeedbackReservoir reservoir = SampleFeedback(coord, indirect);
    lightIndex = reservoir.CandidateIndex;
    return lightIndex != RTXPT_INVALID_LIGHT_INDEX;
}

static const uint c_directNeighbourCount = 5;
static const int2 c_directNeighbourOffsets[c_directNeighbourCount] = {     int2( 0, 0),
                                int2(-1, 0), int2(+1, 0), int2( 0,-1), int2( 0,+1),
                                //int2(-1,-1), int2(+1,-1), int2(-1,+1), int2(+1,+1) 
                                };

// Flood fill empty neighbours and apply feedback into global sampler
[numthreads(16, 16, 1)] void ProcessFeedbackHistoryP0( uint2 dispatchThreadID : SV_DispatchThreadID, uint2 groupID : SV_GroupID )
{
    uint2 pixelCoord = dispatchThreadID.xy;

    bool indirect = false; // prevent direct and indirect parts, which are stacked one on the other, bleeding into each other
    if( pixelCoord.y >= g_const.FeedbackResolution.y )
    {
        indirect = true;
        pixelCoord.y -= g_const.FeedbackResolution.y;
    }

    uint lightIndex = RTXPT_INVALID_LIGHT_INDEX;

    if( pixelCoord.x < g_const.FeedbackResolution.x && pixelCoord.y < g_const.FeedbackResolution.y )
    {
        if( g_const.DebugDrawType == (int)LightingDebugViewType::FeedbackRaw || g_const.DebugDrawType == (int)LightingDebugViewType::MissingFeedback )
        {
            uint dbgLightIndex = 0;
            bool hasSample = SampleFeedback( int2(pixelCoord), indirect, dbgLightIndex ) && dbgLightIndex != RTXPT_INVALID_LIGHT_INDEX;

            if( g_const.DebugDrawType == (int)LightingDebugViewType::MissingFeedback && g_const.DebugDrawDirect == !indirect )
                DebugPixel( pixelCoord.xy, float4( 1 - hasSample, hasSample*0.3, 0, 0.95) );
            if( g_const.DebugDrawType == (int)LightingDebugViewType::FeedbackRaw && g_const.DebugDrawDirect == !indirect )
                DebugPixel( pixelCoord.xy, float4( ColorFromHash(Hash32(dbgLightIndex)), 0.95) );
        }

        SampleGenerator sampleGenerator = SampleGenerator::make( SampleGeneratorVertexBase::make( pixelCoord, 0, g_const.UpdateCounter ) );

        LightFeedbackReservoir reservoir = SampleFeedback( int2(pixelCoord)+c_directNeighbourOffsets[0], indirect );

        if ( reservoir.CandidateIndex == RTXPT_INVALID_LIGHT_INDEX )    // search neighbourhood but only if center is invalid
        {
            for( int i = 1; i < c_directNeighbourCount; i++ )
            {
                LightFeedbackReservoir other = SampleFeedback( int2(pixelCoord)+c_directNeighbourOffsets[i], indirect );
                if( other.CandidateIndex == RTXPT_INVALID_LIGHT_INDEX )
                    continue;
                other.Scale(0.05); // TODO; move this outside - make the central one 20x larger :)
                reservoir.Merge( sampleNext1D( sampleGenerator ), other );
                //reservoir.Add( sampleNext1D( sampleGenerator ), other.CandidateIndex, other.CandidateWeight * ((i==0)?20:1) );
            }
        }
        lightIndex = reservoir.CandidateIndex;

        if (lightIndex == RTXPT_INVALID_LIGHT_INDEX)
            reservoir = LightFeedbackReservoir::make();

    #if RTXPT_LIGHTING_NEEAT_ENABLE_RESERVOIR_HISTORY
        u_feedbackReservoirBufferScratch[pixelCoord.xy + uint2(0, indirect?g_const.FeedbackResolution.y:0)] = reservoir.Pack8Byte();
    #endif

        // write processed index - this is now current frame's index; only do this for direct lighting
    #if RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER == 0
        if( !indirect )
    #endif
            u_processedFeedbackBuffer[pixelCoord + uint2(0, indirect?g_const.FeedbackResolution.y:0)] = lightIndex;
    }

    // // for debugging
    // if (lightIndex != RTXPT_INVALID_LIGHT_INDEX)
    //     InterlockedAdd( u_controlBuffer[0].ValidFeedbackCount, 1 );

#if 0 // simple verison with no wave intrinsics - for reference & debugging
    InterlockedAdd( u_perLightProxyCounters[min(lightIndex, g_const.TotalLightCount)], 1 ); // when lightIndex == RTXPT_INVALID_LIGHT_INDEX, we store "non-valid feedback" in the special last place
#else // optimized wave intrinsics variants
    // new SM 6.5 version!
    uint4 matchingBitmask = WaveMatch(lightIndex);
    uint4 matchingCount4 = countbits(matchingBitmask);
    uint matchingCount = matchingCount4.x+matchingCount4.y+matchingCount4.z+matchingCount4.w;
    #if 0
    int4 highLanes = (int4)(firstbithigh(matchingBitmask) | uint4(0, 0x20, 0x40, 0x60));
    // The signed max should be the highest lane index in the group.
    uint highLane = (uint)max(max(max(highLanes.x, highLanes.y), highLanes.z), highLanes.w);
    bool weAreFirst = WaveGetLaneIndex() == highLane;
    #else // simpler version? seems at least as fast
    bool weAreFirst = WaveMultiPrefixCountBits(1, matchingBitmask) == 0;
    #endif
    if (weAreFirst)
    {
        // we use u_perLightProxyCounters[g_const.TotalLightCount] as the place for NonValidFeedbackCount to avoid storing it separately; when lightIndex == RTXPT_INVALID_LIGHT_INDEX, it's stored there
        InterlockedAdd( u_perLightProxyCounters[min(lightIndex, g_const.TotalLightCount)], matchingCount );
    }
#endif
}

int2 MirrorCoord( const int2 inCoord, const int2 maxResolution )
{
    int2 ret = select(inCoord>=0, inCoord, -inCoord);
    ret = select(ret<maxResolution, ret, 2*maxResolution-2-ret);
    return clamp( ret, 0.xx, maxResolution ); // no handling of more than 1 screen away
}

// these are used <only> for filling in the gaps
uint SampleLightGlobal(inout SampleGenerator sampleGenerator)
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    float rnd = sampleNext1D(sampleGenerator);
    uint totalProxyCount = controlInfo.SamplingProxyCount;
    uint indexInIndex = clamp( uint(rnd * totalProxyCount), 0, totalProxyCount-1 );    // when rnd guaranteed to be [0, 1), clamp is unnecessary

    return u_lightSamplingProxies[indexInIndex];
}

uint SampleLightLocalHistoric(uint2 tilePos, bool indirect, inout SampleGenerator sampleGenerator)
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    uint indexInIndex = sampleGenerator.Next() % RTXPT_LIGHTING_NARROW_PROXY_COUNT;

    return RemapPastToCurrent(controlInfo, UnpackMiniListLight(u_narrowSamplingBuffer[ uint3(tilePos.xy + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0), indexInIndex) ]));
}

// This is where reprojection happens, as well as insertion of historical sampling
[numthreads(8, 8, 1)] void ProcessFeedbackHistoryP1( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    bool indirect = false; // prevent direct and indirect parts, which are stacked one on the other, bleeding into each other
#if RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER
    if( dispatchThreadID.y >= g_const.FeedbackResolution.y )
    {
        indirect = true;
        dispatchThreadID.y -= g_const.FeedbackResolution.y;
    }
#endif

    if( dispatchThreadID.x >= g_const.FeedbackResolution.x || dispatchThreadID.y >= g_const.FeedbackResolution.y )
        return;
 
    SampleGenerator sampleGenerator = SampleGenerator::make( SampleGeneratorVertexBase::make( dispatchThreadID.xy, 0, g_const.UpdateCounter ), SampleGeneratorEffectSeed::Base, false, 1 );

    {
        int2 pixelPos = dispatchThreadID;
        float3 screenSpaceMotion = ConvertMotionVectorToPixelSpace( pixelPos, t_motionVectors[pixelPos] );
        int2 prevPixelPos = int2( float2(pixelPos) + screenSpaceMotion.xy + 0.5.xx /* + sampleNext2D(sampleGenerator)*/ );

        // if wrong/missing motion vectors, use current as backup
        if( !(all(prevPixelPos >= 0.xx) && all(prevPixelPos < g_const.FeedbackResolution.xy)) )
            prevPixelPos = pixelPos;

        // note: it might not be a bad idea to stochastically ignore 10% of motion vectors and sample from original - this catches reflected speculars in lateral motion that stay in place

        // part 1: sample direct feedback buffer
        uint lightIndex = RTXPT_INVALID_LIGHT_INDEX;
        if( all(prevPixelPos >= 0.xx) && all(prevPixelPos < g_const.FeedbackResolution.xy) )
            lightIndex = u_processedFeedbackBuffer[prevPixelPos + uint2(0, indirect?g_const.FeedbackResolution.y:0)].x;

        // part 2:
        uint lightIndexHistoric = RTXPT_INVALID_LIGHT_INDEX;
        if( all(prevPixelPos >= 0.xx) && all(prevPixelPos < g_const.FeedbackResolution.xy) && controlInfo.LastFrameLocalSamplesAvailable )
        {
            uint2 tilePos = (prevPixelPos+controlInfo.LocalSamplingTileJitterPrev) / RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE;
            lightIndexHistoric = SampleLightLocalHistoric(tilePos, indirect, sampleGenerator);
            // if( lightIndex == RTXPT_INVALID_LIGHT_INDEX )
            //     lightIndex = SampleLightLocalHistoric(tilePos, indirect, sampleGenerator);
        }
        if( lightIndex == RTXPT_INVALID_LIGHT_INDEX )
            lightIndex = SampleLightGlobal(sampleGenerator);
        if( lightIndexHistoric == RTXPT_INVALID_LIGHT_INDEX )
            lightIndexHistoric = SampleLightGlobal(sampleGenerator); // historic can be invalid if there is no mapping (light no longer present)

        // uint2 tilePos = prevPixelPos / RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE;
        // if( all(tilePos == uint2(7,7)) )
        //     DebugPrint( "P:{0}, I1:{1}, I2:{2}", prevPixelPos, lightIndex, lightIndexHistoric );

        u_reprojectedFeedbackBuffer[pixelPos + uint2(0, indirect?g_const.FeedbackResolution.y:0)] = uint2( lightIndex, lightIndexHistoric );

        if( g_const.DebugDrawType == (int)LightingDebugViewType::FeedbackProcessed && g_const.DebugDrawDirect == !indirect )
            DebugPixel( pixelPos.xy, float4( ColorFromHash(Hash32(lightIndex)), 0.95) );
        if( g_const.DebugDrawType == (int)LightingDebugViewType::FeedbackHistoric && g_const.DebugDrawDirect == !indirect )
            DebugPixel( pixelPos.xy, float4( ColorFromHash(Hash32(lightIndexHistoric)), 0.95) );
    }

    if( dispatchThreadID.x < g_const.LRFeedbackResolution.x || dispatchThreadID.y < g_const.LRFeedbackResolution.y )
    {
        LightFeedbackReservoir reservoir = LightFeedbackReservoir::make();

        int2 blockTopLeft    = dispatchThreadID.xy * RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE;
        for( uint x = 0; x < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE; x++ )
            for( uint y = 0; y < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE; y++ )
            {
                uint2 pixelPos = blockTopLeft + uint2(x,y);

                float3 screenSpaceMotion = ConvertMotionVectorToPixelSpace( pixelPos, t_motionVectors[pixelPos] );
                int2 prevPixelPos = int2( float2(pixelPos) + screenSpaceMotion.xy + 0.5.xx );

                if( all(prevPixelPos >= 0.xx) && all(prevPixelPos < g_const.FeedbackResolution.xy) )
                {
                    LightFeedbackReservoir other = SampleFeedback( prevPixelPos, indirect );
                    reservoir.Add( sampleNext1D( sampleGenerator ), other.CandidateIndex, other.CandidateWeight );
                }
            }

        uint lightIndex = reservoir.CandidateIndex;
        if( lightIndex == RTXPT_INVALID_LIGHT_INDEX )
            lightIndex = SampleLightGlobal(sampleGenerator);

        u_reprojectedLRFeedbackBuffer[dispatchThreadID.xy + uint2(0, indirect?g_const.LRFeedbackResolution.y:0)] = lightIndex;

        if( g_const.DebugDrawType == (int)LightingDebugViewType::FeedbackLowRes && g_const.DebugDrawDirect == !indirect )
            for( int x = 0; x < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE; x++ )
                for( int y = 0; y < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE; y++ )
                 {
                     uint2 pixelPos = blockTopLeft + uint2(x,y);
                     DebugPixel( pixelPos.xy, float4( ColorFromHash(Hash32(lightIndex)), 0.95) );
                 }
    }
}

// This fills the local sampling tile buffer (u_narrowSamplingBuffer) with light indices - it does not find duplicates and sort 
void FillTile( uint2 tilePos, bool indirect )
{
    const LightingControlData controlInfo = u_controlBuffer[0];
    int margin = (RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE - RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE)/2;
    int2 cellTopLeft   = tilePos.xy * RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - (int2)controlInfo.LocalSamplingTileJitter;
    int2 windowTopLeft  = (int2)cellTopLeft - margin;
    int2 lrWindowTopLeft = int2( float2(cellTopLeft+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE*0.5) / RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE.xx - RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE*0.5 + 0.5.xx );

    uint currentlyCollectedCount = 0;

    bool debugTile = all( g_const.MouseCursorPos >= cellTopLeft ) && all((g_const.MouseCursorPos-cellTopLeft) < RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE.xx );

    uint2 actualTilePos = tilePos.xy + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0);  // support for handling of indirect as well

    for ( int x = 0; x < RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE; x++ )
        for ( int y = 0; y < RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE; y++ )
        {
            int2 pixelPos = windowTopLeft + int2(x,y);
            int2 srcCoord = MirrorCoord(pixelPos, g_const.FeedbackResolution);

            uint lightIndexA = u_reprojectedFeedbackBuffer[srcCoord + uint2(0, indirect?g_const.FeedbackResolution.y:0)].x;

            if ( lightIndexA == RTXPT_INVALID_LIGHT_INDEX )// || lightIndexB == RTXPT_INVALID_LIGHT_INDEX )
            {
                DebugPixel( cellTopLeft, float4(1,0,0,1) );
                DebugPrint("Bad light read from {0} - missing barrier or etc?", srcCoord);
            }

            u_narrowSamplingBuffer[ uint3(actualTilePos, currentlyCollectedCount++) ] = PackMiniListLightAndCount( lightIndexA, 1 );

#if RTXPT_LIGHTING_USE_ADDITIONAL_HISTORY_SAMPLES
            uint lightIndexH = u_reprojectedFeedbackBuffer[srcCoord + uint2(0, indirect?g_const.FeedbackResolution.y:0)].y;  // note, .y is historic samples!
            if ( lightIndexH == RTXPT_INVALID_LIGHT_INDEX )// || lightIndexB == RTXPT_INVALID_LIGHT_INDEX )
            {
                DebugPixel( cellTopLeft, float4(1,0,0,1) );
                DebugPrint("Bad historic light read from {0} - missing barrier or etc?", srcCoord);
            }
            u_narrowSamplingBuffer[ uint3(actualTilePos, currentlyCollectedCount++) ] = PackMiniListLightAndCount( lightIndexH, 1 );
#endif
        }

    for( int x = 0; x < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE; x++ )
        for( int y = 0; y < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE; y++ )
        {
            int2 lrPixelPos = lrWindowTopLeft + int2(x,y);
            int2 lrSrcCoord = MirrorCoord(lrPixelPos, g_const.LRFeedbackResolution);

            uint lightIndex = u_reprojectedLRFeedbackBuffer[lrSrcCoord + uint2(0, indirect?g_const.LRFeedbackResolution.y:0)].x;
            if( lightIndex == RTXPT_INVALID_LIGHT_INDEX )
                DebugPrint("LR bad light read from {0} - missing barrier or etc?", lrSrcCoord);

            u_narrowSamplingBuffer[ uint3(actualTilePos, currentlyCollectedCount++) ] = PackMiniListLightAndCount( lightIndex, 1 );
        }  

    SampleGenerator sampleGenerator = SampleGenerator::make( SampleGeneratorVertexBase::make( tilePos, 0, g_const.UpdateCounter ) );
    for ( int i = 0; i < RTXPT_LIGHTING_TOP_UP_SAMPLES; i++ )
    {
        uint ox = sampleGenerator.Next() % RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE;
        uint oy = sampleGenerator.Next() % RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE;

        int2 pixelPos = cellTopLeft + int2(ox,oy);
        int2 srcCoord = MirrorCoord(pixelPos, g_const.FeedbackResolution);

        uint lightIndex = u_reprojectedFeedbackBuffer[srcCoord + uint2(0, indirect?g_const.FeedbackResolution.y:0)].x;  // note, .y is historic samples!
        u_narrowSamplingBuffer[ uint3(actualTilePos, currentlyCollectedCount++) ] = PackMiniListLightAndCount( lightIndex, 1 );
    }

    if ( currentlyCollectedCount != RTXPT_LIGHTING_NARROW_PROXY_COUNT )// || lightIndexB == RTXPT_INVALID_LIGHT_INDEX )
    {
        DebugPixel( cellTopLeft, float4(1,0,0,1) );
        DebugPrint("Wrong number of lights in FillTile {0}, {1} - missing barrier or etc?", actualTilePos, currentlyCollectedCount);
    }

 }

[numthreads(8, 8, 1)]
void ProcessFeedbackHistoryP2( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const LightingControlData controlInfo = u_controlBuffer[0];
    uint2 tilePos = dispatchThreadID.xy;

    bool indirect = false; // prevent direct and indirect parts, which are stacked one on the other, bleeding into each other
#if RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER
    if( tilePos.y >= g_const.NarrowSamplingResolution.y )
    {
        indirect = true;
        tilePos.y -= g_const.NarrowSamplingResolution.y;
    }
#endif

    if( tilePos.x >= g_const.NarrowSamplingResolution.x || tilePos.y >= g_const.NarrowSamplingResolution.y )
        return;

    FillTile(tilePos, indirect);
}

[numthreads(8, 8, 1)]
void ProcessFeedbackHistoryP3a( uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadId )
{
    const LightingControlData controlInfo = u_controlBuffer[0];
    uint2 tilePos = dispatchThreadID.xy;
    if( tilePos.x >= g_const.NarrowSamplingResolution.x || tilePos.y >= g_const.NarrowSamplingResolution.y*2 )
        return;

    int2 cellTopLeft   = tilePos.xy * RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - (int2)controlInfo.LocalSamplingTileJitter;
    bool debugTile = all( g_const.MouseCursorPos >= cellTopLeft ) && all((g_const.MouseCursorPos-cellTopLeft) < RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE.xx );
    debugTile = false;

#if 0
    SortedLightList
    SortedLightLLRBTree<RTXPT_LIGHTING_NARROW_PROXY_COUNT> container = SortedLightLLRBTree<RTXPT_LIGHTING_NARROW_PROXY_COUNT>::empty();
#else
    HashBucketSortTable<RTXPT_LIGHTING_NARROW_PROXY_COUNT> container = HashBucketSortTable<RTXPT_LIGHTING_NARROW_PROXY_COUNT>::empty();
#endif

    for( uint i = 0; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
        container.InsertOrIncCounter( UnpackMiniListLight(u_narrowSamplingBuffer[ uint3(tilePos, i) ]), debugTile );

    int sampleCount = container.Store(u_narrowSamplingBuffer, tilePos, debugTile);
    if( sampleCount != RTXPT_LIGHTING_NARROW_PROXY_COUNT )
        DebugPrint("SampleCount wrong {0}", sampleCount);
}

/// *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE ***
groupshared uint g_lowestLightIndexInGroup;
groupshared uint g_numMatchingThreadsInGroup;
groupshared uint g_tupleCount;

#if 0
[numthreads(RTXPT_LIGHTING_NARROW_PROXY_COUNT, 1, 1)]
void ProcessFeedbackHistoryP3b( uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadId )
{
    uint indexInTile = dispatchThreadID.x;
    uint2 tileCoord = dispatchThreadID.yz;

#if 1
    bool dbgPrint = dispatchThreadID.y == 0 && dispatchThreadID.z == 0;
    
    uint myLightIndex = UnpackMiniListLight( u_narrowSamplingBuffer[ uint3(tileCoord, indexInTile) ] );

    // if (dbgPrint) DebugPrint( "light index at {0} is {1}", indexInTile, myLightIndex );

    // Initialise group shared variables and make sure all warps are synchronised
    g_lowestLightIndexInGroup = 0xffffffff;
    g_numMatchingThreadsInGroup = 0;
    g_tupleCount = 0;
    // We keep all threads active, so this remains true
    bool waveIsFirstLane = WaveIsFirstLane();
    // This should be faster, because we know that all the threads are active, so we should
    // be able to simply mask dispatchThreadID.x on the wave size.
    // But it's not.  Don't know why.
    //bool waveIsFirstLane = (dispatchThreadID.x & (WaveGetLaneCount() - 1)) == 0;
    //bool waveIsFirstLane = (dispatchThreadID.x & (32 - 1)) == 0;
    bool isFirstThreadInGroup = (dispatchThreadID.x == 0);

    GroupMemoryBarrierWithGroupSync();
 
    //
    // This loop continues while there are still unique light indices.
    // (One iteration per unique light)
    //
    //int loopCounter = 0;
    while (true)
    {
        //
        // Find the lowest value of myLightIndex across all threads in the group
        //
        uint lowestLightIndexInWave = WaveActiveMin(myLightIndex);
        if (waveIsFirstLane)
        {
            InterlockedMin(g_lowestLightIndexInGroup, lowestLightIndexInWave);
        }
        GroupMemoryBarrierWithGroupSync(); // Wait for all waves to contribute to g_lowestLightIndexInGroup

        // if (dbgPrint /*&& myLightIndex != 0xffffffff*/ && dispatchThreadID.x == 0 ) DebugPrint( "min found at loop counter {0} is {1}", loopCounter, g_lowestLightIndexInGroup );

        if (g_lowestLightIndexInGroup == 0xffffffff)
        {
            // if (dbgPrint) DebugPrint( "exit in the loop it {0}", loopCounter );

            // If that's the lowest index, then it means that we've been through
            // all of the unique light indices, so we've finished.
            break;
        }

        //
        // Count the threads across the whole thread group that share this light index
        //
        uint lowestLightIndexInGroup = g_lowestLightIndexInGroup; // Take a local copy
        if (isFirstThreadInGroup)
        {
            g_numMatchingThreadsInGroup = 0; // Zero for upcoming interlocked add
        }
        bool thisThreadMatches = (myLightIndex == lowestLightIndexInGroup);
        uint numMatchingThreadsInWave = WaveActiveCountBits(thisThreadMatches);
        GroupMemoryBarrierWithGroupSync(); // Wait for g_numMatchingThreadsInGroup to be zero
        //while (g_numMatchingThreadsInGroup != 0); // Spin. (This doesn't work BTW)
        if (waveIsFirstLane && numMatchingThreadsInWave)
        {
            g_lowestLightIndexInGroup = 0xffffffff; // Reset for next loop iteration (as long as one thread in the group does this, we're ok)
            // Add up the wave totals across the whole thread group
            InterlockedAdd(g_numMatchingThreadsInGroup, numMatchingThreadsInWave); //< Count must have been zeroed previously
        }
        GroupMemoryBarrierWithGroupSync(); // Wait for all waves to add to g_numMatchingThreadsInGroup
        
        //if (dbgPrint && thisThreadMatches ) DebugPrint( "thread {0} is {1}, {2}", dispatchThreadID.x, lowestLightIndexInGroup, g_numMatchingThreadsInGroup );
        //[branch] // numMatchingThreadsInWave is uniform across the wave
        if(numMatchingThreadsInWave)
        {
            //
            // Write out the tuples for this light index.
            // One for each thread that matches lowestLightIndexInGroup.
            //
            uint tupleIdxWaveBase;
            if (waveIsFirstLane)
            {
                // Allocate space in the tuple buffer for the number of threads in this
                // wave that will write to it.
                // NB: tupleIdxWaveBase will only be valid for the first thread in the wave
                //uint equalCount = WaveActiveCountBits(thisThreadMatches);
                InterlockedAdd(g_tupleCount, numMatchingThreadsInWave, tupleIdxWaveBase);

                // if (dbgPrint)  DebugPrint( "thread {0}, lindex {1}, count {2}", dispatchThreadID.x, lowestLightIndexInGroup, numMatchingThreadsInWave );
            }

            // Calculate the location in the buffer that this thread should write its tuple
            //tupleIdxWaveBase = WaveReadLaneFirst(tupleIdxWaveBase);
            tupleIdxWaveBase = WaveReadLaneAt(tupleIdxWaveBase, 0);
            uint idx = tupleIdxWaveBase + WavePrefixCountBits(thisThreadMatches);

            if (thisThreadMatches)
            {
#if 0 // validate-assert
                if (u_narrowSamplingBuffer[uint3(tileCoord.xy, idx)] != PackMiniListLightAndCount(lowestLightIndexInGroup, g_numMatchingThreadsInGroup) )
                    DebugPrint( "Validation error, tile {0}, pos {1}", tileCoord, idx );
#else
                // Write the tuple
                u_narrowSamplingBuffer[uint3(tileCoord.xy, idx)] = PackMiniListLightAndCount(lowestLightIndexInGroup, g_numMatchingThreadsInGroup);
#endif
                // Now we effectively kill this thread because its light index has been handled.
                // But we do it without actually killing the thread because we need all the
                // waves to stay in the same flow control in order to be able
                // to use GroupMemoryBarrierWithGroupSync
                myLightIndex = 0xffffffff;

                //if (dbgPrint) DebugPrint( "thread {0} base {1} outIndex {2} tuple {3}", dispatchThreadID.x, tupleIdxWaveBase, idx, tuple );
            }
        }

        //if (dbgPrint && WaveIsFirstLane())
        //{
        //    GroupMemoryBarrierWithGroupSync(); // not needed
        //    DebugPrint( "LOOP {0} FINISHED, counter: {1}", loopCounter, g_tupleCount );
        // }

        //loopCounter++;
    }
#endif
}

#else

#define WAVE_LANE_COUNT 32
#define NUM_WAVES_IN_A_GROUP 4

[numthreads(WAVE_LANE_COUNT, NUM_WAVES_IN_A_GROUP, 1)]
void ProcessFeedbackHistoryP3b(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadId)
{
    uint2 tileCoord = dispatchThreadID.yz;

    bool dbgPrint = dispatchThreadID.y == 0 && dispatchThreadID.z == 0;
    
    const uint kNumSamplesPerThread = RTXPT_LIGHTING_NARROW_PROXY_COUNT / WAVE_LANE_COUNT;
    // Load the samples that this thread will deal with
    uint threadLocalSamples[kNumSamplesPerThread];
    for (uint i = 0; i < kNumSamplesPerThread; ++i)
    {
        uint indexInTile = (WAVE_LANE_COUNT * i) + dispatchThreadID.x;
        //uint indexInTile = (dispatchThreadID.x * kNumSamplesPerThread) + i; //< This pattern is ultimately slower
        threadLocalSamples[i] = UnpackMiniListLight(u_narrowSamplingBuffer[uint3(tileCoord, indexInTile)]);
    }

    //
    // This loop continues while there are still unique light indices.
    // (One iteration per unique light)
    //
    int loopCounter = 0;
    uint tupleIdx = 0;
    uint activeSamplesInThread = (1u << kNumSamplesPerThread) - 1;
    while (activeSamplesInThread)
    {
        //
        // Find the lowest light sample index across the whole tile
        //
        uint lowestLightIndexInThread = 0xffffffff;
        for (uint i = 0; i < kNumSamplesPerThread; ++i)
        {
            if (activeSamplesInThread & (1u << i))
                lowestLightIndexInThread = min(lowestLightIndexInThread, threadLocalSamples[i]);
        }
        uint lowestLightIndexInTile = WaveActiveMin(lowestLightIndexInThread);

        //if (dbgPrint) DebugPrint("min found at loop counter {0} is {1}, {2}", loopCounter, lowestLightIndexInThread, lowestLightIndexInTile);

        //
        // Count the number of samples across the whole tile that share this light index
        //
        uint numMatchingSamplesInThread = 0;
        const uint activeSamplesInWave = WaveActiveBitOr(activeSamplesInThread);
        for (uint i = 0; i < kNumSamplesPerThread; ++i)
        {
            if (activeSamplesInWave & (1u << i))
            {
                numMatchingSamplesInThread += (threadLocalSamples[i] == lowestLightIndexInTile) ? 1 : 0;
            }
        }
        uint numMatchingSamplesInTile = WaveActiveSum(numMatchingSamplesInThread);

        //
        // Write out the tuples for this light index.
        // One for each thread sample that matches.
        //
        uint tuple = PackMiniListLightAndCount(lowestLightIndexInTile, numMatchingSamplesInTile);
        //uint threadTupleIdx = tupleIdx + WavePrefixSum(numMatchingSamplesInThread); // This approach is strangely slower
        //tupleIdx += numMatchingSamplesInTile;
        for (uint i = 0; i < kNumSamplesPerThread; ++i)
        {
            if (activeSamplesInWave & (1u << i))
            {
                bool match = threadLocalSamples[i] == lowestLightIndexInTile;
                if (match)
                {
                    // Write the tuple
                    uint threadTupleIdx = tupleIdx + WavePrefixCountBits(true);
                    u_narrowSamplingBuffer[uint3(tileCoord.xy, threadTupleIdx)] = tuple;
                    // Kill this light index sample so we don't write it again
                    //threadLocalSamples[i] = 0xffffffff;
                    activeSamplesInThread &= ~(1u << i);
                }
                tupleIdx += WaveActiveCountBits(match);
            }
        }

        loopCounter++;
    }

    // note: full validation is in ProcessFeedbackHistoryDebugViz
}

#endif

// _bucketSize is essential, things don't work otherwise
template <uint _bucketSize> void DescendingSortInsertOrIncCounter(inout uint elements[_bucketSize], inout uint elementCount, uint newLightTuple)
{
    uint insertLocation = 0;

    uint newLightIndex = UnpackMiniListLight(newLightTuple);

#if 0 // linear search
    for( int i = elementCount-1; i >= 0; i-- )
    {
        uint iLight = UnpackMiniListLight(elements[i]);
        if( iLight == newLightIndex)
        {
            elements[i] += UnpackMiniListCount(newLightTuple);   // no overflow checking, assume counter bits portion is enough
            return;// true;
        }
        if( iLight > newLightIndex )
        {
            insertLocation = i+1;
            break;
        }
    }
#else // binary search
    int indexLeft = 0; 
    int indexRight = elementCount-1;
    while( indexLeft <= indexRight )
    {
        int indexMiddle = (indexLeft+indexRight)/2;
        const uint eMiddle = elements[indexMiddle];
        uint iLightMiddle = UnpackMiniListLight(eMiddle);

        if( iLightMiddle > newLightIndex )
            indexLeft = indexMiddle+1;
        else if( iLightMiddle < newLightIndex )
            indexRight = indexMiddle-1;
        else
        {
            elements[indexMiddle] = eMiddle + UnpackMiniListCount(newLightTuple);   // no overflow checking, implicit packing
            return; // true;
        }
    }
    insertLocation = indexLeft;
#endif

    // // do we have more space in the list? if not - we have to indicate we couldn't store this one
    // if( elementCount == _MaxSize )
    //     return false; // overflow!

    // make space (shift everything by 1 to the right)
    for( uint i = elementCount; i > insertLocation; i-- )
        elements[i] = elements[i-1];
            
    // and finally, insert 
    elements[insertLocation] = newLightTuple;
    elementCount = elementCount+1;
}

// ************************************************************************************************************************************************************
// Parts of below bitonic sort code is originally from https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/Bitonic32PreSortCS.hlsl 
// Enclosed license: 
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author:  James Stanard 
// ************************************************************************************************************************************************************
// Takes Value and widens it by one bit at the location of the bit in the mask.  A one is inserted in the space.  OneBitMask must have one and only one bit set.
uint InsertOneBit( uint Value, uint OneBitMask )
{
    uint Mask = OneBitMask - 1;
    return (Value & ~Mask) << 1 | (Value & Mask) | OneBitMask;
}
//
// For simplest variant of bitonic sort, list must be power of two - so enforce it
STATIC_ASSERT( ( RTXPT_LIGHTING_NARROW_PROXY_COUNT & ( RTXPT_LIGHTING_NARROW_PROXY_COUNT - 1 ) ) == 0 );
groupshared uint g_localData[RTXPT_LIGHTING_NARROW_PROXY_COUNT];
groupshared uint g_localDataRangeLR[RTXPT_LIGHTING_NARROW_PROXY_COUNT];
//
void LastScanAndWriteOut( uint2 tileCoord, uint loc )
{
    uint lightIndex = g_localData[loc];
    uint indexL = g_localDataRangeLR[loc] >> 16;
    uint indexR = g_localDataRangeLR[loc] & 0xFFFF;
    while( true )
    {
        uint nextIndexL = g_localDataRangeLR[indexL] >> 16;
        uint nextIndexR = g_localDataRangeLR[indexR] & 0xFFFF;
        if (nextIndexL == indexL && nextIndexR == indexR)
            break;
        indexL = nextIndexL;
        indexR = nextIndexR;
    }
    uint count = indexR-indexL+1;
    u_narrowSamplingBuffer[ uint3(tileCoord, loc) ] = PackMiniListLightAndCount(lightIndex, count);
}
//
[numthreads(RTXPT_LIGHTING_NARROW_PROXY_COUNT/2, 1, 1)] // <- we need number of threads that is half of the list size - so we could support up to 2048
void ProcessFeedbackHistoryP3c(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadId)
{
    const uint threadID = groupThreadID.x;
    uint2 tileCoord = groupID.xy;

    // Stage 1: Load to local storage
    g_localData[threadID] = UnpackMiniListLight(u_narrowSamplingBuffer[uint3(tileCoord, threadID)]);
    g_localData[threadID + RTXPT_LIGHTING_NARROW_PROXY_COUNT/2] = UnpackMiniListLight(u_narrowSamplingBuffer[uint3(tileCoord, threadID + RTXPT_LIGHTING_NARROW_PROXY_COUNT/2)]);

    GroupMemoryBarrierWithGroupSync();

    // Stage 2: bitonic sort
    // This is better unrolled because it reduces ALU and because some architectures can load/store two LDS items in a single instruction as long as their separation is a compile-time constant.
    [unroll] for (uint k = 2; k <= RTXPT_LIGHTING_NARROW_PROXY_COUNT; k <<= 1)
    {
        //[unroll]
        for (uint j = k / 2; j > 0; j /= 2)
        {
            uint Index2 = InsertOneBit(threadID, j);
            uint Index1 = Index2 ^ (k == 2 * j ? k - 1 : j);

            uint A = g_localData[Index1];
            uint B = g_localData[Index2];

            if (A>B)
            {
                // Swap the keys
                g_localData[Index1] = B;
                g_localData[Index2] = A;
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Stage 3a: count duplicates - first pass
    {
        uint indexA = threadID*2;
        uint indexB = indexA+1;
        uint valA = g_localData[indexA];
        uint valB = g_localData[indexB];
        
        int indexL = indexA;
        int indexR = indexB;
        for (uint i = 0; i < 4; i++)    // the bigger the number of steps here, the faster the second pass is (small step is good for all different values in the array, large step is good when there's few but long ones)
        {
            indexL = indexL-1;
            indexR = indexR+1;
            if (indexL == -1 || g_localData[indexL] != valA)    
                indexL++;   // if beyond the left end or not same, go back one step
            if (indexR == RTXPT_LIGHTING_NARROW_PROXY_COUNT || g_localData[indexR] != valB)    
                indexR--;   // if beyond the right end or not the same, go step one back
        }

        // write out 
        g_localDataRangeLR[indexA] = (indexL << 16) | ((valA == valB)?(indexR):(indexA));
        g_localDataRangeLR[indexB] = (((valA == valB)?(indexL):(indexB)) << 16) | (indexR);
    }

    GroupMemoryBarrierWithGroupSync();

    // Stage 3b: complete duplicate count and write out
    {
        LastScanAndWriteOut( tileCoord, threadID );
        LastScanAndWriteOut( tileCoord, threadID + RTXPT_LIGHTING_NARROW_PROXY_COUNT/2 );
    }

    // debug print
    // if (threadID == 0)
    // {
    //     for( int i = 0; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
    //     {
    //         uint indexL = g_localDataRangeLR[i] >> 16;
    //         uint indexR = g_localDataRangeLR[i] & 0xFFFF;
    //         uint count = indexR-indexL+1;
    // 
    //         DebugPrint("", i, UnpackMiniListLight(g_localData[i]), UnpackMiniListCount(g_localData[i]), indexL, indexR, count);
    //     }
    // }


    #if 0
//    if (threadID == 0)
//        g_validationCounter = 0;
//    g_validationCounters[threadID] = 0;
//    GroupMemoryBarrierWithGroupSync();


    const uint kTotalSamples = RTXPT_LIGHTING_NARROW_PROXY_COUNT;
    const uint kBucketCount = 32;
    const uint kBucketSize = 32;

    uint threadLocalSamples[kBucketSize];
    uint threadLocalSampleCount = 0;

    bool overfilled = false;

    uint firstPassCounter = 0;
    while (firstPassCounter < kTotalSamples)
    {
        // first find 1 that matches our hash; overflow also goes here
        uint acceptedTuple = RTXPT_INVALID_LIGHT_INDEX;
        while (firstPassCounter < kTotalSamples)
        {
            uint candidateTuple = u_narrowSamplingBuffer[uint3(tileCoord, firstPassCounter)];
            firstPassCounter++;
            uint lightIndex = UnpackMiniListLight(candidateTuple);

            uint bucketIndex = Hash32(lightIndex+0x9e3779b9) % 32;
            // DebugPrint("", threadID, bucketIndex);

            if (bucketIndex == threadID)
            {
                acceptedTuple = candidateTuple;
                // InterlockedAdd(g_validationCounters[threadID], 1);
                // InterlockedAdd(g_validationCounter, 1);
                break;
            }
        }

        // no more, we can get out
        if (acceptedTuple != RTXPT_INVALID_LIGHT_INDEX)
        {
            // error with inputs - missing barrier?
            // if (UnpackMiniListCount(acceptedTuple)!=1)
            //     DebugPrint("", threadID, 0.42, 0.42, 0.42, 0.42, 0.42, UnpackMiniListCount(acceptedTuple));
            DescendingSortInsertOrIncCounter<kBucketSize>(threadLocalSamples, threadLocalSampleCount, acceptedTuple);

            uint overfillElement = RTXPT_INVALID_LIGHT_INDEX;
            if (threadLocalSampleCount == kBucketSize) // if overfileld (last empty space filled), pop it off the list and put it on the queue
            {
                overfillElement = threadLocalSamples[threadLocalSampleCount-1];
                threadLocalSampleCount--;
                overfilled = true;
                DebugPrint("", threadID, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42);
                break;
            }
        }
        // add loop here 
        // DebugPrint("", threadID, accepted);
    }

    //DebugPrint("", threadID, threadLocalSampleCount, (uint)overfilled);

    GroupMemoryBarrierWithGroupSync();  // we'll be re-writing data in u_narrowSamplingBuffer[uint3(tileCoord, ...)] now

    // if (threadID == 0)
    //     DebugPrint();

    // for (int printThreadID = 0; printThreadID < 32; printThreadID++)
    // {
    //     GroupMemoryBarrierWithGroupSync();
    // 
    //     if (printThreadID == threadID)
    //     {
    //         DebugPrint();
    //         for ( int i = 0; i < threadLocalSampleCount; i++ )
    //             DebugPrint("", threadID, i, (uint)UnpackMiniListLight(threadLocalSamples[i]), (uint)UnpackMiniListCount(threadLocalSamples[i]));
    //         DebugPrint();
    //     }
    // }
    // 
    // if (threadID == 0)
    //     DebugPrint();

    uint collectedSoFarTotal = 0;

    // Stage 2 - keep popping the next smallest from all 32 threads until we're done
    uint mySmallestToken = RTXPT_INVALID_LIGHT_INDEX;
    while(true)
    {
        // if mySmallestToken is 'empty' and our list isn't empty, pop it.
        if (mySmallestToken == RTXPT_INVALID_LIGHT_INDEX && threadLocalSampleCount > 0)
            mySmallestToken = threadLocalSamples[--threadLocalSampleCount];

        uint globalSmallestToken = WaveActiveMin(mySmallestToken);

        // check if we're done
        if (globalSmallestToken == RTXPT_INVALID_LIGHT_INDEX)
            break;

        // empty "mine"
        if (globalSmallestToken == mySmallestToken)
            mySmallestToken = RTXPT_INVALID_LIGHT_INDEX;

        uint outIndex = UnpackMiniListLight(globalSmallestToken);
        uint outCount = UnpackMiniListCount(globalSmallestToken);
        //collectedSoFarTotal += outCount;
        
        // output (naive)
        if (threadID == 0)
        {
            // DebugPrint("", outIndex, outCount);
            for( int i = 0; i < outCount; i++ )
                u_narrowSamplingBuffer[uint3(tileCoord, collectedSoFarTotal++)] = globalSmallestToken;
        }
    }

    if (threadID == 0 && (collectedSoFarTotal != 128))
    {
        DebugPrint("", collectedSoFarTotal, kTotalSamples);
        DebugPrint();
        //DebugPrint("", g_validationCounter);
        DebugPrint();      
        DebugPrint();      
    }
    // GroupMemoryBarrierWithGroupSync();
    // DebugPrint("", threadID, g_validationCounters[threadID]);
    #endif
}
/// *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE ***


[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void DebugDrawLights( uint dispatchThreadID : SV_DispatchThreadID )
{
    if( dispatchThreadID >= g_const.TotalLightCount )
        return;

    uint lightIndex = dispatchThreadID;
    PolymorphicLightInfoFull light = LoadLight( lightIndex );

    const float alpha = 0.8;
    DebugDrawLight(light, alpha);
}


[numthreads(8, 8, 1)]
void ProcessFeedbackHistoryDebugViz( uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadId )
{
    const LightingControlData controlInfo = u_controlBuffer[0];
    uint2 tilePos = dispatchThreadID.xy;

    bool indirect = false; // prevent direct and indirect parts, which are stacked one on the other, bleeding into each other
#if RTXPT_LIGHTING_NEEAT_ENABLE_INDIRECT_LOCAL_LAYER
    if( tilePos.y >= g_const.NarrowSamplingResolution.y )
    {
        indirect = true;
        tilePos.y -= g_const.NarrowSamplingResolution.y;
    }
#endif

    if( any(tilePos >= g_const.NarrowSamplingResolution) )
        return;
    int margin = (RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE - RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE)/2;
    int2 cellTopLeft   = tilePos.xy * RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - (int2)controlInfo.LocalSamplingTileJitter;
    int2 windowTopLeft  = (int2)cellTopLeft - margin;
    int2 lrWindowTopLeft = int2( float2(cellTopLeft+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE*0.5) / RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE.xx - RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE*0.5 + 0.5.xx );

    bool debugTile = all( g_const.MouseCursorPos >= cellTopLeft ) && all((g_const.MouseCursorPos-cellTopLeft) < RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE.xx );
    debugTile &= g_const.DebugDrawDirect == !indirect;
    if( debugTile && g_const.DebugDrawTileLights && g_const.DebugDrawDirect == !indirect )
    {
        const float maxCount = RTXPT_LIGHTING_NARROW_PROXY_COUNT;
        for( int i = 0; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
        {
            uint counter;
            uint lightIndex;
            UnpackMiniListLightAndCount( u_narrowSamplingBuffer[ uint3(tilePos.xy + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0), i) ], lightIndex, counter );

            PolymorphicLightInfoFull light = LoadLight( lightIndex );
            float3 lightPos = light.Base.Center;

            // float3 lightDirToCamera = lightPos - controlInfo.SceneCameraPos.xyz;
            // float dist = length(lightDirToCamera);
            // float size = dist * 0.015;
            // 
            // float3 norm = normalize(lightDirToCamera);
            // float3 tang, bitang;
            // BranchlessONB(norm, tang, bitang);

            float alpha = min( 1.0, float(counter)/maxCount + 0.15 );

            DebugDrawLight(light, alpha);

            DebugLine( float3(windowTopLeft + RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE.xx / 2, FLT_MAX), lightPos, float4(1.0-indirect, 1.0, indirect, 0.05) );
        }
    }

    if( g_const.DebugDrawDirect == !indirect )
    {
        float3 heatmapCol = float3(1-indirect,indirect,0);
        if (g_const.DebugDrawType == (int)LightingDebugViewType::TileHeatmap )
        {
            SortedLightList<RTXPT_LIGHTING_NARROW_PROXY_COUNT> localList = SortedLightList<RTXPT_LIGHTING_NARROW_PROXY_COUNT>::empty();
            for( uint i = 0; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
                localList.InsertOrIncCounter( UnpackMiniListLight(u_narrowSamplingBuffer[ uint3(tilePos + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0), i) ]) );

            int numberOfDifferentLights = localList.Count;
            heatmapCol = GradientHeatMap( (float(numberOfDifferentLights) / float(RTXPT_LIGHTING_NARROW_PROXY_COUNT))*1.2f );
        }

        for( int x = 0; x < RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE; x++ )
            for( int y = 0; y < RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE; y++ )
            {
                int2 pixelPos = windowTopLeft + int2(x,y);
                if( any(pixelPos<int2(0,0)) || any(pixelPos>=int2(g_const.FeedbackResolution.x,g_const.FeedbackResolution.y)) )
                    continue;
                bool insideCell = all( pixelPos >= cellTopLeft ) && all( (pixelPos - cellTopLeft) < RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE.xx );
                if( debugTile )
                    DebugPixel( pixelPos, float4( insideCell, 1-indirect, indirect, 1 ) );
                if( !debugTile && (x == margin || y == margin) )
                    DebugPixel( pixelPos, float4( 0, 0, 1, 0.15 ) );

                if (g_const.DebugDrawType == (int)LightingDebugViewType::TileHeatmap )
                    DebugPixel( pixelPos, float4(heatmapCol, 0.95) );
            }

        for( int x = 0; x < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE; x++ )
            for( int y = 0; y < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE; y++ )
            {
                int2 lrPixelPos = lrWindowTopLeft + int2(x,y);
                int2 lrSrcCoord = MirrorCoord(lrPixelPos, g_const.LRFeedbackResolution);

                if( debugTile && (x == 0 || y == 0) )
                    DebugPixel( lrSrcCoord*RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE, float4( 0, 1, 1, 1.0 ) );
                if( debugTile && ( ((x == (RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE-1)) || (y == (RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE-1) ) ) ) )
                    DebugPixel( (lrSrcCoord+1)*RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE.xx - 1.xx, float4( 0, 1, 1, 1.0 ) );
            }
    }


    if (g_const.DebugDrawType == (int)LightingDebugViewType::ValidateCorrectness)
    {
        uint dataToValidate[RTXPT_LIGHTING_NARROW_PROXY_COUNT];
        for( int i = 0 ; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
            dataToValidate[i] = u_narrowSamplingBuffer[ uint3(tilePos + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0), i) ];

        FillTile(tilePos, indirect);  // this fills u_narrowSamplingBuffer from scratch

        SortedLightList<RTXPT_LIGHTING_NARROW_PROXY_COUNT> localList = SortedLightList<RTXPT_LIGHTING_NARROW_PROXY_COUNT>::empty();
        for( uint i = 0; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
            localList.InsertOrIncCounter( UnpackMiniListLight(u_narrowSamplingBuffer[ uint3(tilePos + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0), i) ]) );
        bool allGood = localList.Validate(dataToValidate, debugTile);

        for( int x = 0; x < RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE; x++ )
            for( int y = 0; y < RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE; y++ )
            {
                int2 pixelPos = windowTopLeft + int2(x,y);
                if( !allGood )
                    DebugPixel( pixelPos, float4(1,0,0,1) );
                else
                    DebugPixel( pixelPos, float4(0,0.5,0,0.9) );
            }

        // put back original data in
        for( int i = 0 ; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
            u_narrowSamplingBuffer[ uint3(tilePos + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0), i) ] = dataToValidate[i];
    }

#if 0 // validate remapping - note, this validation doesn't work correctly if history<->current mapping isn't 1:1 which can happen with env quad trees if they get rebuilt differently due to different content
    {   // start from current to past
        uint historicIndex = u_historyRemapCurrentToPast[lightIndex];
        if(historicIndex != RTXPT_INVALID_LIGHT_INDEX)
        {
            if( historicIndex >= controlInfo.HistoricTotalLightCount )
                DebugPrint( "1 - out of range at lightIndex {0}, historicIndex {1}", lightIndex, historicIndex );

            uint recoveredCurrent = u_historyRemapPastToCurrent[historicIndex];
            if( recoveredCurrent != lightIndex )
                DebugPrint( "1 - wrong at lightIndex {0}, historicIndex {1}", lightIndex, historicIndex );
        }
    }
    {   // start from past to current
        uint recoveredCurrent = u_historyRemapPastToCurrent[lightIndex];
        if(recoveredCurrent != RTXPT_INVALID_LIGHT_INDEX)
        {
            if( recoveredCurrent >= lightCount )
                DebugPrint( "2 - out of range at lightIndex {0}, recoveredCurrent {1}", lightIndex, recoveredCurrent );

            uint recoveredHistoric = u_historyRemapCurrentToPast[recoveredCurrent];
            if( recoveredHistoric != lightIndex )
                DebugPrint( "2 - wrong at lightIndex {0}, recoveredCurrent {1}", lightIndex, recoveredCurrent );
        }
    }
#endif
}

void DebugDrawLight(const PolymorphicLightInfoFull lightInfo, float alpha)
{
    float3 radiance = PolymorphicLight::UnpackColor(lightInfo.Base);

    float4 color = float4( /*Reinhard(radiance)*/ColorFromHash(lightInfo.Extended.UniqueID), alpha );

    float maxR = max(radiance.x, max(radiance.y, radiance.z));
    float lineBrightness = 1.0;
    if( maxR < 1e-7f )
    {
        alpha *= 0.6;
        color = float4(0.03, 0, 0.03, alpha);
        lineBrightness = 0.06;
    }

    switch (PolymorphicLight::DecodeType(lightInfo))
    {
#if POLYLIGHT_SPHERE_ENABLE
    case PolymorphicLightType::kSphere:         DebugDrawLightSphere(lightInfo,         color, float4(0, lineBrightness*0.2, lineBrightness, alpha) ); break;
#endif
#if POLYLIGHT_POINT_ENABLE
    case PolymorphicLightType::kPoint:          DebugDrawLightPoint(lightInfo,          color, float4(0, lineBrightness*0.2, lineBrightness, alpha) ); break;
#endif
#if POLYLIGHT_TRIANGLE_ENABLE
    case PolymorphicLightType::kTriangle:       DebugDrawLightTriangle(lightInfo,       color, float4(0, lineBrightness, 0, alpha) ); break;
#endif
#if POLYLIGHT_DIRECTIONAL_ENABLE
    case PolymorphicLightType::kDirectional:    DebugDrawLightDirectional(lightInfo,    color, float4(0, lineBrightness*0.2, lineBrightness, alpha) ); break;
#endif
#if POLYLIGHT_ENV_ENABLE
    case PolymorphicLightType::kEnvironment:    DebugDrawLightEnvironment(lightInfo,    color, float4(lineBrightness, 0, 0, alpha) ); break;
#endif
#if POLYLIGHT_QT_ENV_ENABLE
    case PolymorphicLightType::kEnvironmentQuad:DebugDrawLightEnvironmentQuad(lightInfo,color, float4(lineBrightness, 0, 0, alpha) ); break;
#endif
    default: break;
    }
}

void DebugDrawLightSphere(in const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor)
{
    SphereLight light = SphereLight::Create(lightInfo);

    DebugSphere( light.position, light.radius, color, lineColor );

    if( light.shaping.isSpot )
        DebugLine( light.position, light.position + light.shaping.primaryAxis * light.radius * 2, color );
}

void DebugDrawLightPoint(in const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor)   {}

void DebugDrawLightTriangle(in const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor)
{
    TriangleLight light = TriangleLight::Create(lightInfo);

    float3 a = light.base;
    float3 b = light.base+light.edge1;
    float3 c = light.base+light.edge2;

    DebugTriangle( a, b, c, color );

    DebugLine( a, b, lineColor ); 
    DebugLine( b, c, lineColor ); 
    DebugLine( c, a, lineColor ); 

}

void DebugDrawLightEnvironmentQuad(in const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor)
{
    EnvironmentQuadLight light = EnvironmentQuadLight::Create(lightInfo);

    float2 subTexelPosTL = float2( ((float)light.NodeX+0) / (float)light.NodeDim, ((float)light.NodeY+0) / (float)light.NodeDim );
    float2 subTexelPosTR = float2( ((float)light.NodeX+1) / (float)light.NodeDim, ((float)light.NodeY+0) / (float)light.NodeDim );
    float2 subTexelPosBL = float2( ((float)light.NodeX+0) / (float)light.NodeDim, ((float)light.NodeY+1) / (float)light.NodeDim );
    float2 subTexelPosBR = float2( ((float)light.NodeX+1) / (float)light.NodeDim, ((float)light.NodeY+1) / (float)light.NodeDim );

    float range = DISTANT_LIGHT_DISTANCE;
    float3 tl = EnvironmentQuadLight::ToWorld(oct_to_ndir_equal_area_unorm( subTexelPosTL )) * range;
    float3 tr = EnvironmentQuadLight::ToWorld(oct_to_ndir_equal_area_unorm( subTexelPosTR )) * range;
    float3 bl = EnvironmentQuadLight::ToWorld(oct_to_ndir_equal_area_unorm( subTexelPosBL )) * range;
    float3 br = EnvironmentQuadLight::ToWorld(oct_to_ndir_equal_area_unorm( subTexelPosBR )) * range;

    // color = float4( Reinhard( light.Weight.xxx * 0.1 ), 0.5 );

#if 0 // full tessellated sphere - a bit too much for usefulness
    if( length(tl-br) > length(tr-bl) )
    {
        if ( color.a > 0 )
        {
            DebugTriangle( tl, tr, bl, color ); 
            DebugTriangle( tr, br, bl, color );
        }
        if ( lineColor.a > 0 )
        {
            DebugLine( tl, tr, lineColor ); 
            DebugLine( tr, bl, lineColor ); 
            DebugLine( tl, bl, lineColor ); 
            DebugLine( tr, br, lineColor ); 
        }
    }
    else
    {
        if ( color.a > 0 )
        {
            DebugTriangle( tl, br, bl, color ); 
            DebugTriangle( tl, br, tr, color );
        }
        if ( lineColor.a > 0 )
        {
            DebugLine( tl, br, lineColor ); 
            DebugLine( br, bl, lineColor ); 
            DebugLine( tl, bl, lineColor ); 
            DebugLine( tl, tr, lineColor ); 
        }
    }
#else
    DebugLine( tl, tr, lineColor ); 
    DebugLine( tl, bl, lineColor ); 
    DebugLine( br, tr, lineColor ); 
    DebugLine( br, bl, lineColor ); 
#endif

}

void DebugDrawLightDirectional(in const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor) {}
void DebugDrawLightEnvironment(in const PolymorphicLightInfoFull lightInfo, float4 color, float4 lineColor) {}


#endif // #if !defined(__cplusplus)

#endif // #ifndef __LIGHTS_BAKER_HLSL__