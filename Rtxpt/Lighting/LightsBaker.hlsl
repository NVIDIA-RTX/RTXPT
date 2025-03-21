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

#define LLB_SCRATCH_BUFFER_SIZE         (24*1024*1024)

#define RTXPT_LIGHTING_CPJ_BLOCKSIZE    1024

#define LLB_MAX_TRIANGLES_PER_TASK      16
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

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>

#include "../SubInstanceData.h"
#include "../PathTracer/Materials/MaterialPT.h"

#include "../ShaderDebug.hlsli"
#include "../PathTracer/Utils/Math/MathHelpers.hlsli"
#include "../PathTracer/Lighting/LightingTypes.h"
#include "../PathTracer/Lighting/LightingConfig.h"
#include "../PathTracer/Lighting/PolymorphicLight.hlsli"
#include "../PathTracer/Lighting/LightingAlgorithms.hlsli"
#include "../PathTracer/Utils/NoiseAndSequences.hlsli"
#include "../PathTracer/SampleGenerators.hlsli"

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
Texture2D<float>                            t_envmapImportanceMap           : register(t12);

StructuredBuffer<SubInstanceData>   t_SubInstanceData                       : register(t1);
StructuredBuffer<InstanceData>      t_InstanceData                          : register(t2);
StructuredBuffer<GeometryData>      t_GeometryData                          : register(t3);
StructuredBuffer<MaterialPTData>    t_MaterialPTData                        : register(t5);

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



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// ENVMAP SECTION
///
float EnvironmentComputeWeight( uint dim, uint x, uint y )
{
    uint dimLog2 = (uint)log2( (float)dim );
    int mipLevel = g_const.EnvMapImportanceMapMIPCount - dimLog2 - 1;
    float areaMul = pow(4.0,mipLevel);
    return max( 0, t_envmapImportanceMap.Load( int3( x, y, mipLevel ) ) * average(g_const.EnvMapParams.ColorMultiplier) * g_const.DistantVsLocalRelativeImportance );
};
//
float EnvironmentComputeWeightForQTBuild( uint dim, uint x, uint y )
{
    uint dimLog2 = (uint)log2( (float)dim );
    int mipLevel = g_const.EnvMapImportanceMapMIPCount - dimLog2 - 1;
    float areaMul = pow(4.0,mipLevel);
    float weightNodeArea = max( 1e-7f, t_envmapImportanceMap.Load( int3( x, y, mipLevel ) ) ); // make it never zero;
    weightNodeArea = pow(weightNodeArea, 0.80);  // this slightly reduces direct shadow quality but adds a lot of detail when sunlight obscured but receiving light from other parts of the sky
    float ret = areaMul * weightNodeArea;
    return ret;
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
//
float3 EnvironmentQuadLight::SampleLocalSpace(float3 localDir)
{
    return float3(0,0,0); // not needed here - in case needed for debugging - add!
}
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
groupshared uint    g_nodes[RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT];
groupshared float   g_nodeWeights[RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT];
groupshared float   g_totalWeight[ENV_LIGHTS_BAKE_THREADS];
groupshared float   g_findMax[ENV_LIGHTS_BAKE_THREADS];
groupshared uint    g_findMaxIndex[ENV_LIGHTS_BAKE_THREADS];
[numthreads(ENV_LIGHTS_BAKE_THREADS, 1, 1)] // note, Dispatch size is (1, 1, 1)
void EnvLightsBake( uint dispatchThreadID : SV_DispatchThreadID, uint groupThreadID : SV_GroupThreadId )
{
    const uint baseNodeCount = RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION*RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION;

    GroupMemoryBarrierWithGroupSync();

    // Init base nodes
    for( int i = 0; i < (baseNodeCount+ENV_LIGHTS_BAKE_THREADS-1)/ENV_LIGHTS_BAKE_THREADS; i++ )
    {
        uint lightIndex = i * ENV_LIGHTS_BAKE_THREADS + groupThreadID;

        if( lightIndex < baseNodeCount )
        {
            uint nodeDim    = RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION;
            uint nodeX      = lightIndex / RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION;
            uint nodeY      = lightIndex % RTXPT_LIGHTING_ENVMAP_QT_BASE_RESOLUTION;
            
            g_nodes[lightIndex]         = EQTNodePack(nodeDim, nodeX, nodeY);
            g_nodeWeights[lightIndex]   = EnvironmentComputeWeightForQTBuild(nodeDim, nodeX, nodeY);

#if 0
            uint _nodeDim;
            uint _nodeX  ;
            uint _nodeY  ;
            EQTNodeUnpack(g_nodes[lightIndex], _nodeDim, _nodeX, _nodeY);
            if( nodeDim != _nodeDim || nodeX != _nodeX || nodeY != _nodeY )
                DebugPrint("Error with EQTNodePack/EQTNodeUnpack", (int)42);
#endif
        }
    }

    // Quad tree build 
    GroupMemoryBarrierWithGroupSync(); // g_nodes/g_nodeWeights were touched, have to sync
    uint nodeCount = baseNodeCount; // every thread keeps their node count
    for( int si = 0; si < RTXPT_LIGHTING_ENVMAP_QT_SUBDIVISIONS; si++ ) // we know exactly how many subdivisions we'll make
    {
        // Step one: find the max weight node index within this thread (TODO: read https://on-demand.gputechconf.com/gtc/2010/presentations/S12312-DirectCompute-Pre-Conference-Tutorial.pdf and see if we can optimize)
        float localMax      = 0;
        int localMaxIndex   = -1;
        uint blocks = (nodeCount + ENV_LIGHTS_BAKE_THREADS - 1) / ENV_LIGHTS_BAKE_THREADS;
        for( uint block = 0; block < blocks; block++ )
        {
            int index = block * ENV_LIGHTS_BAKE_THREADS + dispatchThreadID;
            float testWeight = (index < nodeCount)?(g_nodeWeights[index]):(-1);
            if( testWeight > localMax )
            {
                localMax = testWeight;
                localMaxIndex = index;
            }
        }
        g_findMax[dispatchThreadID]         = localMax;
        g_findMaxIndex[dispatchThreadID]    = localMaxIndex;

        // make sure the g_findMax and g_findMaxIndex are available to all threads
        GroupMemoryBarrierWithGroupSync();

        // Step two: reduce across the thread group except last two threads
        [unroll] for( uint reduceI = 2; reduceI < ENV_LIGHTS_BAKE_THREADS; reduceI *= 2 )
        {
            if( dispatchThreadID < ENV_LIGHTS_BAKE_THREADS / reduceI )
            {
                uint otherIndex = dispatchThreadID + ENV_LIGHTS_BAKE_THREADS / reduceI;
                float otherMax = g_findMax[otherIndex];
                if( g_findMax[dispatchThreadID] < otherMax )
                {
                    g_findMax[dispatchThreadID]         = otherMax;
                    g_findMaxIndex[dispatchThreadID]    = g_findMaxIndex[otherIndex];
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Reduce across the final two threads (0 and 1) in the group
        float globalMax     = g_findMax[0];
        int globalMaxIndex  = g_findMaxIndex[0];
        if( g_findMax[1] > globalMax )
        {
            globalMax       = g_findMax[1];
            globalMaxIndex  = g_findMaxIndex[1];
        }
        if( globalMaxIndex == -1 )
            DebugPrint("This shouldn't happen ever");

        // this will store loaded data for the max weight node
        uint nodeDim; uint nodeX; uint nodeY;

        EQTNodeUnpack( g_nodes[globalMaxIndex], nodeDim, nodeX, nodeY );

        GroupMemoryBarrierWithGroupSync(); // this is due to reading from g_nodes[] above, as we'll be modifying it

        // use 4 threads to handle splitting - better than serializing;
        if( dispatchThreadID < 4 /*&& globalMaxIndex != -1*/ )
        {
            nodeDim *= 2; // resolution of the cubemap face that the new node belongs to - increases by 2 with every subdivision!
            nodeX = nodeX*2+(dispatchThreadID%2);
            nodeY = nodeY*2+(dispatchThreadID/2);
            uint newNodeIndex = (dispatchThreadID==0)?(globalMaxIndex):(nodeCount+dispatchThreadID-1);  // reusing the existing node's storage in the first thread, allocating new for remaining 3

            //DebugPrint("It {0}, {1} on {2} -> {3} on {4}", si, uint3(nodeDim/2, nodeX/2, nodeY/2), globalMaxIndex, uint3(nodeDim, nodeX, nodeY), newNodeIndex );

            g_nodes[newNodeIndex]         = EQTNodePack( nodeDim, nodeX, nodeY );

            // mark final nodes with negative weight - it's still correct, just inverted sign; this prevents them from being subdivided in the future loop
            float finalNodeFlag = (nodeDim < g_const.EnvMapImportanceMapResolution)?(1.0):(-1.0);       // note - g_const.EnvMapImportanceMapResolution sets subdivision max due to various considerations such as the lookup map being the same resolution
            g_nodeWeights[newNodeIndex]   = finalNodeFlag * EnvironmentComputeWeightForQTBuild(nodeDim, nodeX, nodeY);
        }

        GroupMemoryBarrierWithGroupSync(); // since we've just modified g_nodes and g_nodeWeights, we must sync up

        nodeCount += 3; // we're always adding 4 new nodes, one in the place of the old one and 3 new ones, so update the count
    }

    //g_nodeCount = nodeCount;

    // Final pass - update historic current-to-past mapping if available, and do any remaining per-quad processing
    GroupMemoryBarrierWithGroupSync();
    //nodeCount = g_nodeCount;
    if( nodeCount != RTXPT_LIGHTING_ENVMAP_QT_TOTAL_NODE_COUNT )
        DebugPrint("Node number overflow/underflow");

    for( int i = 0; i < (nodeCount+ENV_LIGHTS_BAKE_THREADS-1)/ENV_LIGHTS_BAKE_THREADS; i++ )
    {
        uint lightIndex = i * ENV_LIGHTS_BAKE_THREADS + groupThreadID;

        if( lightIndex < nodeCount )
        {
            EnvironmentQuadLight envLight;
            EQTNodeUnpack( g_nodes[lightIndex], envLight.NodeDim, envLight.NodeX, envLight.NodeY );
            envLight.Weight = EnvironmentComputeWeight(envLight.NodeDim, envLight.NodeX, envLight.NodeY);

            uint uniqueID = Hash32CombineSimple( Hash32CombineSimple(Hash32(envLight.NodeX), Hash32(envLight.NodeY)), Hash32(envLight.NodeDim) );

            PolymorphicLightInfoFull lightFull = envLight.Store(uniqueID);
#if 1       // figure out our "world location" and patch it into the lightInfo; used for debugging only - feel free to remove in production code!
            float2 subTexelPos = float2( ((float)envLight.NodeX+0.5) / (float)envLight.NodeDim, ((float)envLight.NodeY+0.5) / (float)envLight.NodeDim );
            float3 localDir = oct_to_ndir_equal_area_unorm(subTexelPos);
            float3 worldDir = EnvironmentQuadLight::ToWorld(localDir);
            lightFull.Base.Center = worldDir * DISTANT_LIGHT_DISTANCE;
            //DebugPrint("", lightFull.Base.Center );
#endif

            u_lightsBuffer[lightIndex] = lightFull.Base;
            u_lightsExBuffer[lightIndex] = lightFull.Extended;

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
            u_historyRemapCurrentToPast[lightIndex] = historicIndex;

//            DebugPrint("", packedLightInfo.Base.Center, packedLightInfo.Base.ColorTypeAndFlags, packedLightInfo.Base.Direction1, packedLightInfo.Base.Direction2, packedLightInfo.Base.Scalars, packedLightInfo.Base.LogRadiance);

#if 0
            DebugPrint("envLight index {0}: ", lightIndex, envLight.NodeDim, envLight.NodeX, envLight.NodeY );
#endif

#if 0
            EnvironmentQuadLight _envLight = envLight.Create(lightFull);
            if( envLight.NodeX != _envLight.NodeX || envLight.NodeY != _envLight.NodeY || envLight.NodeDim != _envLight.NodeDim || envLight.Weight != _envLight.Weight )
                DebugPrint("Error with EnvironmentQuadLight Store / Create (pack/unpack)", envLight.NodeX, _envLight.NodeX, envLight.NodeY, _envLight.NodeY, envLight.NodeDim, _envLight.NodeDim, envLight.Weight, _envLight.Weight);
#endif
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


[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void BakeEmissiveTriangles( uint3 dispatchThreadID : SV_DispatchThreadID ) // note, this is adding triangle lights only - analytic lights have been added on the CPU side already
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    if( dispatchThreadID.x >= g_const.TriangleLightTaskCount )
        return;

    EmissiveTrianglesProcTask task = u_scratchBuffer.Load<EmissiveTrianglesProcTask>(dispatchThreadID.x * sizeof(EmissiveTrianglesProcTask));

    InstanceData instance = t_InstanceData[task.InstanceIndex];
    //uint geometryInstanceIndex = instance.firstGeometryIndex + task.geometryIndex;
    GeometryData geometry = t_GeometryData[instance.firstGeometryIndex + task.GeometryIndex];   // <- can precompute this into task.geometryIndex

    uint materialIndex = t_SubInstanceData[instance.firstGeometryInstanceIndex + task.GeometryIndex].GlobalGeometryIndex_MaterialPTDataIndex & 0xFFFF;
    MaterialPTData material = t_MaterialPTData[materialIndex];

    //DebugPrint( "tID {0}; fgii {1}, fgi {2}, ng {3}", dispatchThreadID, instance.firstGeometryInstanceIndex, instance.firstGeometryIndex, instance.numGeometries  );

    // if( task.EmissiveLightMappingOffset != (instance.firstGeometryInstanceIndex + task.GeometryIndex) )
    //     DebugPrint( "ELMO {0}, FGII {1}, GI{2}", task.EmissiveLightMappingOffset, instance.firstGeometryIndex, task.GeometryIndex );

    int triangleCount = task.TriangleIndexTo-task.TriangleIndexFrom;

    // culling removed unfortunately to maintain fixed memory allocation and track it from the CPU side
    uint collectedLightCount = 0;
    // PolymorphicLightInfo collectedLights[LLB_MAX_TRIANGLES_PER_TASK];

    ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.indexBufferIndex)];
    ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(geometry.vertexBufferIndex)];

    for( uint triangleIdx = task.TriangleIndexFrom; triangleIdx < task.TriangleIndexTo; triangleIdx++ )
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

        if ((material.EmissiveTextureIndex != 0xFFFFFFFF) && (geometry.texCoord1Offset != ~0u) && ((material.Flags & MaterialPTFlags_UseEmissiveTexture) != 0))
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

        uint uniqueID = Hash32CombineSimple( Hash32CombineSimple(Hash32(collectedLightCount), Hash32(task.InstanceIndex)), Hash32(task.GeometryIndex) );

        uint lightIndex = task.DestinationBufferOffset+collectedLightCount;

        PolymorphicLightInfoFull lightFull = triLight.Store(uniqueID);
        u_lightsBuffer[lightIndex] = lightFull.Base;
        u_lightsExBuffer[lightIndex] = lightFull.Extended;

        uint historicIndex = RTXPT_INVALID_LIGHT_INDEX;
        if( task.HistoricBufferOffset != RTXPT_INVALID_LIGHT_INDEX )
        {
            historicIndex = task.HistoricBufferOffset+collectedLightCount;
            u_historyRemapPastToCurrent[historicIndex] = lightIndex;
        }

        u_historyRemapCurrentToPast[lightIndex] = historicIndex;

        collectedLightCount++;
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
    const uint lightCount = controlInfo.TotalLightCount;
    if( lightIndex >= lightCount )
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
    const int to = min( from + LLB_LOCAL_BLOCK_SIZE, controlInfo.TotalLightCount );

    // this breaks stuff - something to do with group memory barrier sync
    // if( from >= controlInfo.TotalLightCount )
    //     return;

    float blockWeightSum = 0.0;
    for( int lightIndex = from; lightIndex < to; lightIndex ++ )
    {
        if( lightIndex >= controlInfo.TotalLightCount )
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
        for( int lightIndex = 0; lightIndex < controlInfo.TotalLightCount; lightIndex ++ )
            testSum += u_lightWeights[ lightIndex ];

        if( !RelativelyEqual( controlInfo.WeightsSum(), testSum, 5e-5f ) )
            DebugPrint( "Compute weight sum {0}, test: {1}", controlInfo.WeightsSum(), testSum );
    }
#endif

    const LightingControlData controlInfo = u_controlBuffer[0];

    const uint lightIndex = dispatchThreadID;
    const uint lightCount = controlInfo.TotalLightCount;
    if( lightIndex >= lightCount )
        return;

    const uint cTotalSamplingProxiesBudget = RTXPT_LIGHTING_SAMPLING_PROXY_RATIO*(max( controlInfo.TotalLightCount, RTXPT_LIGHTING_MAX_LIGHTS/20 ) );    // Sampling proxies budget is based on current total lights or 5% of max supported lights, whichever is greater. This allows small number of lights to benefit from better balancing, without adding too much to the overall cost.
    const float weightSum = asfloat(controlInfo.WeightsSumUINT);

    // this is what comes from past frame's feedback on light usage
    const float feedbackWeight = (float)u_perLightProxyCounters[lightIndex] * weightSum / (float)max( 1.0, (controlInfo.ValidFeedbackCount) );

    // if( dispatchThreadID==0 )
    //     DebugPrint("Valid count {0} ", controlInfo.ValidFeedbackCount );


    // combine computed light weights with historical usage-based feedback weight
    const float lightWeight = lerp( u_lightWeights[ lightIndex ], feedbackWeight, g_const.GlobalFeedbackUseRatio );

    uint lightSamplingProxies = 0;
    if( lightWeight > 0 )
        // if controlInfo.ImportanceSamplingType==0, we use 1 proxy per light - all this is unnecessary but kept in to reduce code complexity as "uniform" mode is for reference/testing only anyway
        lightSamplingProxies = (controlInfo.ImportanceSamplingType==0)?(1):(uint( ceil( (float(cTotalSamplingProxiesBudget-controlInfo.TotalLightCount) * lightWeight) / weightSum ) ));

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

[numthreads(1, 1, 1)]
void ComputeProxyBaselineOffsets( uint dispatchThreadID : SV_DispatchThreadID, uint groupThreadID : SV_GroupThreadId )
{
    const LightingControlData controlInfo = u_controlBuffer[0];
    const uint lightCount = controlInfo.TotalLightCount;

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

[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void CreateProxyJobs( uint dispatchThreadID : SV_DispatchThreadID, uint groupThreadID : SV_GroupThreadId )
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    const uint lightIndex = dispatchThreadID;
    const uint lightCount = controlInfo.TotalLightCount;
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
            if ( lightIndex >= controlInfo.TotalLightCount )
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

LightFeedbackReservoir SampleFeedback(const LightingControlData controlInfo, int2 coord, bool indirect)
{
    if( !(all(coord >= 0.xx) && all(coord < g_const.FeedbackResolution.xy)) )
        return LightFeedbackReservoir::make();

    LightFeedbackReservoir reservoir = LightFeedbackReservoir::Unpack8Byte( u_feedbackReservoirBuffer[ (uint2)coord.xy + uint2(0, indirect?g_const.FeedbackResolution.y:0) ] );

    // Remap historic index
    uint lightIndex = RemapPastToCurrent(controlInfo, reservoir.CandidateIndex);
    if( lightIndex == RTXPT_INVALID_LIGHT_INDEX )
        return LightFeedbackReservoir::make();

    reservoir.CandidateIndex = lightIndex;
    return reservoir;
}

bool SampleFeedback(const LightingControlData controlInfo, int2 coord, bool indirect, inout uint lightIndex)
{
    LightFeedbackReservoir reservoir = SampleFeedback(controlInfo, coord, indirect);
    lightIndex = reservoir.CandidateIndex;
    return lightIndex != RTXPT_INVALID_LIGHT_INDEX;
}

static const uint c_directNeighbourCount = 5;
static const int2 c_directNeighbourOffsets[c_directNeighbourCount] = {     int2( 0, 0),
                                int2(-1, 0), int2(+1, 0), int2( 0,-1), int2( 0,+1),
                                //int2(-1,-1), int2(+1,-1), int2(-1,+1), int2(+1,+1) 
                                };

// Flood fill empty neighbours and apply history as global feedback; 
// Note: we're processing in 3x3 blocks to avoid too many InterlockedAdds on the same memory location; we tried 2x2, 4x4, 4x2 - 3x3 is the best.
// Note2: Johannes suggested WaveMatch - might be perfect use case
[numthreads(8, 8, 1)] void ProcessFeedbackHistoryP0( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const LightingControlData controlInfo = u_controlBuffer[0];

    uint2 collectedLights[9];
    uint collectedLightsCount = 0;

    for( uint x = 0; x < 3; x++ )
    {
        for( uint y = 0; y < 3; y++ )
        {
            uint2 pixelCoord = dispatchThreadID.xy * uint2(3,3) + uint2(x, y);

            bool indirect = false; // prevent direct and indirect parts, which are stacked one on the other, bleeding into each other
            if( pixelCoord.y >= g_const.FeedbackResolution.y )
            {
                indirect = true;
                pixelCoord.y -= g_const.FeedbackResolution.y;
            }

            if( pixelCoord.x >= g_const.FeedbackResolution.x || pixelCoord.y >= g_const.FeedbackResolution.y )
                continue;

            if( g_const.DebugDrawType == (int)LightingDebugViewType::FeedbackRaw || g_const.DebugDrawType == (int)LightingDebugViewType::MissingFeedback )
            {
                uint dbgLightIndex = 0;
                bool hasSample = SampleFeedback( controlInfo, int2(pixelCoord), indirect, dbgLightIndex ) && dbgLightIndex != RTXPT_INVALID_LIGHT_INDEX;

                if( g_const.DebugDrawType == (int)LightingDebugViewType::MissingFeedback && g_const.DebugDrawDirect == !indirect )
                    DebugPixel( pixelCoord.xy, float4( 1 - hasSample, hasSample*0.3, 0, 0.95) );
                if( g_const.DebugDrawType == (int)LightingDebugViewType::FeedbackRaw && g_const.DebugDrawDirect == !indirect )
                    DebugPixel( pixelCoord.xy, float4( ColorFromHash(Hash32(dbgLightIndex)), 0.95) );
            }

            SampleGenerator sampleGenerator = SampleGenerator::make( SampleGeneratorVertexBase::make( pixelCoord, 0, g_const.SampleIndex ) );

            uint lightIndex = RTXPT_INVALID_LIGHT_INDEX;

            LightFeedbackReservoir reservoir = SampleFeedback( controlInfo, int2(pixelCoord)+c_directNeighbourOffsets[0], indirect );
            for( int i = 1; i < c_directNeighbourCount; i++ )
            {
                LightFeedbackReservoir other = SampleFeedback( controlInfo, int2(pixelCoord)+c_directNeighbourOffsets[i], indirect );
                if( other.CandidateIndex == RTXPT_INVALID_LIGHT_INDEX )
                    continue;
                other.Scale(0.05); // TODO; move this outside - make the central one 20x larger :)
                reservoir.Merge( sampleNext1D( sampleGenerator ), other );
                //reservoir.Add( sampleNext1D( sampleGenerator ), other.CandidateIndex, other.CandidateWeight * ((i==0)?20:1) );
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

            // increase the global counters; there's a high cost to this InterlockedAdd - i.e. 0.05ms for everything except this and 0.3ms for everything plus this
            if (lightIndex != RTXPT_INVALID_LIGHT_INDEX)
            {
            #if 0
                //uint k = sampleNext1D( sampleGenerator ) * u_controlBuffer[0].TotalLightCount;    <- perf test; when adding to random it's not nearly as costly
                InterlockedAdd( u_perLightProxyCounters[lightIndex], 1 );

                // if( all(dispatchThreadID.xy == 0.xx) )
                //     DebugPrint("Adding {0} to index {1}", 1, lightIndex );
            #else
                bool found = false;
                [unroll(8)] for (uint i = 0; i < collectedLightsCount; i++) // unroll(8) is ok because it's max-1 and 9 is max
                {
                    if( collectedLights[i].x == lightIndex )
                    {
                        collectedLights[i].y++;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    collectedLights[collectedLightsCount++] = uint2(lightIndex,1);
                }
            #endif
            }
        }
    }

    uint totalCount = 0;

    // here's where we could also somehow "steal" lights from other threads in the warp to minimize InterlockedAdd calls
    [unroll(9)] for (uint i = 0; i < collectedLightsCount; i++)
    {
        uint2 light = collectedLights[i];
        InterlockedAdd( u_perLightProxyCounters[light.x], light.y );

        totalCount += light.y;
        
        // if( all(dispatchThreadID.xy == 0.xx) )
        //     DebugPrint("Adding {0} to index {1}", light.y, light.x );
    }

    // look into https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/waveprefixcountbytes
    InterlockedAdd( u_controlBuffer[0].ValidFeedbackCount, totalCount );

    // if( all(dispatchThreadID.xy == uint2(0,0)) )
    //     DebugPrint("totalCount {0}", totalCount );
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
 
    SampleGenerator sampleGenerator = SampleGenerator::make( SampleGeneratorVertexBase::make( dispatchThreadID.xy, 0, g_const.SampleIndex ), SampleGeneratorEffectSeed::Base, false, 1 );

    {
        int2 pixelPos = dispatchThreadID;
        float3 screenSpaceMotion = ConvertMotionVectorToPixelSpace( pixelPos, t_motionVectors[pixelPos] );
        int2 prevPixelPos = int2( float2(pixelPos) + screenSpaceMotion.xy + 0.5.xx /* + sampleNext2D(sampleGenerator)*/ );

        // if wrong/missing motion vectors, use current as backup
        if( !(all(prevPixelPos >= 0.xx) && all(prevPixelPos < g_const.FeedbackResolution.xy)) )
            prevPixelPos = pixelPos;

        // note: it might not be a bad idea to stochastically ignore 10% of motion vectors and sample from original - this catches reflected speculars in lateral motion that stay in place

        // part 1: sample direct feedback buffer - do a search for 3x3 kernel before falling back to global sample
        uint lightIndex = RTXPT_INVALID_LIGHT_INDEX;
        if( all(prevPixelPos >= 0.xx) && all(prevPixelPos < g_const.FeedbackResolution.xy) )
            lightIndex = u_processedFeedbackBuffer[prevPixelPos + uint2(0, indirect?g_const.FeedbackResolution.y:0)].x;

        // part 2:
        uint lightIndexHistoric = RTXPT_INVALID_LIGHT_INDEX;
        if( all(prevPixelPos >= 0.xx) && all(prevPixelPos < g_const.FeedbackResolution.xy) && controlInfo.LastFrameLocalSamplesAvailable )
        {
            uint2 tilePos = (prevPixelPos+controlInfo.LocalSamplingTileJitterPrev) / RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE;
            lightIndexHistoric = SampleLightLocalHistoric(tilePos, indirect, sampleGenerator);
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
                    LightFeedbackReservoir other = SampleFeedback( controlInfo, prevPixelPos, indirect );
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
    int2 lrWindowTopLeft = int2( float2(cellTopLeft+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE*0.5) / RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE.xx - RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE*0.5 + 0.5.xx );

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

            u_narrowSamplingBuffer[ uint3(actualTilePos, currentlyCollectedCount++) ] = PackMiniListLightAndCount( lightIndexA, 0 );
        }

    for( int x = 0; x < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE; x++ )
        for( int y = 0; y < RTXPT_LIGHTING_LR_SAMPLING_BUFFER_WINDOW_SIZE; y++ )
        {
            int2 lrPixelPos = lrWindowTopLeft + int2(x,y);
            int2 lrSrcCoord = MirrorCoord(lrPixelPos, g_const.LRFeedbackResolution);

            uint lightIndex = u_reprojectedLRFeedbackBuffer[lrSrcCoord + uint2(0, indirect?g_const.LRFeedbackResolution.y:0)].x;
            if( lightIndex == RTXPT_INVALID_LIGHT_INDEX )
                DebugPrint("LR bad light read from {0} - missing barrier or etc?", lrSrcCoord);

            u_narrowSamplingBuffer[ uint3(actualTilePos, currentlyCollectedCount++) ] = PackMiniListLightAndCount( lightIndex, 0 );
        }  

    SampleGenerator sampleGenerator = SampleGenerator::make( SampleGeneratorVertexBase::make( tilePos, 0, g_const.SampleIndex ) );
    for ( int i = 0; i < RTXPT_LIGHTING_HISTORIC_SAMPLING_COUNT; i++ )
    {
        uint ox = sampleGenerator.Next() % RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE;
        uint oy = sampleGenerator.Next() % RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE;

        int2 pixelPos = windowTopLeft + int2(ox,oy);
        int2 srcCoord = MirrorCoord(pixelPos, g_const.FeedbackResolution);

        uint lightIndex = u_reprojectedFeedbackBuffer[srcCoord + uint2(0, indirect?g_const.FeedbackResolution.y:0)].y;  // note, .y is historic samples!

        if ( lightIndex == RTXPT_INVALID_LIGHT_INDEX )// || lightIndexB == RTXPT_INVALID_LIGHT_INDEX )
        {
            DebugPixel( cellTopLeft, float4(1,0,0,1) );
            DebugPrint("Bad historic light read from {0} - missing barrier or etc?", srcCoord);
        }
        u_narrowSamplingBuffer[ uint3(actualTilePos, currentlyCollectedCount++) ] = PackMiniListLightAndCount( lightIndex, 0 );
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
/// *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE *** EXPERIMENTAL CODE ***


[numthreads(LLB_NUM_COMPUTE_THREADS, 1, 1)]
void DebugDrawLights( uint dispatchThreadID : SV_DispatchThreadID )
{
    const uint totalLightCount = u_controlBuffer[0].TotalLightCount;
    if( dispatchThreadID >= totalLightCount )
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
    int2 lrWindowTopLeft = int2( float2(cellTopLeft+RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE*0.5) / RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE.xx - RTXPT_LIGHTING_LR_SAMPLING_BUFFER_SCALE*0.5 + 0.5.xx );

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


#define LLB_ENABLE_LLRB_VALIDATION
#ifdef LLB_ENABLE_LLRB_VALIDATION

    uint dataToValidate[RTXPT_LIGHTING_NARROW_PROXY_COUNT];
    for( int i = 0 ; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
        dataToValidate[i] = u_narrowSamplingBuffer[ uint3(tilePos + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0), i) ];

    FillTile(tilePos, indirect);  // this fills u_narrowSamplingBuffer from scratch

    SortedLightList<RTXPT_LIGHTING_NARROW_PROXY_COUNT> localList = SortedLightList<RTXPT_LIGHTING_NARROW_PROXY_COUNT>::empty();
    for( uint i = 0; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
        localList.InsertOrIncCounter( UnpackMiniListLight(u_narrowSamplingBuffer[ uint3(tilePos + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0), i) ]) );
    bool allGood = localList.Validate(dataToValidate, debugTile);

    if( !allGood )
    {
        for( int x = 0; x < RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE; x++ )
            for( int y = 0; y < RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE; y++ )
            {
                int2 pixelPos = windowTopLeft + int2(x,y);
                DebugPixel( pixelPos, float4(1,0,0,1) );
            }
    }

    // put back original data in
    for( int i = 0 ; i < RTXPT_LIGHTING_NARROW_PROXY_COUNT; i++ )
        u_narrowSamplingBuffer[ uint3(tilePos + uint2(0, indirect?g_const.NarrowSamplingResolution.y:0), i) ] = dataToValidate[i];
#endif

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