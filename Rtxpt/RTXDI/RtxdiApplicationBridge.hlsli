/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/*
This header file is the bridge between the RTXDI resampling functions
and the application resources and parts of shader functionality.

The RTXDI SDK provides the resampling logic, and the application provides
other necessary aspects:
    - Material BRDF evaluation;
    - Ray tracing and transparent/alpha-tested material processing;
    - Light sampling functions and emission profiles.

The structures and functions that are necessary for SDK operation
start with the RAB_ prefix (for RTXDI-Application Bridge).

All structures defined here are opaque for the SDK, meaning that
it makes no assumptions about their contents, they are just passed
between the bridge functions.
*/

#ifndef RTXDI_APPLICATION_BRIDGE_HLSLI
#define RTXDI_APPLICATION_BRIDGE_HLSLI

#include <donut/shaders/bindless.h>
#include <donut/shaders/binding_helpers.hlsli>

// RTXDI has slightly different lighting setup although code is shared, so there's a RTXDI specific config
#define POLYLIGHT_OVERRIDE_CONFIG
#define POLYLIGHT_SPHERE_ENABLE         1
#define POLYLIGHT_POINT_ENABLE          1   // handled by sphere
#define POLYLIGHT_TRIANGLE_ENABLE       1
#define POLYLIGHT_DIRECTIONAL_ENABLE    1   // baked into envmap (Distant lighting code)
#define POLYLIGHT_ENV_ENABLE            1   // handled by Distant lighting code, not polymorphic light
#define POLYLIGHT_QT_ENV_ENABLE         0   // environment map quad tree in equal area octahedral mapping
#define POLYLIGHT_CONFIGURED


#include "ShaderParameters.h"
#include "SurfaceData.hlsli"
#include "../Bindings/ShaderResourceBindings.hlsli"
#include "../PathTracerBridgeDonut.hlsli"

// RTXDI resources
StructuredBuffer<PolymorphicLightInfoFull> t_LightDataBuffer    : register(t21 VK_DESCRIPTOR_SET(2));
Buffer<float2> t_NeighborOffsets                            : register(t22 VK_DESCRIPTOR_SET(2));
Buffer<uint> t_LightIndexMappingBuffer                      : register(t23 VK_DESCRIPTOR_SET(2));
Texture2D t_LocalLightPdfTexture                            : register(t25 VK_DESCRIPTOR_SET(2));
StructuredBuffer<uint> t_GeometryInstanceToLight            : register(t26 VK_DESCRIPTOR_SET(2));

// Screen-sized UAVs
RWStructuredBuffer<RTXDI_PackedDIReservoir> u_LightReservoirs : register(u13 VK_DESCRIPTOR_SET(2));
RWStructuredBuffer<RTXDI_PackedGIReservoir> u_GIReservoirs  : register(u14 VK_DESCRIPTOR_SET(2));

// RTXDI UAVs
RWBuffer<uint2> u_RisBuffer                                 : register(u15 VK_DESCRIPTOR_SET(2));
RWBuffer<uint4> u_RisLightDataBuffer                        : register(u16 VK_DESCRIPTOR_SET(2));

// Other
ConstantBuffer<RtxdiBridgeConstants> g_RtxdiBridgeConst     : register(b5 VK_DESCRIPTOR_SET(2));

SamplerState s_EnvironmentSampler                           : register(s4 VK_DESCRIPTOR_SET(2));

#define RTXDI_RIS_BUFFER u_RisBuffer
#define RTXDI_LIGHT_RESERVOIR_BUFFER u_LightReservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER t_NeighborOffsets
#define RTXDI_GI_RESERVOIR_BUFFER u_GIReservoirs

#define IES_SAMPLER s_EnvironmentMapSampler

#define RTXDI_ENVIRONMENT_MAP t_EnvironmentMap

#if !defined(RTXPT_RTXDI_RESOURCES_ONLY)

#include "PolymorphicLightRTXDI.hlsli"

static const bool kSpecularOnly = false;
static const float kMinRoughness = 0.05f;

//Types
typedef PathTracerSurfaceData RAB_Surface;

enum class RayHitType : uint
{
    TriangleHit,
    NoHit,
};

struct RayHitInfo
{
    void InitTriangleHit(uint _instanceID, uint _geometryIndex, uint _primitiveIndex, float2 _barycentrics)
    {
        hitType = RayHitType::TriangleHit;
        instanceID = _instanceID;
        geometryIndex = _geometryIndex;
        primitiveIndex = _primitiveIndex;
        barycentrics = _barycentrics;
    }

    void InitNoHit() 
    {
        hitType = RayHitType::NoHit;
        instanceID = 0;
        geometryIndex = 0;
        primitiveIndex = 0;
        barycentrics = 0;
    }

    RayHitType hitType;
    uint instanceID;
    uint geometryIndex;
    uint primitiveIndex;
    float2 barycentrics;
};

#if !USE_RAY_QUERY
struct RayPayload
{
    float3 throughput;
    float committedRayT;
    uint instanceIndex;
    uint geometryIndex;
    uint primitiveIndex;
    bool frontFace;
    float2 barycentrics;
};

// Not implemented, use alphaTest() instead 
bool considerTransparentMaterial(uint instanceIndex, uint geometryIndex, uint triangleIndex, float2 rayBarycentrics, inout float3 throughput)
{
    return false;
}

struct RayAttributes
{
    float2 uv;
};

[shader("miss")]
void Miss(inout RayPayload payload : SV_RayPayload)
{
}

[shader("closesthit")]
void ClosestHit(inout RayPayload payload : SV_RayPayload, in RayAttributes attrib : SV_IntersectionAttributes)
{
    payload.committedRayT = RayTCurrent();
    payload.instanceIndex = InstanceIndex();
    payload.geometryIndex = GeometryIndex();
    payload.primitiveIndex = PrimitiveIndex();
    payload.frontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    payload.barycentrics = attrib.uv;
}

[shader("anyhit")]
void AnyHit(inout RayPayload payload : SV_RayPayload, in RayAttributes attrib : SV_IntersectionAttributes)
{
    if (!considerTransparentMaterial(InstanceIndex(), GeometryIndex(), PrimitiveIndex(), attrib.uv, payload.throughput))
        IgnoreHit();
}
#endif

RayHitInfo TraceVisibilityRay(RaytracingAccelerationStructure accelStruct, RayDesc ray)
{
#if USE_RAY_QUERY
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rayQuery;
    rayQuery.TraceRayInline(accelStruct, RAY_FLAG_NONE, 0xff, ray);

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            [branch] 
            if (Bridge::AlphaTestVisibilityRay(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
            ))
            {
                rayQuery.CommitNonOpaqueTriangleHit();
            }
        }
    }

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        RayHitInfo outHitInfo;
        outHitInfo.InitTriangleHit(rayQuery.CommittedInstanceID(), rayQuery.CommittedGeometryIndex(), rayQuery.CommittedPrimitiveIndex(), rayQuery.CommittedTriangleBarycentrics());
        return outHitInfo;
    }
    else
    {
        RayHitInfo outHitInfo;
        outHitInfo.InitNoHit();
        return outHitInfo;
    }
#else
    RayPayload payload = (RayPayload)0;
    payload.instanceIndex = ~0u;
    payload.throughput = 1.0;
    TraceRay(accelStruct, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, INSTANCE_MASK_ALL, 0, 0, 0, ray, payload);

    if (payload.instanceIndex == ~0u)
    {
        RayHitInfo outHitInfo;
        outHitInfo.InitNoHit();
        return outHitInfo;
    }
    else
    {
        RayHitInfo outHitInfo;
        outHitInfo.InitTriangleHit(payload.instanceIndex, payload.geometryIndex, payload.primitiveIndex, payload.barycentrics);
        return outHitInfo;
    }
#endif
}

struct RAB_LightSample
{
    float3 position;
    float3 normal;
    float3 radiance;
    float solidAnglePdf;
    PolymorphicLightType lightType;
};

typedef PolymorphicLightInfoFull RAB_LightInfo;
// Using the RTXPT Sample Generator
typedef SampleGenerator RAB_RandomSamplerState;


// Initialized the random sampler for a given pixel or tile index.
// The pass parameter is provided to help generate different RNG sequences
// for different resampling passes, which is important for image quality.
// In general, a high quality RNG is critical to get good results from ReSTIR.
// A table-based blue noise RNG dose not provide enough entropy, for example.
RAB_RandomSamplerState RAB_InitRandomSampler(uint2 index, uint pass)
{
    return SampleGenerator::make(SampleGeneratorVertexBase::make(index, pass, Bridge::getSampleIndex()));
}

// Draws a random number X from the sampler, so that (0 <= X < 1).
float RAB_GetNextRandom(inout RAB_RandomSamplerState rng)
{
    return sampleNext1D(rng);
}


//Empty type constructors
RAB_Surface RAB_EmptySurface()
{
    return PathTracerSurfaceData::makeEmpty();
}

RAB_LightInfo RAB_EmptyLightInfo()
{
    return (RAB_LightInfo)0;
}

RAB_LightSample RAB_EmptyLightSample()
{
    return (RAB_LightSample)0;
}

int2 RAB_ClampSamplePositionIntoView(int2 pixelPosition, bool previousFrame)
{
    return clamp(pixelPosition, 0, int2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) - 1);
}

RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool previousFrame)
{
    return getGBufferSurface(pixelPosition, previousFrame);
}

// Checks if the given surface is valid, see RAB_GetGBufferSurface.
bool RAB_IsSurfaceValid(RAB_Surface surface)
{
    return !surface.isEmpty();
}

// Returns the world position of the given surface
float3 RAB_GetSurfaceWorldPos(RAB_Surface surface)
{
    // Should we use posW instead?
    //return surface.shadingData.posW;
    return surface.ComputeNewRayOrigin();
    //return surface.position;
}

// Returns the world position of the given surface
float3 RAB_GetNewRayOrigin(RAB_Surface surface)
{
    // Should we use posW instead?
    return surface.ComputeNewRayOrigin();
}

// Returns the world shading normal of the given surface
float3 RAB_GetSurfaceNormal(RAB_Surface surface)
{
    return surface.GetNormal();
}

// Returns the linear depth of the given surface.
// It doesn't have to be linear depth in a strict sense (i.e. viewPos.z),
// and can be distance to the camera or primary path length instead.
// Just make sure that the motion vectors' .z component follows the same logic.
float RAB_GetSurfaceLinearDepth(RAB_Surface surface)
{
    return surface.GetViewDepth();
}


//Lights and Samples
// Loads polymorphic light data from the global light buffer.
RAB_LightInfo RAB_LoadLightInfo(uint index, bool previousFrame)
{
    return t_LightDataBuffer[index];
}

// Stores triangle light data into a tile.
// Returns true if this light can be stored in a tile (i.e. compacted).
// If it cannot, for example it's a shaped light, this function returns false and doesn't store.
// A basic implementation can ignore this feature and always return false, which is just slower.
bool RAB_StoreCompactLightInfo(uint linearIndex, RAB_LightInfo lightInfo)
{
    uint4 data1, data2;
    if (!PolymorphicLight::PackCompactInfo(lightInfo, data1, data2))
        return false;

    u_RisLightDataBuffer[linearIndex * 2 + 0] = data1;
    u_RisLightDataBuffer[linearIndex * 2 + 1] = data2;

    return true;
}

// Loads triangle light data from a tile produced by the presampling pass.
RAB_LightInfo RAB_LoadCompactLightInfo(uint linearIndex)
{
    uint4 packedData1, packedData2;
    packedData1 = u_RisLightDataBuffer[linearIndex * 2 + 0];
    packedData2 = u_RisLightDataBuffer[linearIndex * 2 + 1];
    return PolymorphicLight::UnpackCompactInfo(packedData1, packedData2);
}

// Translates the light index from the current frame to the previous frame (if currentToPrevious = true)
// or from the previous frame to the current frame (if currentToPrevious = false).
// Returns the new index, or a negative number if the light does not exist in the other frame.
int RAB_TranslateLightIndex(uint lightIndex, bool currentToPrevious)
{
    // In this implementation, the mapping buffer contains both forward and reverse mappings,
    // stored at different offsets, so we don't care about the currentToPrevious parameter.
    uint mappedIndexPlusOne = t_LightIndexMappingBuffer[lightIndex];

    // The mappings are stored offset by 1 to differentiate between valid and invalid mappings.
    // The buffer is cleared with zeros which indicate an invalid mapping.
    // Subtract that one to make this function return expected values.
    return int(mappedIndexPlusOne) - 1;
}

// Returns the direction and distance from the surface to the light sample
void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample,
    out float3 o_lightDir,
    out float o_lightDistance)
{
    if (lightSample.lightType == PolymorphicLightType::kEnvironment /*|| lightSample.lightType == PolymorphicLightType::kDirectional*/)
    {
        o_lightDir = -lightSample.normal;
        o_lightDistance = DISTANT_LIGHT_DISTANCE;
    }
    else
    {
        float3 toLight = lightSample.position - RAB_GetSurfaceWorldPos(surface);
        o_lightDistance = length(toLight);
        o_lightDir = toLight / o_lightDistance;
        //o_lightDistance = max(0, o_lightDistance - g_RtxdiBridgeConst.rayEpsilon);
    }
}

// Return true if the light sample comes from an analytic light
bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return lightSample.lightType == PolymorphicLightType::kPoint || 
        lightSample.lightType == PolymorphicLightType::kDirectional;
}

// Returns the solid angle PDF of the light sample 
float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
    return lightSample.solidAnglePdf;
}

float2 RAB_GetEnvironmentMapRandXYFromDir(float3 worldDir)
{
    return Encode_Oct(worldDir);
}

// Computes the probability of a particular direction being sampled from the environment map
// relative to all the other possible directions, based on the environment map pdf texture.
float RAB_EvaluateEnvironmentMapSamplingPdf(float3 L)
{
    if (!g_RtxdiBridgeConst.environmentMapImportanceSampling)
        return 1.0;

    EnvMapSampler envMapSampler = EnvMapSampler::make( s_EnvironmentMapImportanceSampler, t_EnvironmentMapImportanceMap, g_Const.envMapImportanceSamplingParams,
                                                        t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams );
    return envMapSampler.MIPDescentEvalPdf(L);
}

// Evaluates pdf for a particular light
float RAB_EvaluateLocalLightSourcePdf(uint lightIndex)
{
    uint2 pdfTextureSize = g_RtxdiBridgeConst.localLightPdfTextureSize.xy;
    uint2 texelPosition = RTXDI_LinearIndexToZCurve(lightIndex);
    float texelValue = t_LocalLightPdfTexture[texelPosition].r;

    int lastMipLevel = g_RtxdiBridgeConst.localLightPdfLastMipLevel;
    float averageValue = t_LocalLightPdfTexture.mips[lastMipLevel][uint2(0, 0)].x;

    // See the comment at 'sum' in RAB_EvaluateEnvironmentMapSamplingPdf.
    // The same texture shape considerations apply to local lights.
    float sum = averageValue * square(1u << lastMipLevel);

    return texelValue / sum;
}

float3 _Schlick_Fresnel(float3 F0, float VdotH)
{
    return F0 + (1 - F0) * pow(max(1 - VdotH, 0), 5);
}

//Sampling functions
float getSurfaceDiffuseProbability(RAB_Surface surface)
{
    float diffuseWeight = luminance(surface.GetDiffuse());
    float specularWeight = luminance(_Schlick_Fresnel(surface.GetSpecular(), dot(surface.GetView(), surface.GetNormal())));
    float sumWeights = diffuseWeight + specularWeight;
    return sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);
}

// Computes the weight of the given light samples when the given surface is
// shaded using that light sample. Exact or approximate BRDF evaluation can be
// used to compute the weight. ReSTIR will converge to a correct lighting result
// even if all samples have a fixed weight of 1.0, but that will be very noisy.
// Scaling of the weights can be arbitrary, as long as it's consistent
// between all lights and surfaces.
float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    if (lightSample.solidAnglePdf <= 0)
        return 0;

    float3 toLight;// = normalize(lightSample.position - RAB_GetSurfaceWorldPos(surface));
    float dis;
    RAB_GetLightDirDistance(surface, lightSample, toLight, dis);

#ifdef RAB_NO_TRANSMISSION_MATERIAL  // we have BSDFs so this early out breaks some surfaces
    if (dot(toLight, RAB_GetSurfaceNormal(surface)) <= 0)
        return 0;
#endif

    float3 fullBRDF = surface.Eval(toLight);
    return luminance(fullBRDF * lightSample.radiance) / lightSample.solidAnglePdf;

}

// Computes the weight of the given light for arbitrary surfaces located inside 
// the specified volume. Used for world-space light grid construction.
float RAB_GetLightTargetPdfForVolume(RAB_LightInfo light, float3 volumeCenter, float volumeRadius)
{
    return PolymorphicLight::GetWeightForVolume(light, volumeCenter, volumeRadius);
}

// Performs importance sampling of the surface's BRDF and returns the sampled direction.
bool RAB_GetSurfaceBrdfSample(RAB_Surface surface, inout RAB_RandomSamplerState rng, out float3 dir)
{
    BSDFSample result;
    surface.Sample(rng, result, true);

    dir = result.wo;
    return dot(RAB_GetSurfaceNormal(surface), dir) > 0.f;
}

// Computes the PDF of a particular direction being sampled by RAB_GetSurfaceBrdfSample.
float RAB_GetSurfaceBrdfPdf(RAB_Surface surface, float3 dir)
{
#ifdef RAB_NO_TRANSMISSION_MATERIAL  // we have BSDFs so this early out breaks some surfaces
   if (dot(RAB_GetSurfaceNormal(surface), dir) <= 0.f)
        return 0;
#endif
    return surface.EvalPdf(dir, true);
}


// Samples a polymorphic light relative to the given receiver surface.
// For most light types, the "uv" parameter is just a pair of uniform random numbers, originally
// produced by the RAB_GetNextRandom function and then stored in light reservoirs.
// For importance sampled environment lights, the "uv" parameter has the texture coordinates
// in the PDF texture, normalized to the (0..1) range.
RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
    PolymorphicLightSample pls = PolymorphicLight::CalcSample(lightInfo, uv, RAB_GetSurfaceWorldPos(surface));

    RAB_LightSample lightSample;
    lightSample.position = pls.Position;
    lightSample.normal = pls.Normal;
    lightSample.radiance = pls.Radiance;
    lightSample.solidAnglePdf = pls.SolidAnglePdf;
    lightSample.lightType = PolymorphicLight::DecodeType(lightInfo);

    return lightSample;
}

uint getLightIndex(uint firstGeometryInstanceIndex, uint geometryIndex, uint primitiveIndex)
{
    uint lightIndex = RTXDI_InvalidLightIndex;
    uint geometryInstanceIndex = firstGeometryInstanceIndex + geometryIndex;
    lightIndex = t_GeometryInstanceToLight[geometryInstanceIndex];

    //return lightIndex + primitiveIndex;
    return lightIndex == RTXDI_InvalidLightIndex ? lightIndex : lightIndex + primitiveIndex;
}

RayDesc setupVisibilityRay(RAB_Surface surface, RAB_LightSample lightSample, float offset = 0.001)
{
    float3 surfacePos = RAB_GetNewRayOrigin(surface);
   // float3 toLight = lightSample.position - surfacePos;
    float3 toLight;// = normalize(lightSample.position - RAB_GetSurfaceWorldPos(surface));
    float dis;
    RAB_GetLightDirDistance(surface, lightSample, toLight, dis);

    RayDesc ray;
    ray.TMin = 0;// offset;
    ray.TMax = max(offset, dis/*length(toLight)*/ - offset);
    ray.Direction = normalize(toLight);
    ray.Origin = surfacePos;

    return ray;
}

RayDesc setupVisibilityRay(RAB_Surface surface, float3 samplePosition, float offset = 0.001)
{
    float3 L = samplePosition - surface.GetPosW();

    const bool isViewFrontFace = true; //dot(surface.GetView(), surface.GetFaceN()) > 0;
    const bool isLightFrontFace = dot(L, surface.GetFaceNCorrected()) > 0;

    float3 origin = surface.ComputeNewRayOrigin(isViewFrontFace == isLightFrontFace);

    L = samplePosition - origin;
    float dist = length(L);
    L /= dist;

    RayDesc ray;
    ray.TMin = 0;
    ray.TMax = max(0, dist - offset);
    ray.Direction = L;
    ray.Origin = origin;
    return ray;
}

bool GetConservativeVisibility(RaytracingAccelerationStructure accelStruct, RayDesc ray)
{
    const RayHitInfo res = TraceVisibilityRay(accelStruct, ray);

    const bool visible = res.hitType == RayHitType::NoHit;

    return visible;
}

// Traces an expensive visibility ray that considers all alpha tested and transparent geometry along the way.
// Only used in FinalSampling so only supports USE_RAY_QUERY=1
// Not a required bridge function.
// Uses the RTXPT Bridge alpha test
bool GetFinalVisibility(RaytracingAccelerationStructure accelStruct, RayDesc ray)
{
    const RayHitInfo res = TraceVisibilityRay(accelStruct, ray);

    const bool visible = res.hitType == RayHitType::NoHit;

    return visible;
}

// Return true if anything was hit. If false, RTXDI will do environment map sampling
// o_lightIndex: If hit, must be a valid light index for RAB_LoadLightInfo, if no local light was hit, must be RTXDI_InvalidLightIndex
// randXY: The randXY that corresponds to the hit location and is the same used for RAB_SamplePolymorphicLight
bool RAB_TraceRayForLocalLight(float3 origin, float3 direction, float tMin, float tMax,
    out uint o_lightIndex, out float2 o_randXY)
{
    o_lightIndex = RTXDI_InvalidLightIndex;
    o_randXY = 0;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;

    float2 hitUV;
    bool hitAnything;

    const RayHitInfo hitInfo;

#if USE_RAY_QUERY
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rayQuery;
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xff, ray);

    while (rayQuery.Proceed())
    {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            [branch]
            if (Bridge::AlphaTestVisibilityRay(
                rayQuery.CandidateInstanceID(),
                rayQuery.CandidateInstanceIndex(),
                rayQuery.CandidateGeometryIndex(),
                rayQuery.CandidatePrimitiveIndex(),
                rayQuery.CandidateTriangleBarycentrics()
            ))
            {
                rayQuery.CommitNonOpaqueTriangleHit();
            }
        }
    }

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        hitInfo.InitTriangleHit(rayQuery.CommittedInstanceID(), rayQuery.CommittedGeometryIndex(), rayQuery.CommittedPrimitiveIndex(), rayQuery.CommittedTriangleBarycentrics());
    else
        hitInfo.InitNoHit();
#else
    RayPayload payload = (RayPayload)0;
    payload.instanceIndex = ~0u;
    payload.throughput = 1.0;
    TraceRay(SceneBVH, RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, INSTANCE_MASK_ALL, 0, 0, 0, ray, payload);

    if (payload.instanceIndex == ~0u)
        hitInfo.InitNoHit();
    else
        hitInfo.InitTriangleHit(payload.instanceIndex, payload.geometryIndex, payload.primitiveIndex, payload.barycentrics);

#endif

    hitAnything = hitInfo.hitType == RayHitType::TriangleHit;

    if (hitAnything)
    {
        o_lightIndex = getLightIndex(hitInfo.instanceID, hitInfo.geometryIndex, hitInfo.primitiveIndex);
        hitUV = hitInfo.barycentrics;
    }

    if (o_lightIndex != RTXDI_InvalidLightIndex)
        o_randXY = randomFromBarycentric(hitUVToBarycentric(hitUV));

    return hitAnything;
}

//Misc Functions
// Traces a cheap visibility ray that returns approximate, conservative visibility
// between the surface and the light sample. Conservative means if unsure, assume the light is visible.
// Significant differences between this conservative visibility and the final one will result in more noise.
// This function is used in the spatial resampling functions for ray traced bias correction.
bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
    const RayDesc ray = setupVisibilityRay(surface, lightSample, g_RtxdiBridgeConst.rayEpsilon);

    return GetConservativeVisibility(SceneBVH, ray);
}

// Same as RAB_GetConservativeVisibility but for temporal resampling.
// When the previous frame TLAS and BLAS are available, the implementation should use the previous position and the previous AS.
// When they are not available, use the current AS. That will result in transient bias.
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, RAB_LightSample lightSample)
{
    const RayDesc ray = setupVisibilityRay(currentSurface, lightSample, g_RtxdiBridgeConst.rayEpsilon);

    /*if (g_ResamplingConst.enablePreviousTLAS)
        return GetConservativeVisibility(PrevSceneBVH, previousSurface, lightSample);
    else*/
    return GetConservativeVisibility(SceneBVH, ray);
}

// Forward declare the SDK function that's used in RAB_AreMaterialsSimilar
bool RTXDI_CompareRelativeDifference(float reference, float candidate, float threshold);

// Compares the materials of two surfaces, returns true if the surfaces
// are similar enough that we can share the light reservoirs between them.
// If unsure, just return true.
bool RAB_AreMaterialsSimilar(RAB_Surface a, RAB_Surface b)
{
    const float roughnessThreshold = 0.5;
    const float reflectivityThreshold = 0.25;
    const float albedoThreshold = 0.25;

    if (a.GetPlaneHash() != b.GetPlaneHash())
        return false;

    if (!RTXDI_CompareRelativeDifference(a.GetRoughness(), b.GetRoughness(), roughnessThreshold))
        return false;

    if (abs(calcLuminance(a.GetSpecular()) - calcLuminance(b.GetSpecular())) > reflectivityThreshold)
        return false;

    if (abs(calcLuminance(a.GetDiffuse()) - calcLuminance(b.GetDiffuse())) > albedoThreshold)
        return false;

    return true;
}

//Helper functions not defined by RTXDI

// The motion vectors rendered by the G-buffer pass match what is expected by NRD and DLSS.
// In case of dynamic resolution, there is a difference that needs to be corrected...
//
// The rendered motion vectors are computed as:
//     (previousUV - currentUV) * currentViewportSize
//
// The motion vectors necessary for pixel reprojection are:
//     (previousUV * previousViewportSize - currentUV * currentViewportSize)
//
float3 convertMotionVectorToPixelSpace(
    PlanarViewConstants view,
    PlanarViewConstants viewPrev,
    int2 pixelPosition,
    float3 motionVector)
{
    float2 currentPixelCenter = float2(pixelPosition.xy) + 0.5;
    float2 previousPosition = currentPixelCenter + motionVector.xy;
    previousPosition *= viewPrev.viewportSize * view.viewportSizeInv;
    motionVector.xy = previousPosition - currentPixelCenter;
    return motionVector;
}

// Compute incident radience
void ComputeIncidentRadience(RAB_Surface surface, float inversePDF, RAB_LightSample lightSample, 
    out float3 Li, out float3 dir, out float distance)
{
    Li = (0.f, 0.f, 0.f);
    dir = 0;
    distance = 0;

    if (any(lightSample.radiance > 0))
    {
        // Compute incident radience
        Li = (lightSample.radiance / lightSample.solidAnglePdf) * inversePDF ;

        RAB_GetLightDirDistance(surface, lightSample, dir, distance);
        // Subtract epsilon to account for the offset in ray origin
        distance = max(0, distance - g_RtxdiBridgeConst.rayEpsilon);
    }
}


#ifdef RTXDI_WITH_RESTIR_GI

bool RAB_ValidateGISampleWithJacobian(inout float jacobian)
{
    // Sold angle ratio is too different. Discard the sample.
    if (jacobian > 10.0 || jacobian < 1 / 10.0) {
        return false;
    }

    // clamp Jacobian.
    jacobian = clamp(jacobian, 1 / 3.0, 3.0);

    return true;
}

float RAB_GetGISampleTargetPdfForSurface(float3 samplePosition, float3 sampleRadiance, RAB_Surface surface)
{
    float3 L = normalize(samplePosition - surface.GetPosW());

    float3 reflectedRadiance = surface.Eval(L) * sampleRadiance;
    return max(0, luminance(reflectedRadiance));
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, float3 samplePosition)
{
    const RayDesc ray = setupVisibilityRay(surface, samplePosition, g_RtxdiBridgeConst.rayEpsilon);

    return GetConservativeVisibility(SceneBVH, ray);
}

bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, float3 samplePosition)
{
    const RayDesc ray = setupVisibilityRay(currentSurface, samplePosition, g_RtxdiBridgeConst.rayEpsilon);

    return GetConservativeVisibility(SceneBVH, ray);
}

RAB_Surface RAB_GetMaterial(RAB_Surface currentSurface)
{
    return currentSurface;
}

#endif // RTXDI_WITH_RESTIR_GI

#endif // #if !defined(RTXPT_RTXDI_RESOURCES_ONLY)

#endif // RTXDI_APPLICATION_BRIDGE_HLSLI
