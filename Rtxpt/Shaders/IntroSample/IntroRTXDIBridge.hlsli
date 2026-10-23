/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_RTXDI_BRIDGE_HLSLI__
#define __INTRO_RTXDI_BRIDGE_HLSLI__

// This first learning step uses RTXDI's uniform local-light Streaming RIS path,
// extended with a second RTXDI infinite-light region for analytic directional
// disk lights. Unlike the full sample, the intro path does NOT bake directional
// lights into the environment cubemap - they are sampled analytically here.
// Keep presampled RIS/ReGIR disabled so the bridge does not need those resources.
#ifndef RTXDI_ENABLE_PRESAMPLING
#define RTXDI_ENABLE_PRESAMPLING 0
#endif

// RTXDI_WITH_RESTIR_GI must precede the RTXDI headers; this is what RTXDI/ShaderParameters.h (not deployed with
// Rtxpt/Shaders, so not includable here) provided
#define RTXDI_WITH_RESTIR_GI
#include <Rtxdi/DI/ReSTIRDIParameters.h>
#include "IntroReSTIRDIConstants.hlsli"

// Emissive triangles are baked into the local region and sampled only by ReSTIR DI (Step 8).
// A pass that samples analytic locals only (e.g. Streaming RIS / Step 7) can compile the triangle
// light type out - it never selects a triangle, so the dispatch branch is dead code there - by
// defining INTRO_RTXDI_ENABLE_TRIANGLE_LIGHTS 0 before including this header.
#ifndef INTRO_RTXDI_ENABLE_TRIANGLE_LIGHTS
#define INTRO_RTXDI_ENABLE_TRIANGLE_LIGHTS 1
#endif

// ReSTIR DI (Step 8) can additionally importance-sample the environment map as a synthetic kEnvironment
// light. Passes that want it define INTRO_RTXDI_ENABLE_ENV_LIGHT 1 before including this header, which
// compiles in the env polymorphic light type + its CalcSample implementation (IntroReSTIRDIEnvLight.hlsli).
// Other passes leave it 0 and pay nothing.
#ifndef INTRO_RTXDI_ENABLE_ENV_LIGHT
#define INTRO_RTXDI_ENABLE_ENV_LIGHT 0
#endif

#define POLYLIGHT_OVERRIDE_CONFIG
#define POLYLIGHT_SPHERE_ENABLE         1
#define POLYLIGHT_POINT_ENABLE          1
#define POLYLIGHT_TRIANGLE_ENABLE       INTRO_RTXDI_ENABLE_TRIANGLE_LIGHTS
#define POLYLIGHT_DIRECTIONAL_ENABLE    1
#define POLYLIGHT_ENV_ENABLE            INTRO_RTXDI_ENABLE_ENV_LIGHT
#define POLYLIGHT_QT_ENV_ENABLE         0
#define POLYLIGHT_LIGHT_CLUSTER         0
#define POLYLIGHT_CONFIGURED
#include "../PathTracer/Lighting/PolymorphicLight.hlsli"

// Split into two parallel SoA buffers (Base + Extended) to mirror LightsBaker's t_Lights /
// t_LightsEx layout. RAB_LoadLightInfo composes the full struct on demand; the StochasticLightingPass
// only reads Base since directional / spherical analytic light sampling never touches Extended.
StructuredBuffer<PolymorphicLightInfo>   t_IntroRTXDILightsBase : register(t90);
StructuredBuffer<PolymorphicLightInfoEx> t_IntroRTXDILightsEx   : register(t91);

RWStructuredBuffer<RTXDI_PackedDIReservoir> u_LightReservoirs   : register(u94);
Buffer<float2>                              t_NeighborOffsets   : register(t94);
ConstantBuffer<IntroReSTIRDIConstants>      g_IntroReSTIRConst  : register(b12);

#define RTXDI_LIGHT_RESERVOIR_BUFFER u_LightReservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER t_NeighborOffsets

static const float kIntroRTXDIRayEpsilon = 1e-4;

#include "IntroSampleShadowRay.hlsli"

// Build the RTXDI initial sampling parameter block as a static constant.
// Uniform local-light sampling plus one infinite-light (directional) sample per pixel.
// No presampling, no environment / brdf samples.
static RTXDI_DIInitialSamplingParameters MakeIntroRTXDIInitialSamplingParameters()
{
    RTXDI_DIInitialSamplingParameters p;
    p.numLocalLightSamples = 8;
    p.numInfiniteLightSamples = 1;
    // Environment candidates only when the pass compiled in env support, the constant requests them, and
    // PrepareLights actually appended the env light this frame.
#if INTRO_RTXDI_ENABLE_ENV_LIGHT
    p.numEnvironmentSamples = (g_Const.lightControl.EnvironmentLightPresent != 0) ? g_IntroReSTIRConst.numEnvironmentSamples : 0;
    p.environmentMapImportanceSampling = (g_Const.lightControl.EnvironmentLightPresent != 0);
#else
    p.numEnvironmentSamples = 0;
    p.environmentMapImportanceSampling = false;
#endif
    p.numBrdfSamples = 0;
    p.brdfCutoff = 0.0001;
    p.brdfRayMinT = 0.001;
    p.localLightSamplingMode = ReSTIRDI_LocalLightSamplingMode_UNIFORM;
    p.enableInitialVisibility = true;
    return p;
}

// Full local region: analytic locals + baked emissive triangles, contiguous in memory.
// Used by ReSTIR DI (Step 8), whose temporal/spatial reuse amortizes sampling many triangle lights.
static RTXDI_LightBufferRegion MakeIntroRTXDILocalLightRegion()
{
    RTXDI_LightBufferRegion region;
    region.firstLightIndex = g_Const.lightControl.LocalRegionFirstLightIndex;
    region.numLights = g_Const.lightControl.LocalRegionNumLights;
    return region;
}

// Analytic-only local region: same start, but excludes the trailing triangles.
// Used by Streaming RIS (Step 7) so uniform RIS candidates aren't diluted across many triangle lights.
static RTXDI_LightBufferRegion MakeIntroRTXDILocalLightRegionAnalyticOnly()
{
    RTXDI_LightBufferRegion region;
    region.firstLightIndex = g_Const.lightControl.LocalRegionFirstLightIndex;
    region.numLights = g_Const.lightControl.LocalRegionNumAnalyticLights;
    return region;
}

// Triangle-only local region: the trailing emissive-triangle block of the local region [A | T].
// ReSTIR DI (Step 8) samples this as a separate RIS stream from the analytic locals so a few
// spot/point lights don't have to compete against many triangle lights in one uniform draw.
static RTXDI_LightBufferRegion MakeIntroRTXDILocalLightRegionTriangleOnly()
{
    RTXDI_LightBufferRegion region;
    region.firstLightIndex = g_Const.lightControl.LocalRegionFirstLightIndex + g_Const.lightControl.LocalRegionNumAnalyticLights;
    region.numLights = g_Const.lightControl.LocalRegionNumLights - g_Const.lightControl.LocalRegionNumAnalyticLights;
    return region;
}

static RTXDI_LightBufferRegion MakeIntroRTXDIInfiniteLightRegion()
{
    RTXDI_LightBufferRegion region;
    region.firstLightIndex = g_Const.lightControl.InfiniteRegionFirstLightIndex;
    region.numLights = g_Const.lightControl.InfiniteRegionNumLights;
    return region;
}

struct IntroRTXDISurface
{
    float3 worldPos;
    float roughness;
    float3 normal;
    bool valid;
    float3 viewDir;
    float linearDepth;       // view-space Z; matches the .z component of u_MotionVectors
    float3 diffuseColor;
    float alpha2;
    float3 specularF0;
    float NdotV;
};

IntroRTXDISurface BuildIntroRTXDISurface(uint2 pixelPos, float depth, PackedGBufferSurface packedSurface)
{
    IntroRTXDISurface surface;
    surface.valid = depth > 0;
    surface.worldPos = ReconstructWorldPos(pixelPos, depth, g_Const.view.pixelOffset, g_Const.view.viewportSizeInv, g_Const.view.matClipToWorldNoOffset);
    surface.normal = normalize(packedSurface.GetSpecNormal());
    // View-space Z so the depth metric matches the motion-vector .z component used by
    // RTXDI temporal reprojection (motion.z = prevClipPos.w - clipPos.w; clipPos.w == viewPos.z).
    surface.linearDepth = mul(float4(surface.worldPos, 1), g_Const.view.matWorldToView).z;

    float3 rayDir = normalize(Bridge::computeCameraRay(pixelPos).dir);
    surface.viewDir = -rayDir;

    float3 baseColor = packedSurface.GetBaseColor();
    float metal = packedSurface.GetMetalnes();
    float roughness = packedSurface.GetRoughness();
    surface.roughness = roughness;

    if (packedSurface.GetShaderId() == ShaderIdGlassGGX)
    {
        surface.specularF0 = lerp(float3(0.08, 0.08, 0.08), baseColor, metal);
        surface.diffuseColor = 0;
    }
    else
    {
        surface.specularF0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metal);
        surface.diffuseColor = baseColor * (1 - metal) * (1 - surface.specularF0);
    }
    surface.NdotV = saturate(dot(surface.normal, surface.viewDir));

    float r2 = max(roughness * roughness, 1e-5);
    surface.alpha2 = r2 * r2;
    return surface;
}

// Build a current-frame RAB surface from the bound g-buffer textures. Used by
// the RTXDI SDK temporal/spatial paths via RAB_GetGBufferSurface(_, false).
IntroRTXDISurface BuildIntroRTXDISurfaceCurrent(int2 pixelPos)
{
    IntroRTXDISurface invalid = (IntroRTXDISurface)0;
    invalid.valid = false;

    int2 imageSize = int2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight);
    if (any(pixelPos < 0) || any(pixelPos >= imageSize))
        return invalid;

    PackedGBufferSurface packed;
    packed.LoadFromRenderTargets(uint2(pixelPos));
    float depth = u_Depth[uint2(pixelPos)];
    if (depth <= 0)
        return invalid;

    return BuildIntroRTXDISurface(uint2(pixelPos), depth, packed);
}

// Build a previous-frame RAB surface from history textures. The RTXDI temporal pass only
// reads worldPos / normal / linearDepth / valid from the temporal surface (see
// Rtxdi/DI/TemporalResampling.hlsli), so the other shading fields are left at their
// zero-initialized defaults to keep this cheap.
IntroRTXDISurface BuildIntroRTXDISurfacePrevious(int2 pixelPos)
{
    IntroRTXDISurface invalid = (IntroRTXDISurface)0;
    invalid.valid = false;

    int2 imageSize = int2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight);
    if (any(pixelPos < 0) || any(pixelPos >= imageSize))
        return invalid;

    float depth = t_PrevDepth[uint2(pixelPos)];
    if (depth <= 0)
        return invalid;

    IntroRTXDISurface surface = invalid;
    surface.worldPos = ReconstructWorldPos(uint2(pixelPos), depth,
        g_Const.previousView.pixelOffset,
        g_Const.previousView.viewportSizeInv,
        g_Const.previousView.matClipToWorldNoOffset);
    surface.linearDepth = mul(float4(surface.worldPos, 1), g_Const.previousView.matWorldToView).z;
    surface.normal = octToNdirUnorm32(t_PrevSpecNormal32[uint2(pixelPos)]);
    surface.valid = true;
    return surface;
}

struct IntroRTXDILightSample
{
    float3 position;
    float3 normal;
    float3 radiance;
    float solidAnglePdf;
    PolymorphicLightType lightType;
};

typedef IntroRTXDISurface RAB_Surface;
typedef PolymorphicLightInfoFull RAB_LightInfo;
typedef IntroRTXDILightSample RAB_LightSample;
typedef RNGState RAB_RandomSamplerState;
typedef RAB_RandomSamplerState RTXDI_RandomSamplerState;
// RAB_Material is the type passed to RAB_AreMaterialsSimilar by the SDK spatial paths.
// We just alias it to the full surface; the similarity check picks the few fields it
// needs (roughness, metalness via specularF0/diffuseColor) and ignores the rest.
typedef IntroRTXDISurface RAB_Material;

float RTXDI_GetNextRandom(inout RTXDI_RandomSamplerState rng)
{
    return rng.getNextUniformRand();
}

float RAB_GetNextRandom(inout RAB_RandomSamplerState rng)
{
    return rng.getNextUniformRand();
}

RAB_LightSample RAB_EmptyLightSample()
{
    return (RAB_LightSample)0;
}

RAB_LightInfo RAB_LoadLightInfo(uint index, bool previousFrame)
{
    return PolymorphicLightInfoFull::make(t_IntroRTXDILightsBase[index], t_IntroRTXDILightsEx[index]);
}

float3 RAB_GetSurfaceWorldPos(RAB_Surface surface)
{
    return surface.worldPos;
}

float3 RAB_GetSurfaceNormal(RAB_Surface surface)
{
    return surface.normal;
}

bool RAB_IsSurfaceValid(RAB_Surface surface)
{
    return surface.valid;
}

bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return lightSample.lightType == PolymorphicLightType::kPoint ||
        lightSample.lightType == PolymorphicLightType::kSphere ||
        lightSample.lightType == PolymorphicLightType::kDirectional;
}

float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
    return lightSample.solidAnglePdf;
}

void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample, out float3 lightDir, out float lightDistance)
{
    if (lightSample.lightType == PolymorphicLightType::kDirectional)
    {
        // For directional lights, CalcSample stashes the unit toLight vector directly in
        // lightSample.position (see DirectionalLight::CalcSample in PolymorphicLight.hlsli).
        // The light is at infinity, so use a large but TMax-safe distance for the shadow ray.
        lightDir = lightSample.position;
        lightDistance = 1e6;
        return;
    }

    float3 toLight = lightSample.position - surface.worldPos;
    lightDistance = length(toLight);
    lightDir = lightDistance > 0 ? toLight / lightDistance : 0;
}

RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
    PolymorphicLightSample pls = PolymorphicLight::CalcSample(lightInfo, uv, surface.worldPos);

    RAB_LightSample lightSample;
    lightSample.position = pls.Position;
    lightSample.normal = pls.Normal;
    lightSample.radiance = pls.Radiance;
    lightSample.solidAnglePdf = pls.SolidAnglePdf;
    lightSample.lightType = PolymorphicLight::DecodeType(lightInfo);
    return lightSample;
}

float3 IntroRTXDIEvaluateBRDF(RAB_Surface surface, float3 lightDir, float NdotL)
{
    float3 specular = CookTorranceBRDF(lightDir, surface.normal, surface.viewDir, surface.specularF0, surface.alpha2, NdotL, surface.NdotV);
    float3 diffuse = surface.diffuseColor * K_1_PI;
    return diffuse+specular;
}

float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    if (!surface.valid || lightSample.solidAnglePdf <= 0)
        return 0;

    float3 lightDir;
    float lightDistance;
    RAB_GetLightDirDistance(surface, lightSample, lightDir, lightDistance);

    float NdotL = saturate(dot(surface.normal, lightDir));
    if (NdotL <= 0)
        return 0;

#if 0
    float3 reflectedRadiance = IntroRTXDIEvaluateBRDF(surface, lightDir, NdotL) * lightSample.radiance * NdotL;
    return Luminance(reflectedRadiance) / lightSample.solidAnglePdf;
#else
    return dot(lightSample.radiance, 1) * NdotL / lightSample.solidAnglePdf;
#endif
}

float RAB_SurfaceEvaluateBrdfPdf(RAB_Surface surface, float3 dir)
{
    return 0;
}

bool RAB_SurfaceImportanceSampleBrdf(RAB_Surface surface, inout RAB_RandomSamplerState rng, out float3 dir)
{
    dir = 0;
    return false;
}

bool RAB_TraceRayForLocalLight(float3 origin, float3 direction, float minT, float maxT, out uint lightIndex, out float2 randXY)
{
    lightIndex = RTXDI_InvalidLightIndex;
    randXY = 0;
    return false;
}

float RAB_EvaluateLocalLightSourcePdf(uint lightIndex)
{
    return 0;
}

float2 RAB_GetEnvironmentMapRandXYFromDir(float3 worldDir)
{
    return 0;
}

float RAB_EvaluateEnvironmentMapSamplingPdf(float3 worldDir)
{
    return 0;
}

bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
    float3 lightDir;
    float lightDistance;
    RAB_GetLightDirDistance(surface, lightSample, lightDir, lightDistance);
    if (dot(surface.normal, lightDir) <= 0)
        return false;
    return TraceVisibilityRay(surface.worldPos + surface.normal * kIntroRTXDIRayEpsilon,
                          lightDir, max(kIntroRTXDIRayEpsilon, lightDistance - kIntroRTXDIRayEpsilon));
}

// =====================================================================================
// RTXDI SDK callbacks for temporal resampling (RAB_GetGBufferSurface, depth metric,
// light index translation, temporal visibility).
// =====================================================================================

RAB_Surface RAB_EmptySurface()
{
    RAB_Surface s = (RAB_Surface)0;
    s.valid = false;
    return s;
}

RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool previousFrame)
{
    if (previousFrame)
        return BuildIntroRTXDISurfacePrevious(pixelPosition);
    return BuildIntroRTXDISurfaceCurrent(pixelPosition);
}

float RAB_GetSurfaceLinearDepth(RAB_Surface surface)
{
    return surface.linearDepth;
}

// Reservoirs store absolute indices into alternating light-buffer halves. Preserve the
// offset within the half when translating to the other frame. The host resets history
// whenever the light layout changes, so matching offsets still identify the same light.
int RAB_TranslateLightIndex(uint lightIndex, bool currentToPrevious)
{
    const uint capacity = g_IntroReSTIRConst.lightRegionCapacity;
    if (g_IntroReSTIRConst.temporalHistoryValid == 0 || capacity == 0)
        return -1;

    const uint currentBase = g_Const.lightControl.InfiniteRegionFirstLightIndex;
    const uint sourceBase = currentToPrevious ? currentBase : capacity - currentBase;
    if (lightIndex < sourceBase || lightIndex - sourceBase >= g_IntroReSTIRConst.lightCount)
        return -1;

    return int(capacity - sourceBase + (lightIndex - sourceBase));
}

// Conservative visibility check used by raytraced temporal bias correction (enabled from the UI
// via the "Temporal Ray-Traced Bias Correction" toggle; Basic mode never invokes this). Trace a
// visibility ray for the temporal normalization term. We have no previous-frame acceleration
// structure, so we use the current surface and TLAS, as in the shared RTXDI bridge. This is an
// approximation under motion; final shading still tests visibility at the current surface.
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, RAB_LightSample lightSample)
{
    float3 lightDir;
    float lightDistance;
    RAB_GetLightDirDistance(currentSurface, lightSample, lightDir, lightDistance);
    if (dot(currentSurface.normal, lightDir) <= 0)
        return false;
    return TraceVisibilityRay(currentSurface.worldPos + currentSurface.normal * kIntroRTXDIRayEpsilon,
                          lightDir, max(kIntroRTXDIRayEpsilon, lightDistance - kIntroRTXDIRayEpsilon));
}

int2 RAB_ClampSamplePositionIntoView(int2 pixelPosition, bool previousFrame)
{
    return clamp(pixelPosition, int2(0, 0),
        int2(g_Const.ptConsts.imageWidth, g_Const.ptConsts.imageHeight) - 1);
}

// =====================================================================================
// RTXDI SDK callbacks for spatial resampling (RAB_GetMaterial / RAB_AreMaterialsSimilar).
// Spatial reuse runs in screen-space across the *current* frame G-buffer and combines
// neighboring pixels' reservoirs if their surfaces are similar enough. We expose the
// surface itself as the "material" - the similarity test below picks the fields it cares
// about (roughness + albedo) and the check is only invoked when
// sparams.enableMaterialSimilarityTest is true at runtime.
// =====================================================================================

RAB_Material RAB_GetMaterial(RAB_Surface surface)
{
    return surface;
}

RAB_LightInfo RAB_EmptyLightInfo()
{
    return (RAB_LightInfo)0;
}

bool RAB_AreMaterialsSimilar(RAB_Material a, RAB_Material b)
{
    const float roughnessThreshold = 0.5;
    const float reflectivityThreshold = 0.25;
    const float albedoThreshold = 0.25;

    if (abs(a.roughness - b.roughness) > roughnessThreshold * max(a.roughness, b.roughness) + 1e-3)
        return false;
    if (abs(Luminance(a.specularF0) - Luminance(b.specularF0)) > reflectivityThreshold)
        return false;
    if (abs(Luminance(a.diffuseColor) - Luminance(b.diffuseColor)) > albedoThreshold)
        return false;
    return true;
}

// Environment-light CalcSample implementation + importance-map resource (only when this pass opted in).
// Included last so PolymorphicLight's EnvironmentLight declaration and g_Const are already in scope.
#include "IntroReSTIRDIEnvLight.hlsli"

#endif // __INTRO_RTXDI_BRIDGE_HLSLI__
