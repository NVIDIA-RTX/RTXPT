/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "PathTracer/Config.h" // must always be included first

#include "SERUtils.hlsli"

#if PATH_TRACER_MODE==PATH_TRACER_MODE_BUILD_STABLE_PLANES
#define SER_USE_SORTING 0
#else
#define SER_USE_SORTING 1
#endif

#include "PathTracer/PathTracerTypes.hlsli"

#include "Bindings/ShaderResourceBindings.hlsli"
#if PT_USE_RESTIR_GI
#include "Bindings/ReSTIRBindings.hlsli"
#endif

#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"
#include "Bindings/GBufferBindings.hlsli"
#include "IntroCommon.hlsli"

struct PathContextLite
{
    float3 rayOrigin;
    float3 rayDirection;
    float3 throughput;
    float curBounce;
    float3 accumRadiance;
    uint rngSeed;

    void initRngSeed(uint2 pixelPos)
    {
        uint frameIndex = g_Const.ptConsts.frameIndex;
        rngSeed = Hash32Combine(Hash32(frameIndex), (pixelPos.x << 16) | pixelPos.y);
    }

    float getNextUniformRand()
    {
        uint u1 = Hash32(rngSeed);
        rngSeed = u1;

        return ToFloat0To1(u1);
    }

    float2 getNextUniformRand2()
    {
        uint u1 = Hash32(rngSeed);
        uint u2 = Hash32(u1);
        rngSeed = u2;

        return float2(ToFloat0To1(u1), ToFloat0To1(u2));
    }
};

PathContextLite InitEmptyPath(uint2 pixelPos)
{
    PathContextLite pathContext;

    pathContext.throughput = 1;
    pathContext.accumRadiance = 0;
    pathContext.curBounce = 0; // TODO: Allow starting the ray from an existing G-Buffer instead of tracing primary visibility.

    Ray cameraRay = Bridge::computeCameraRay(pixelPos); // note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)
    pathContext.rayDirection = cameraRay.dir;
    pathContext.rayOrigin = cameraRay.origin;

    pathContext.initRngSeed(pixelPos);

    return pathContext;
}

// ============================================================================
// NEE (Next Event Estimation) helpers for analytical lights
// ============================================================================

// Unpack radiance from log-encoded uint16
float NEE_UnpackLightRadiance(uint logRadiance)
{
    return (logRadiance == 0) ? 0 : exp2((float(logRadiance - 1) / 65534.0) * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance);
}

// Unpack RGB color and radiance from polymorphic light
float3 NEE_UnpackLightColor(PolymorphicLightInfo lightInfo)
{
    float3 color = Unpack_R8G8B8_UFLOAT(lightInfo.ColorTypeAndFlags);
    float radiance = NEE_UnpackLightRadiance(lightInfo.LogRadiance & 0xffff);
    return color * radiance;
}

// Decode the polymorphic light type from the packed flags
PolymorphicLightType NEE_DecodeLightType(PolymorphicLightInfo lightInfo)
{
    uint typeCode = (lightInfo.ColorTypeAndFlags >> kPolymorphicLightTypeShift) & kPolymorphicLightTypeMask;
    return (PolymorphicLightType)typeCode;
}

// Evaluate spot light cone attenuation
float NEE_EvaluateSpotAttenuation(PolymorphicLightInfo lightBase, PolymorphicLightInfoEx lightEx, float3 worldPos)
{
    bool hasShaping = (lightBase.ColorTypeAndFlags & kPolymorphicLightShapingEnableBit) != 0;
    if (!hasShaping)
        return 1.0;

    float3 spotDir = OctToNDirUnorm32(lightEx.PrimaryAxis);
    float cosConeAngle = f16tof32(lightEx.CosConeAngleAndSoftness);
    float coneSoftness = f16tof32(lightEx.CosConeAngleAndSoftness >> 16);
    float minFalloff = (lightBase.ColorTypeAndFlags & kPolymorphicLightShapingUseMinFalloff) ? kMinSpotlightFalloff : 0.0;

    float3 lightToSurface = normalize(worldPos - lightBase.Center);
    float cosTheta = dot(spotDir, lightToSurface);
    float smoothFalloff = smoothstep(cosConeAngle, cosConeAngle + coneSoftness, cosTheta);

    return max(minFalloff, smoothFalloff);
}

// Trace a shadow ray using inline RayQuery. Returns true if the path is unoccluded.
bool TraceShadowRay(float3 origin, float3 direction, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0;
    ray.TMax = distance;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(SceneBVH, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
    query.Proceed();

    return query.CommittedStatus() == COMMITTED_NOTHING;
}

// Evaluate NEE: pick one random analytic light, trace shadow ray, shade if visible.
// Returns the direct lighting contribution (unweighted by throughput).
float3 EvaluateNEE(float3 hitPos, float3 normal, float3 viewDir,
                   float3 baseColor, float roughness, float metallic,
                   float3 F0, float hitDistance, inout PathContextLite pathContext)
{
    LightingControlData ctrl = t_LightsCB[0];
    uint numLights = ctrl.AnalyticLightCount;
    if (numLights == 0)
        return 0;

    uint analyticStart = ctrl.EnvmapQuadNodeCount;

    // Pick one light uniformly at random
    uint lightIndex = analyticStart + min(uint(pathContext.getNextUniformRand() * numLights), numLights - 1);

    PolymorphicLightInfo lightBase = t_Lights[lightIndex];
    PolymorphicLightInfoEx lightEx = t_LightsEx[lightIndex];

    // Light direction and distance
    float3 lightPos = lightBase.Center;
    float3 toLight = lightPos - hitPos;
    float dist2 = dot(toLight, toLight);
    float dist = sqrt(dist2);
    float3 L = toLight / dist;

    float NdotL = dot(normal, L);
    if (NdotL <= 0)
        return 0;
    NdotL = saturate(NdotL);

    // Spot cone attenuation
    float spotAtten = NEE_EvaluateSpotAttenuation(lightBase, lightEx, hitPos);
    if (spotAtten <= 0)
        return 0;

    // Shadow ray (offset origin along normal to prevent self-intersection)
    float3 shadowOrigin = hitPos + max(1e-4, 1e-3 * hitDistance) * normal;
    float shadowDist = dist - max(1e-4, 1e-3 * hitDistance);
    if (shadowDist <= 0)
        return 0;

    if (!TraceShadowRay(shadowOrigin, L, shadowDist))
        return 0; // Occluded

    // Compute incident irradiance based on light type
    float3 lightColor = NEE_UnpackLightColor(lightBase);
    float3 irradiance;

    PolymorphicLightType lightType = NEE_DecodeLightType(lightBase);
    if (lightType == PolymorphicLightType::kPoint)
    {
        irradiance = lightColor / dist2;
    }
    else // kSphere
    {
        float radius = f16tof32(lightBase.Scalars);
        float sinThetaMax2 = min((radius * radius) / dist2, 1.0);
        float solidAngle = 2.0 * K_PI * (1.0 - sqrt(max(0.0, 1.0 - sinThetaMax2)));
        irradiance = lightColor * solidAngle;
    }

    irradiance *= spotAtten;

    // --- Cook-Torrance BRDF evaluation ---
    float NdotV = saturate(dot(normal, viewDir));
    float a = roughness * roughness;
    float a2 = a * a;
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;

    float3 H = normalize(viewDir + L);
    float NdotH = saturate(dot(normal, H));
    float VdotH = saturate(dot(viewDir, H));

    // GGX NDF
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (K_PI * denom * denom);

    // Fresnel
    float3 F = F0 + (1.0 - F0) * pow(saturate(1.0 - VdotH), 5.0);

    // Smith's geometry term
    float G1V = NdotV / (NdotV * (1.0 - k) + k);
    float G1L = NdotL / (NdotL * (1.0 - k) + k);
    float G = G1V * G1L;

    // Specular BRDF
    float3 specular = (D * F * G) / max(4.0 * NdotV * NdotL, 1e-4);

    // Diffuse (Lambertian, energy-conserved)
    float3 kD = (1.0 - F) * (1.0 - metallic);
    float3 diffuse = kD * baseColor * K_1_PI;

    // Scale by number of lights (inverse of uniform selection probability)
    return float(numLights) * (diffuse + specular) * irradiance * NdotL;
}

[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    PathContextLite pathContext = InitEmptyPath(pixelPos);

    float3 motionVector = 0;
    const float NumBounces = 5; // 0 bounces means we just return the primary hit or miss
    [loop] // This is needed because otherwise the compiler will complain about unread payload members for the last loop iteration.
    for (float bounceNdx = 0; bounceNdx <= NumBounces; ++bounceNdx)
    {
        RayDesc rayDesc;
        rayDesc.Origin = pathContext.rayOrigin;
        rayDesc.Direction = pathContext.rayDirection;
        rayDesc.TMin = 1e-3;
        rayDesc.TMax = 10000;

        PayloadLite payload; // Intentionally uninitialized to reduce state across the tracing boundary.
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 1, 0, rayDesc, payload);

        // Bounce
        if (payload.hitDistance >= 0)
        {
            // Store motion vectors on primary hits
            if(bounceNdx == 0)
            {
                motionVector = payload.motionVector;
            }

            // Early out on emissives. 
            // This assumes that light emission itself is the strongest contribution on emissive bounces, but allows for a smaller payload.
            if (payload.shaderId == ShaderIdEmissive)
            {
                pathContext.accumRadiance += pathContext.throughput * payload.baseColor;
                break; // Early out on emissives
            }

            // Correct for back facing normals
            float3 normal = payload.normal;
            if (dot(pathContext.rayDirection, normal) > 0)
                normal = -normal;

            // --- NEE: Direct lighting from one random analytic light ---
            {
                float3 viewDir_nee = -pathContext.rayDirection;
                float3 F0_nee = lerp(0.04, payload.baseColor, payload.metal);
                pathContext.accumRadiance += pathContext.throughput *
                    EvaluateNEE(payload.worldPos, normal, viewDir_nee, payload.baseColor,
                                payload.roughness, payload.metal, F0_nee, payload.hitDistance, pathContext);
            }

            // Choose a microfacet normal
            float alpha = payload.roughness * payload.roughness;
            float invPdfVNDF = 1;
            float3 viewDir = -pathContext.rayDirection;
            float3 halfVector = SampleVNDF(pathContext.getNextUniformRand2(), viewDir, normal, alpha, invPdfVNDF);
            float3 reflDir = reflect(pathContext.rayDirection, halfVector);

            // Importance sampling between lobes
            float3 diffuseColor = payload.baseColor * (1 - payload.metal);
            float3 F0 = lerp(payload.shaderId == ShaderIdGlassGGX ? 0.08 : 0.04, payload.baseColor, payload.metal); // Using higher F0 to account for internal reflexions on glass.
            float3 F = Schlick_Fresnel(F0, dot(viewDir, halfVector));

            float specWeight = max(F.x, max(F.y, F.z));
            float diffWeight = max(diffuseColor.x, max(diffuseColor.y, diffuseColor.z));
            float misWeight = specWeight / (specWeight + diffWeight);
            
            // Stochastic sampling between the specular reflection lobe (top lobe) and the bottom lobe, which can either be diffuse or refraction.
            float rnd = pathContext.getNextUniformRand();
            if (rnd < misWeight) // Top lobe, specular reflection
            {                
                // Fresnel
                pathContext.throughput /= misWeight; // Lobe importance sampling inv. pdf.

                pathContext.rayDirection = reflDir;
                float G1 = G1_Smith(payload.roughness, dot(reflDir, normal));
                pathContext.throughput *= F * saturate(G1);
            }
            else // Bottom lobe
            {
                if(payload.shaderId == ShaderIdGlassGGX) // Glass
                {
                    // Keep the ray direction? TODO: Basic refraction
                    // Flip the normal so the next ray starts on the back side of the surface.
                    normal = -normal;
                    bounceNdx -= 0.75; // Extend bounces after going through transparency
                    pathContext.throughput *= diffuseColor;
                }
                else
                {
                    pathContext.rayDirection = normalize(normal + 0.995 * sample_sphere(pathContext.getNextUniformRand2()));
                }

                // Note: Applying ambient occlusion here isn't really the best idea, since it will also muffle emissive contributions, which
                // are really direct ligthing. This won't be an issue once we move to explicit ligh sampling.
                pathContext.throughput *= diffuseColor * (1-F) * (payload.ambientOcclusion / (1 - misWeight));

            }
            pathContext.rayOrigin = payload.worldPos + max(1e-4, 1e-3 * payload.hitDistance) * payload.normal;
        }
        else
        {

            float3 envRadiance = t_EnvironmentMap.SampleLevel(s_EnvironmentMapSampler, pathContext.rayDirection, 0).xyz;
            pathContext.accumRadiance += pathContext.throughput * envRadiance;

            break;
        }
    }

    float3 pathRadiance = pathContext.accumRadiance;

    u_MotionVectors[pixelPos] = float4(motionVector, 0);
    u_OutputColor[pixelPos] = float4(pathRadiance, 1);   // <- alpha 1 is important for screenshots
}

[shader("closesthit")]
void CLOSESTHIT_ENTRY(inout PayloadLite payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib)
{
    // Load geometry data from hit indices.
    DebugContext dummy;
    const DonutGeometryAttributes attrToLoad = GeomAttr_TexCoord | GeomAttr_Normal | GeomAttr_Position | GeomAttr_PrevPosition;;
    DonutGeometrySample donutGS = getGeometryFromHit(InstanceIndex(), GeometryIndex(), PrimitiveIndex(),
        attrib.barycentrics, attrToLoad,
        t_InstanceData, t_GeometryData, t_GeometryDebugData, WorldRayDirection(), dummy);

    // Extract geometry hit info
    payload.worldPos = mul(donutGS.instance.transform, float4(donutGS.objectSpacePosition, 1.0)).xyz;
    payload.motionVector = Bridge::computeMotionVector(donutGS.objectSpacePosition, donutGS.prevObjectSpacePosition);

    // Load material data    
    uint subInstanceDataIndex = donutGS.instance.firstGeometryInstanceIndex + GeometryIndex();
    uint materialIndex = t_SubInstanceData[subInstanceDataIndex].GlobalGeometryIndex_PTMaterialDataIndex & 0xFFFF;

    float3 geoNormal = donutGS.frontFacing ? donutGS.geometryNormal : -donutGS.geometryNormal;
    PBRSurface surface = EvalMaterialSurface(materialIndex, donutGS.texcoord, geoNormal);

    // Store surface info in the payload
    payload.baseColor = surface.emissiveScale > 0 ? surface.emissiveScale * surface.baseColor : surface.baseColor;
    payload.normal = surface.normal;
    payload.roughness = surface.roughness;
    payload.metal = surface.metal;
    payload.ambientOcclusion = 1;

    payload.shaderId = surface.emissiveScale > 0 ? ShaderIdEmissive : (surface.alpha < 1 ? ShaderIdGlassGGX : ShaderIdStandardGGX);
    payload.hitDistance = RayTCurrent();
}

[shader("miss")]
void MISS_ENTRY(inout PayloadLite payload : SV_RayPayload)
{
    payload.hitDistance = -1;
}

[shader("anyhit")]
void ANYHIT_ENTRY(inout PayloadLite payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib)
{
    if (!Bridge::AlphaTest(InstanceID(), InstanceIndex(), GeometryIndex(), PrimitiveIndex(), attrib.barycentrics))
        IgnoreHit();
}
