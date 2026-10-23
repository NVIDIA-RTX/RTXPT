/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../PathTracer/Config.h" // must always be included first

#include "../SERUtils.hlsli"

#if PATH_TRACER_MODE==PT_DENOISING_LAYERS_BUILD
#define SER_USE_SORTING 0
#else
#define SER_USE_SORTING 1
#endif

#define ActiveBSDF StandardBSDF

#include "../PathTracer/PathTracerTypes.hlsli"

#include "../Bindings/ShaderResourceBindings.hlsli"
#if PT_USE_RESTIR_GI
#include "../Bindings/ReSTIRBindings.hlsli"
#endif

#include "IntroBridge.hlsli"
#include "../DonutGeometryHelpers.hlsli"
#include "../Bindings/GBufferBindings.hlsli"
#include "IntroCommon.hlsli"
#include "IntroSampleShadowRay.hlsli"

// Intro RTXDI light buffers (directional lights in the InfiniteRegion, emissive triangles in the trailing
// sub-range of the LocalRegion). Same buffers/registers the fast-hit passes read (see IntroHitLighting.hlsli);
// the reference PT's global binding set already binds these to m_RTXDILightsBase/Ex.
StructuredBuffer<PolymorphicLightInfo>   t_IntroRTXDILightsBase : register(t90);
StructuredBuffer<PolymorphicLightInfoEx> t_IntroRTXDILightsEx   : register(t91);

struct PathContextLite
{
    float3 rayOrigin;
    float3 rayDirection;
    float3 throughput;
    float curBounce;
    float3 accumRadiance;

    // True when the previous scatter was a diffuse reflection, whose emitter hits are covered by the
    // diffuse-only triangle NEE (so the BSDF-ray emission must be dropped to avoid double counting). False
    // after specular reflections (whose emitter reflections this BSDF ray integrates) and after
    // delta/transmissive scatters (glass refraction, diffuse transmission) that NEE cannot sample - those
    // keep their BSDF-ray emission. See the emissive early-out in RAYGEN_ENTRY.
    bool prevBounceNeeCovered;

    RNGState rng;
};

PathContextLite InitEmptyPath(uint2 pixelPos)
{
    PathContextLite pathContext;

    pathContext.throughput = 1;
    pathContext.accumRadiance = 0;
    pathContext.curBounce = 0; // TODO: Allow starting the ray from an existing G-Buffer instead of tracing primary visibility.
    pathContext.prevBounceNeeCovered = false; // primary ray: no prior scatter, so a primary emissive hit is kept

    Ray cameraRay = Bridge::computeCameraRay(pixelPos); // note: all realtime mode subSamples currently share same camera ray at subSampleIndex == 0 (otherwise denoising guidance buffers would be noisy)
    pathContext.rayDirection = cameraRay.dir;
    pathContext.rayOrigin = cameraRay.origin;

    uint frameIndex = g_Const.ptConsts.frameIndex;
    pathContext.rng.initRngSeed(pixelPos, frameIndex);

    return pathContext;
}

// Bound a per-sample radiance deposit to the reference firefly-filter threshold. Scales all channels by the
// same factor (Average-preserving) so hue is kept, and does nothing when threshold <= 0 (filter disabled).
// Trades a small, controlled bias for much lower variance from near-delta specular NEE and uniform light
// selection. threshold is g_Const.ptConsts.fireflyFilterThreshold (set from ReferenceFireflyFilterThreshold).
float3 FireflyClamp(float3 signal, float threshold)
{
    if (threshold <= 0)
        return signal;
    float avg = (signal.x + signal.y + signal.z) / 3.0;
    return (avg > threshold) ? signal * (threshold / avg) : signal;
}

// Evaluate NEE: explicit direct-light sampling. Draws one light from each populated group - analytic
// point/spot lights, directional (infinite) lights, and emissive triangle lights - shadow-tests it, and
// divides by the uniform selection pdf so the sum is an unbiased estimate of single-bounce direct lighting.
// Analytic and directional lights (no scene geometry, so unreachable by BSDF rays) use the full diffuse +
// Cook-Torrance specular BRDF. Emissive triangles use the DIFFUSE BRDF only: their specular reflection is
// integrated by the specular BSDF continuation ray instead (low variance on glossy surfaces; the near-delta
// specular NEE would firefly). The two strategies partition cleanly, so no MIS weight is needed - see the
// emissive early-out in RAYGEN_ENTRY, which drops the BSDF-ray emission only after diffuse-reflection bounces.
// Returns the direct lighting contribution (unweighted by throughput).
float3 EvaluateNEE(float3 hitPos, float3 normal, float3 viewDir,
                   float3 baseColor, float roughness, float metallic,
                   float3 F0, float hitDistance, inout PathContextLite pathContext)
{
    float3 result = 0;

    // Shared shading terms and shadow-ray bias for all light groups.
    const float NdotV = saturate(dot(normal, viewDir));
    const float r2 = max(roughness * roughness, 1e-5);
    const float alpha2 = r2 * r2;
    const float3 diffuse = (1.0 - metallic) * baseColor * K_1_PI;
    const float bias = max(1e-4, 1e-3 * hitDistance);
    const float3 shadowOrigin = hitPos + bias * normal;

    // --- One analytic point/spot light from the analytic sub-range of the local region ---
    {
        uint numLights = g_Const.lightControl.LocalRegionNumAnalyticLights;
        if (numLights > 0)
        {
            uint lightIndex = g_Const.lightControl.LocalRegionFirstLightIndex
                + min(uint(pathContext.rng.getNextUniformRand() * numLights), numLights - 1);

            PolymorphicLightInfo lightBase = t_IntroRTXDILightsBase[lightIndex];
            PolymorphicLightInfoEx lightEx = t_IntroRTXDILightsEx[lightIndex];

            float3 toLight = lightBase.Center - hitPos;
            float dist2 = dot(toLight, toLight);
            float dist = sqrt(dist2);
            float3 L = toLight / dist;

            float NdotL = dot(normal, L);
            float spotAtten = EvaluateSpotAttenuation(lightBase, lightEx, hitPos);
            float shadowDist = dist - bias;
            if (NdotL > 0 && spotAtten > 0 && shadowDist > 0 && TraceVisibilityRay(shadowOrigin, L, shadowDist))
            {
                NdotL = saturate(NdotL);
                float3 irradiance = EvaluatePolymorphicLightIrradiance(lightBase, dist2, spotAtten);
                float3 specular = CookTorranceBRDF(L, normal, viewDir, F0, alpha2, NdotL, NdotV);
                // Divide by the uniform selection pdf (1 / numLights).
                result += float(numLights) * (diffuse + specular) * irradiance * NdotL;
            }
        }
    }

    // --- One directional (infinite) light from the InfiniteRegion of the Intro RTXDI light buffers ---
    // Delta-ish disk light: no scene geometry, so no MIS partner. Mirrors IntroEvaluateHitDirectLighting.
    {
        uint numInfinite = g_Const.lightControl.InfiniteRegionNumLights;
        if (numInfinite > 0)
        {
            uint lightIndex = g_Const.lightControl.InfiniteRegionFirstLightIndex
                + min(uint(pathContext.rng.getNextUniformRand() * numInfinite), numInfinite - 1);

            PolymorphicLightInfo lightBase = t_IntroRTXDILightsBase[lightIndex];

            float3 lightDir = OctToNDirUnorm32(lightBase.Direction1); // light -> world travel direction
            float halfAngle = f16tof32(lightBase.Scalars);
            float solidAngle = f16tof32(lightBase.Scalars >> 16);
            float sinHalfAngle = sin(halfAngle);
            float3 radiance = UnpackLightColor(lightBase);

            // Sample the light's solid-angle disk uniformly.
            float2 diskSample = SampleDiskUniform(pathContext.rng.getNextUniformRand2());
            float3 t, b;
            BranchlessONB(lightDir, t, b);
            float3 distantSample = lightDir + (t * diskSample.x + b * diskSample.y) * sinHalfAngle;
            float3 L = normalize(-distantSample); // toLight

            float NdotL = dot(normal, L);
            if (NdotL > 0 && TraceVisibilityRay(shadowOrigin, L, 1e6))
            {
                NdotL = saturate(NdotL);
                float3 specular = CookTorranceBRDF(L, normal, viewDir, F0, alpha2, NdotL, NdotV);
                // Disk pdf = 1 / solidAngle cancels radiance = color / solidAngle; uniform pick over
                // numInfinite => multiply by numInfinite (divide by pick pdf = 1 / numInfinite).
                result += (diffuse + specular) * radiance * NdotL * solidAngle * numInfinite;
            }
        }
    }

    // --- One emissive triangle light from the triangle sub-range [A | T] of the LocalRegion ---
    // Triangles ARE scene geometry (BSDF rays can hit them), so this is the group the emissive early-out in
    // RAYGEN_ENTRY suppresses on NEE-covered bounces to avoid double counting.
    {
        uint numTri = g_Const.lightControl.LocalRegionNumLights - g_Const.lightControl.LocalRegionNumAnalyticLights;
        if (numTri > 0)
        {
            uint triFirst = g_Const.lightControl.LocalRegionFirstLightIndex + g_Const.lightControl.LocalRegionNumAnalyticLights;
            uint lightIndex = triFirst + min(uint(pathContext.rng.getNextUniformRand() * numTri), numTri - 1);

            PolymorphicLightInfo lightBase = t_IntroRTXDILightsBase[lightIndex];

            // Reconstruct the triangle from the packed light (mirrors TriangleLight::Create).
            float3 edge1 = f16tof32(uint3(lightBase.Direction1, lightBase.Direction2, lightBase.Scalars) & 0xffff);
            float3 edge2 = f16tof32(uint3(lightBase.Direction1, lightBase.Direction2, lightBase.Scalars) >> 16);
            float3 triBase = lightBase.Center - (edge1 + edge2) / 3.0;
            float3 triRadiance = UnpackLightColor(lightBase);

            float3 nCross = cross(edge1, edge2);
            float nLen = length(nCross);
            if (nLen > 0)
            {
                float triArea = 0.5 * nLen;
                float3 triNormal = nCross / nLen;

                // Uniform area sample on the triangle (mirrors TriangleLight::CalcSample).
                float3 bary = SampleTriangleUniform(pathContext.rng.getNextUniformRand2());
                float3 samplePos = ComputeRayOrigin(triBase + edge1 * bary.y + edge2 * bary.z, triNormal);

                float3 toLight = samplePos - hitPos;
                float dist2 = dot(toLight, toLight);
                float dist = sqrt(dist2);
                float3 L = toLight / dist;

                float cosThetaLight = dot(triNormal, -L); // single-sided emitter
                float NdotL = dot(normal, L);
                float shadowDist = dist - bias; // stop short of the emitter (emitters are excluded from NEE occlusion)
                if (cosThetaLight > 0 && NdotL > 0 && shadowDist > 0 && TraceVisibilityRay(shadowOrigin, L, shadowDist))
                {
                    NdotL = saturate(NdotL);
                    // Diffuse BRDF only. The specular reflection of an area emitter is integrated by the
                    // specular BSDF continuation ray (which keeps its emission on a hit); adding the near-delta
                    // specular NEE here would produce enormous fireflies on glossy surfaces.
                    // Solid-angle pdf of the area sample, times the uniform triangle selection pdf (1 / numTri).
                    float solidAnglePdf = pdfAtoW(1.0 / triArea, dist, cosThetaLight);
                    float lightPdf = solidAnglePdf / float(numTri);
                    result += diffuse * triRadiance * NdotL / lightPdf;
                }
            }
        }
    }

    return result;
}

[shader("raygeneration")]
void RAYGEN_ENTRY()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    PathContextLite pathContext = InitEmptyPath(pixelPos);
    
    EnvMap envMap = EnvMap::make(t_EnvironmentMap, s_EnvironmentMapSampler, g_Const.envMapSceneParams);

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
            // Material AO travels in the shared PayloadLite but is only consumed by the G-Buffer
            // pass. Reference it here so the shared payload's access qualifiers are satisfied
            // (mirrors the dummy reads in PopulateGBuffer.hlsl); this must not affect the path-traced result.
            if (payload.ao < 0.0)
                pathContext.throughput *= 1.0;

            // Store motion vectors on primary hits
            if(bounceNdx == 0)
            {
                motionVector = payload.motionVector;
            }

            // Early out on emissives.
            // The triangle NEE covers only the DIFFUSE direct lighting from emitters, so add the BSDF-ray
            // emission unless the previous bounce was a diffuse reflection that already accounted for it. That
            // is: keep it on the primary ray (camera sees the emitter directly), after a specular reflection
            // (triangle NEE is diffuse-only, so this ray carries the specular reflection of emitters), and
            // after a delta/transmissive scatter (glass refraction, diffuse transmission) that NEE cannot
            // sample; drop it only after a diffuse reflection (still terminating the path either way).
            // Assumption: every emissive surface is baked as a triangle NEE light (the common case). Materials
            // with the manual "Disable light sampling" opt-out (IsEmissiveSurfaceLightSampling()==false) are NOT
            // in the NEE set, so this slightly under-counts them on diffuse-reflection bounces. This
            // conservative choice never double-counts; there is no reliable GPU signal here to detect that
            // opt-out (EmissiveLightMappingOffset is the NEE-AT baker's mapping, not guaranteed for this path).
            if (payload.shaderId == ShaderIdEmissive)
            {
                if (bounceNdx == 0 || !pathContext.prevBounceNeeCovered)
                    pathContext.accumRadiance += FireflyClamp(pathContext.throughput * payload.baseColor, g_Const.ptConsts.fireflyFilterThreshold);
                break; // Early out on emissives
            }

            // Correct for back facing normals
            float3 normal = payload.normal;
            if (dot(pathContext.rayDirection, normal) > 0)
                normal = -normal;

            // --- NEE: explicit direct lighting (analytic + directional + emissive triangle lights) ---
            {
                float3 viewDir_nee = -pathContext.rayDirection;
                float3 F0_nee = lerp(0.04, payload.baseColor, payload.metal);
                pathContext.accumRadiance += FireflyClamp(pathContext.throughput *
                    EvaluateNEE(payload.worldPos, normal, viewDir_nee, payload.baseColor,
                                payload.roughness, payload.metal, F0_nee, payload.hitDistance, pathContext),
                    g_Const.ptConsts.fireflyFilterThreshold);
            }

            // Choose a microfacet normal
            float alpha = payload.roughness * payload.roughness;

            float3 viewDir = -pathContext.rayDirection;
            float3 halfVector = SampleVNDF(pathContext.rng.getNextUniformRand2(), viewDir, normal, alpha);
            float3 reflDir = reflect(pathContext.rayDirection, halfVector);

            // Importance sampling between lobes
            float3 diffuseColor = payload.baseColor * (1 - payload.metal);

            float3 F0 = lerp(payload.shaderId == ShaderIdGlassGGX ? 0.08 : 0.04, payload.baseColor, payload.metal); // Using higher F0 to account for internal reflexions on glass.
            // saturate VdotH: SampleVNDF can return a half-vector with dot(viewDir, halfVector) < 0 at grazing
            // angles; a negative VdotH makes Schlick_Fresnel blow up (pow(1-VdotH,5) >> 1) into huge/negative
            // per-channel F that corrupts the throughput.
            float3 F = Schlick_Fresnel(F0, saturate(dot(viewDir, halfVector)));

            // Probability of sampling the specular lobe, from metalness: metals reflect only specularly,
            // dielectrics have a weak specular lobe. Bounded to [0.2, 1] so the lobe inverse-pdfs stay finite
            // and the pick never degenerates. The Fresnel energy split (F on specular, 1-F on diffuse) applies
            // on top.
            float specMISProbability = lerp(0.2, 1, payload.metal);

            // Stochastic sampling between the specular reflection lobe (top lobe) and the bottom lobe, which can either be diffuse or refraction.
            float rnd = pathContext.rng.getNextUniformRand();
            if (rnd < specMISProbability) // Top lobe, specular reflection
            {
                pathContext.throughput /= specMISProbability; // Lobe selection inv. pdf.

                pathContext.rayDirection = reflDir;
                float NoL = saturate(dot(reflDir, normal));
                float NoV = saturate(dot(viewDir, normal));
                pathContext.throughput *= F * SmithGGX_G2_over_G1v(max(1e-4, payload.roughness), NoV, NoL);

                // Triangle NEE is diffuse-only, so this specular ray integrates the specular reflection of
                // area emitters: keep its emission on the next hit.
                pathContext.prevBounceNeeCovered = false;
            }
            else // Bottom lobe
            {
                if(payload.shaderId == ShaderIdGlassGGX) // Glass
                {
                    // Keep the ray direction? TODO: Basic refraction
                    // Flip the normal so the next ray starts on the back side of the surface.
                    normal = -normal;
                    bounceNdx -= 0.75; // Extend bounces after going through transparency

                    pathContext.throughput *= 1 - payload.opacity;

                    // Approx refraction
                    pathContext.rayDirection = normalize(pathContext.rayDirection + 0.5 * normal);

                    // Delta-ish transmission: NEE cannot sample through it, so keep the next emissive hit.
                    pathContext.prevBounceNeeCovered = false;
                }
                else
                {
                    pathContext.rayDirection = normalize(normal + 0.995 * sample_sphere(pathContext.rng.getNextUniformRand2()));
                    rnd = pathContext.rng.getNextUniformRand();
                    if(rnd < payload.diffTx)
                    {
                        pathContext.rayDirection = -pathContext.rayDirection;
                        normal = -normal;
                        // Diffuse transmission to the back hemisphere: NEE (front-hemisphere) does not cover it,
                        // so keep the next emissive hit.
                        pathContext.prevBounceNeeCovered = false;
                    }
                    else
                    {
                        // Front-hemisphere diffuse reflection: covered by NEE, drop the next emissive hit.
                        pathContext.prevBounceNeeCovered = true;
                    }
                }

                // Note: Applying ambient occlusion here isn't really the best idea, since it will also muffle emissive contributions, which
                // are really direct ligthing. This won't be an issue once we move to explicit ligh sampling.
                pathContext.throughput *= diffuseColor * (1 - F) / (1 - specMISProbability);
            }
            pathContext.rayOrigin = payload.worldPos + max(1e-4, 1e-3 * payload.hitDistance) * payload.normal;
        }
        else
        {
            float3 envRadiance = envMap.Eval(pathContext.rayDirection, 0.f);
            pathContext.accumRadiance += FireflyClamp(pathContext.throughput * envRadiance, g_Const.ptConsts.fireflyFilterThreshold);

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
    const DonutGeometryAttributes attrToLoad = GeomAttr_TexCoord | GeomAttr_Normal | GeomAttr_Tangents | GeomAttr_Position | GeomAttr_PrevPosition;
    DonutGeometrySample donutGS = getGeometryFromHit(InstanceIndex(), GeometryIndex(), PrimitiveIndex(),
        attrib.barycentrics, attrToLoad,
        t_InstanceData, t_GeometryData, t_GeometryDebugData, WorldRayDirection(), dummy);

    // Extract geometry hit info
    payload.worldPos = mul(donutGS.instance.transform, float4(donutGS.objectSpacePosition, 1.0)).xyz;
    float3 prevWorldPos = mul(donutGS.instance.prevTransform, float4(donutGS.prevObjectSpacePosition, 1.0)).xyz;
    payload.motionVector = Bridge::computeMotionVector(payload.worldPos, prevWorldPos);

    // Load material data    
    uint subInstanceDataIndex = donutGS.instance.firstGeometryInstanceIndex + GeometryIndex();
    uint materialIndex = t_SubInstanceData[subInstanceDataIndex].AlphaTextureIndex_PTMaterialDataIndex & 0xFFFF;

    // Compute shading normal: start from geometry normal, apply normal map if available
    float3 geoNormal = donutGS.frontFacing ? donutGS.geometryNormal : -donutGS.geometryNormal;
    float3 shadingNormal = ApplyNormalMap(materialIndex, donutGS.texcoord, geoNormal, donutGS.tangent);

    PBRSurface surface = EvalMaterialSurface(materialIndex, donutGS.texcoord);

    // Store surface info in the payload
    float3 emissiveColor = surface.emissiveColor;
    float maxEmissive = max(emissiveColor.x, max(emissiveColor.y, emissiveColor.z));
    payload.baseColor = maxEmissive > 0 ? emissiveColor : surface.baseColor;
    payload.normal = shadingNormal;
    payload.roughness = surface.roughness;
    payload.metal = surface.metal;

    payload.shaderId = maxEmissive > 0 ? ShaderIdEmissive : (surface.opacity < 1 ? ShaderIdGlassGGX : ShaderIdStandardGGX);
    payload.hitDistance = RayTCurrent();
    payload.diffTx = surface.diffTx;
    payload.opacity = surface.opacity;
    payload.ao = surface.ao;
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
