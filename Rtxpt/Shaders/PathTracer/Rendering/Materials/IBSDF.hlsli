/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __IBSDF_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __IBSDF_HLSLI__

#include "../../Config.h"    
#include "Microfacet.hlsli"
#include "LobeType.hlsli"

#ifndef RTXPT_DIFFUSE_SPECULAR_SPLIT
#define RTXPT_DIFFUSE_SPECULAR_SPLIT 1
#endif

static const uint       cMaxDeltaLobes      = 3;            // 3 should be enough (reflection, transmission, clearcoat reflection?) - there's a bit of a register use cost allowing for more than needed
// This represents delta lobe properties with respect to the surface Wi and surface properties (material settings, texture, normal map, etc.)
struct DeltaLobe
{
    float3  thp;                // how much light goes through the lobe with respect to the surface Wi and this->Wo; will be 0.xxx if probability == 0
    float   probability;        // chance this lobe is sampled with current BSDF importance sampling; will be 0 if disabled; 
    float3  dir;                // refracted or reflected direction in world space when returned from StandardBSDF (tangent space when returned from FalcorBSDF); will be 0.xxx if probability == 0; this is where the ray "will go" in unidirectional path tracing
    int     transmission;       // 1 when transmission lobe, 0 when reflection; even though it can be inferred from Wo, this avoids testing Wo vs triangle normal and potential precision issues

#if !defined(__cplusplus) // shader only!
    static DeltaLobe make()     { DeltaLobe ret; ret.thp = 0.xxx; ret.dir = 0.xxx; ret.transmission = false; ret.probability = 0; return ret; }
#endif
};

#if !defined(__cplusplus) // shader only!

/** Describes a BSDF sample.
*/
struct BSDFSample
{
    float3  wo;             ///< Sampled direction in world space (normalized).
    float   pdf;            ///< pdf with respect to solid angle for the sampled direction (wo).
    float3  weight;         ///< Sample weight f(wi, wo) * dot(wo, n) / pdf(wo).
    uint    lobe;           ///< Sampled lobe. This is a combination of LobeType flags (see LobeType.hlsli).
    float   lobeP;          ///< Probability that this lobe sample was picked (including each split between reflection/refraction).

    bool isLobe(LobeType type)
    {
        return (lobe & ((uint)type)) != 0;
    }

    // If delta lobe, returns an unique 2-bit delta lobe identifier (0...3); if not delta lobe returns 0xFFFFFFFF
    // NOTE: this ID must match delta lobe index used in IBSDF::evalDeltaLobes
    uint    getDeltaLobeIndex()
    { 
        if ((lobe & (uint)LobeType::Delta) == 0u)
            return 0xFFFFFFFF;
        return (lobe & (uint)LobeType::Transmission) == 0u;    // if transmission return 0, if reflection return 1; TODO: when clearcoat gets added, use 2 for clearcoat reflection
    }
};

/** Describes BSDF properties.

    These properties may be useful as input to denoisers and other perceptual techniques.
    The BSDF implementations are expected to provide a best fit approximation.
*/
struct BSDFProperties
{
    float3  emission;                       ///< Radiance emitted in the incident direction (wi).
    float   roughness;                      ///< Surface roughness on a perceptually linear scale, where 0.0 = perfectly smooth and 1.0 = maximum roughness.

    // Approximate directional-hemispherical reflectance/transmittance of the BSDF (black-sky albedo).
    // The exact values are given by the integrals of the BSDF over wo given an incident direction wi.
    // The terms are separated into diffuse and non-diffuse components. Due to energy conservation, the sum is expected to be <= 1.0.
    float3  diffuseReflectionAlbedo;        ///< Directional-hemispherical diffuse reflectance. This is the ratio of total energy diffusely reflected to the energy incident along wi.
    float3  diffuseTransmissionAlbedo;      ///< Directional-hemispherical diffuse transmittance. This is the ratio of total energy diffusely transmitted to the energy incident along wi.
    float3  specularReflectionAlbedo;       ///< Directional-hemispherical specular reflectance. This is the ratio of total energy non-diffusely reflected to the energy incident along wi.
    float3  specularTransmissionAlbedo;     ///< Directional-hemispherical specular transmittance. This is the ratio of total energy non-diffusely transmitted to the energy incident along wi.

    // Approximate specular reflectance. This is the color of specular reflection.
    // The diffuse reflectance is approximated as a Lambertian given by the diffuse albedo above.
    float3  specularReflectance;            ///< Specular reflectance at normal incidence (F0). This is in the range [0,1].

    uint    flags;                          ///< Flags storing additional properties.

    enum Flags : uint
    {
        IsTransmissive = 0x1,
    };

    bool isTransmissive( )         { return (flags & (uint)Flags::IsTransmissive) != 0; }

    // Surface BSDF integral estimate for demodulation - the closer it is to the actual bsdf, the less unnecessary blurring denoiser will do. It doesn't know the outgoing 
    // Note: we're not doing correct estimate for transmission - leaving this problem for the reader of these words. You're welcome.
    void estimateSpecDiffBSDF( out float3 outDiffEstimate, out float3 outSpecEstimate, const float3 normal, const float3 viewVector )
    {
    #if 1
        // Note - not clamping estimate here - it can be zero; clamp it at use location
        outDiffEstimate = diffuseReflectionAlbedo+diffuseTransmissionAlbedo; // note, also adding base path throughput to modulation here!
        const float NdotV = saturate(dot(normal, viewVector));
        const float ggxAlpha = roughness * roughness;
        float3 specularReflectance = approxSpecularIntegralGGX(specularReflectionAlbedo, ggxAlpha, NdotV); // note, also adding base path throughput to modulation here!
        specularReflectance += specularTransmissionAlbedo; // best approximation for now
        outSpecEstimate = specularReflectance;
     #else
        outDiffEstimate = float3(1,1,1);
        outSpecEstimate = float3(1,1,1);
     #endif
    }
};

#if 0

/** Interface for a bidirection scattering distribution function (BSDF).

    The term BSDF is used here in a broad sense for the mathematical function which
    describes the scattering of light at a shading location.

    This interface allows evaluation and sampling of the BSDF, and associated functionality.
    Implementations of the interface has all material properties evaluated at the shading location.
    BSDF instances are created and initialized by the corresponding material.
   
    The shading location and its attributes are described by a ShadingData struct.
    The ShadingData view direction field specifies the direction from which the
    shading location is seen. By convention we call this the incident direction (wi).
    The outgoing direction (wo) is the direction in which the transported quantity
    (radiance or importance) is scattered in.

    Conventions:
     - The incident and outgoing direction point away from the shading location.
     - The incident direction (wi) is given by ShadingData.
     - The outgoing direction (wo) is sampled.
     - The foreshortening term with respect to the sampled direction is always included.

    Note:
     - The [anyValueSize(n)] attribute specifies the maximum size in bytes an implementation type
       to IBSDF can be. Slang compiler will check the implementation types and emits an error
       if an implementation exceeds this size.
     - The maximum size can be increased if needed, but it should be kept as small as possible
       to reduce register pressure in case the compiler fails to optimize.
*/
[anyValueSize(68)] // TODO: Reduce to 64B
interface IBSDF
{
    /** Evaluates the BSDF.
        \param[in] sd Shading data.
        \param[in] wo Outgoing direction.
        \param[in,out] sg Sample generator.
        \return Returns f(wi, wo) * dot(wo, n).
    */
    float3 eval<S : ISampleGenerator>(const ShadingData shadingData, const float3 wo, inout S sg);

    /** Samples the BSDF.
        \param[in] sd Shading data.
        \param[in,out] sg Sample generator.
        \param[out] result Generated sample. Only valid if true is returned.
        \param[in] useImportanceSampling Hint to use importance sampling, else default to reference implementation if available.
        \return True if a sample was generated, false otherwise.
    */
    bool sample<S : ISampleGenerator>(const ShadingData shadingData, inout S sg, out BSDFSample result, bool useImportanceSampling = true);

    /** Evaluates the directional pdf for sampling the given direction.
        \param[in] sd Shading data.
        \param[in] wo Outgoing direction.
        \param[in] useImportanceSampling Hint to use importance sampling, else default to reference implementation if available.
        \return PDF with respect to solid angle for sampling direction wo (0 for delta events).
    */
    float evalPdf(const ShadingData shadingData, const float3 wo, bool useImportanceSampling = true);

    /** Return BSDF properties.
        \param[in] sd Shading data.
        \return A struct with properties.
    */
    BSDFProperties getProperties(const ShadingData shadingData);

    /** Return the set of available BSDF lobes.
        \param[in] sd Shading data.
        \return A combination of LobeType flags (see LobeType.hlsli).
    */
    uint getLobes(const ShadingData shadingData);

    // TODO add and explain - see StandardBSDF::evalDeltaLobes
    // void evalDeltaLobes(const ShadingData shadingData, out DeltaLobe deltaLobes[cMaxDeltaLobes], out int deltaLobeCount, out float nonDeltaPart)

}

#endif

#endif // #if !defined(__cplusplus)

#endif // __IBSDF_HLSLI__