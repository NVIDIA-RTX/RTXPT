/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __HOMOGENEOUS_VOLUME_SAMPLER_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __HOMOGENEOUS_VOLUME_SAMPLER_HLSLI__

#include "../../Config.h"    
#include "../../Scene/Material/HomogeneousVolumeData.hlsli"

/** Helper class for sampling homogeneous volumes.
*/
struct HomogeneousVolumeSampler
{
    struct DistanceSample
    {
        float t;                ///< Scatter distance.
        float3 sigmaS;          ///< Scattering coefficient.
        float3 sigmaT;          ///< Extinction coefficient.
        float3 channelProbs;    ///< Channel selection probabilities.

        /** Evaluate the path throughput based on whether a scattering event has happened or not.
            \param[in] hasScattered True if a scattering has happened.
            \param[in] surfaceT Distance to the surface if hasScattered == false.
            \return Returns the throughput.
        */
        float3 evalThroughput(const bool hasScattered, const float surfaceT)
        {
            const float evalT = hasScattered ? t : surfaceT;
            const float3 Tr = exp(-sigmaT * evalT);
            const float3 Trs = exp(-sigmaS * evalT);
            const float3 density = hasScattered ? Trs * sigmaS : Trs;
            const float pdf = dot(channelProbs, density);
            return hasScattered ? Tr * sigmaS / pdf : Tr / pdf;
        }
    };

    /** Sample a scattering distance.
        \param[in] sigmaA Absorption coefficient.
        \param[in] sigmaS Scattering coefficient.
        \param[in] thp Current path throughput (used to compute channel sampling probabilities).
        \param[in,out] sampleGenerator Sample generator.
        \param[out] ds Distance sample.
        \return Returns true if a scattering distance was sampled (false if medium does not scatter).
    */
    static bool sampleDistance(const float3 sigmaA, const float3 sigmaS, const float3 thp, inout SampleGenerator sampleGenerator, out DistanceSample ds)
    {
        if (all(sigmaS == 0.f)) return false;

        const float3 sigmaT = sigmaA + sigmaS;

        // Compute albedo (set to 1 if no scattering for hero selection).
        const float3 albedo = float3(
            sigmaS.r > 0.f ? sigmaS.r / sigmaT.r : 1.f,
            sigmaS.g > 0.f ? sigmaS.g / sigmaT.g : 1.f,
            sigmaS.b > 0.f ? sigmaS.b / sigmaT.b : 1.f
        );

        // Compute probabilities for selecting RGB channels for scatter distance sampling.
        float3 channelProbs = thp * albedo;
        const float channelProbsSum = channelProbs.r + channelProbs.g + channelProbs.b;
        if (channelProbsSum < 1e-8f)
        {
            channelProbs = float3(1.f,1.f,1.f) / 3.f;
        }
        else
        {
            channelProbs /= channelProbsSum;
        }

        // Sample RGB channel.
        const float xi = sampleNext1D(sampleGenerator);
        uint channel;
        if (xi < channelProbs.r)
        {
            channel = 0;
        }
        else if (xi < channelProbs.r + channelProbs.g)
        {
            channel = 1;
        }
        else
        {
            channel = 2;
        }

        const float u = sampleNext1D(sampleGenerator);
        const float t = -log(1.f - u) / sigmaS[channel];

        // Return distance sample.
        ds.t = t;
        ds.sigmaS = sigmaS;
        ds.sigmaT = sigmaT;
        ds.channelProbs = channelProbs;

        return true;
    }

    /** Sample a scattering distance.
        \param[in] vd Medium properties.
        \param[in] thp Current path throughput (used to compute channel sampling probabilities).
        \param[in,out] sampleGenerator Sample generator.
        \param[out] ds Distance sample.
        \return Returns true if a scattering distance was sampled (false if medium does not scatter).
    */
    static bool sampleDistance(const HomogeneousVolumeData vd, const float3 thp, inout SampleGenerator sampleGenerator, out DistanceSample ds)
    {
        return sampleDistance(vd.sigmaA, vd.sigmaS, thp, sampleGenerator, ds);
    }

    /** Evaluate transmittance through a homogeneous medium.
        \param[in] sigmaT Extinction coefficient.
        \param[in] distance Distance through the medium.
        \return Returns the transmittance.
    */
    static float3 evalTransmittance(const float3 sigmaT, const float distance)
    {
        return exp(-distance * sigmaT);
    }

    /** Evaluate transmittance through a homogeneous medium.
        \param[in] vd Medium properties.
        \param[in] distance Distance through the medium.
        \return Returns the transmittance.
    */
    static float3 evalTransmittance(const HomogeneousVolumeData vd, const float distance)
    {
        return evalTransmittance(vd.sigmaA + vd.sigmaS, distance);
    }
};

#endif // __HOMOGENEOUS_VOLUME_SAMPLER_HLSLI__