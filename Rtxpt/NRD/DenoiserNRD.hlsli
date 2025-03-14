/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __DENOISER_NRD_HLSLI__
#define __DENOISER_NRD_HLSLI__

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define NRD_HEADER_ONLY
#include <NRDEncoding.hlsli>
#if NRD_NORMAL_ENCODING != 2 // 2 == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM
#error not configured correctly
#endif
#include <NRD.hlsli>
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma pack_matrix(row_major)

RWTexture2D<float>  u_DenoiserViewspaceZ                    : register(u31);
RWTexture2D<float4> u_DenoiserMotionVectors                 : register(u32);
RWTexture2D<float4> u_DenoiserNormalRoughness               : register(u33);
RWTexture2D<float4> u_DenoiserDiffRadianceHitDist           : register(u34);
RWTexture2D<float4> u_DenoiserSpecRadianceHitDist           : register(u35);
RWTexture2D<float>  u_DenoiserDisocclusionThresholdMix      : register(u36);
RWTexture2D<float>  u_CombinedHistoryClampRelax             : register(u37);

namespace DenoiserNRD
{
    static float3 PostDenoiseProcess(const float3 diffBSDFEstimate, const float3 specBSDFEstimate, inout float4 diffRadianceHitDistDenoised, inout float4 specRadianceHitDistDenoised)
    {
#if USE_RELAX
        diffRadianceHitDistDenoised.xyz = RELAX_BackEnd_UnpackRadiance(diffRadianceHitDistDenoised).xyz;
        specRadianceHitDistDenoised.xyz = RELAX_BackEnd_UnpackRadiance(specRadianceHitDistDenoised).xyz;
#else
        diffRadianceHitDistDenoised.xyz = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(diffRadianceHitDistDenoised).xyz;
        specRadianceHitDistDenoised.xyz = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(specRadianceHitDistDenoised).xyz;
#endif

        diffRadianceHitDistDenoised.xyz *= diffBSDFEstimate;
        specRadianceHitDistDenoised.xyz *= specBSDFEstimate;

        return diffRadianceHitDistDenoised.xyz + specRadianceHitDistDenoised.xyz;
    }
} // namespace DenoiserNRD

#endif // #ifndef __DENOISER_NRD_HLSLI__