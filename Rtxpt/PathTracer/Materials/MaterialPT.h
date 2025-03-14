/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_MATERIAL_H__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_MATERIAL_H__

/// Max number of materials - could be dynamic but isn't for simplicity
#define RTXPT_MATERIAL_MAX_COUNT        32768

// using https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_volume#attenuation convention
struct VolumePTConstants
{
    float3  AttenuationColor;
    float   AttenuationDistance;
};

static const int MaterialPTFlags_UseSpecularGlossModel          = 0x00000001;
//static const int MaterialPTFlags_DoubleSided                    = 0x00000002;
static const int MaterialPTFlags_UseMetalRoughOrSpecularTexture = 0x00000004;
static const int MaterialPTFlags_UseBaseOrDiffuseTexture        = 0x00000008;
static const int MaterialPTFlags_UseEmissiveTexture             = 0x00000010;
static const int MaterialPTFlags_UseNormalTexture               = 0x00000020;
//static const int MaterialPTFlags_UseOcclusionTexture            = 0x00000040;
static const int MaterialPTFlags_UseTransmissionTexture         = 0x00000080;
static const int MaterialPTFlags_MetalnessInRedChannel          = 0x00000100;
static const int MaterialPTFlags_ThinSurface                    = 0x00000200;
static const int MaterialPTFlags_PSDExclude                     = 0x00000400;
static const int MaterialPTFlags_NestedPriorityMask             = 0xF0000000;
static const int MaterialPTFlags_NestedPriorityShift            = 28;
static const int MaterialPTFlags_PSDDominantDeltaLobeP1Mask     = 0x0F000000;
static const int MaterialPTFlags_PSDDominantDeltaLobeP1Shift    = 24;

/// Data with the packed layout in GPU memory
struct MaterialPTData
{
    float3      BaseOrDiffuseColor;
    uint        Flags;

    float3      SpecularColor;
    int         _padding0;

    float3      EmissiveColor;
    float       ShadowNoLFadeout;

    float       Opacity;
    float       Roughness;
    float       Metalness;
    float       NormalTextureScale;

    float       _padding1;
    float       AlphaCutoff;
    float       TransmissionFactor;
    uint        BaseOrDiffuseTextureIndex;

    uint        MetalRoughOrSpecularTextureIndex;
    uint        EmissiveTextureIndex;
    uint        NormalTextureIndex;
    uint        OcclusionTextureIndex;

    uint        TransmissionTextureIndex;
    float       IoR;
    float       ThicknessFactor;
    float       DiffuseTransmissionFactor;

    VolumePTConstants Volume;
};

#if defined(__cplusplus)
#endif

#endif

 