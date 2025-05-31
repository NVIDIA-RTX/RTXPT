/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __HOMOGENEOUS_VOLUME_DATA_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __HOMOGENEOUS_VOLUME_DATA_HLSLI__

#include "../../Config.h"    

struct HomogeneousVolumeData
{
    float3 sigmaA;  ///< Absorption coefficient.
    float3 sigmaS;  ///< Scattering coefficient.
    float g;        ///< Phase function anisotropy.
};

#endif // __HOMOGENEOUS_VOLUME_DATA_HLSLI__