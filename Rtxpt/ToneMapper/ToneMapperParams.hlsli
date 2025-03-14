/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once
#include "../PathTracer/Utils/HostDeviceShared.hlsli"

enum class ToneMapperOperator : uint32_t
{
    Linear,             ///< Linear mapping
    Reinhard,           ///< Reinhard operator
    ReinhardModified,   ///< Reinhard operator with maximum white intensity
    HejiHableAlu,       ///< John Hable's ALU approximation of Jim Heji's filmic operator
    HableUc2,           ///< John Hable's filmic tone-mapping used in Uncharted 2
    Aces,               ///< Aces Filmic Tone-Mapping
};

/** Tone mapper parameters shared between host and device.
    Make sure struct layout follows the HLSL packing rules as it is uploaded as a memory blob.
    Do not use bool's as they are 1 byte in Visual Studio, 4 bytes in HLSL.
    https://msdn.microsoft.com/en-us/library/windows/desktop/bb509632(v=vs.85).aspx
*/
struct ToneMapperParams
{
    float whiteScale;
    float whiteMaxLuminance;
    float _pad0;
    float _pad1;
    float3x4 colorTransform;
};

