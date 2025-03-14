/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef TONEMAPPING_CB_H
#define TONEMAPPING_CB_H

#define TONEMAPPING_AUTOEXPOSURE_CPU     1
#define TONEMAPPING_EXPOSURE_KEY         0.042

enum class ToneMapperOperator : uint32_t
{
	Linear,             ///< Linear mapping
	Reinhard,           ///< Reinhard operator
	ReinhardModified,   ///< Reinhard operator with maximum white intensity
	HejiHableAlu,       ///< John Hable's ALU approximation of Jim Heji's filmic operator
	HableUc2,           ///< John Hable's filmic tone-mapping used in Uncharted 2
	Aces,               ///< Aces Filmic Tone-Mapping
};


struct ToneMappingConstants
{
    float whiteScale;
    float whiteMaxLuminance;
	uint toneMapOperator;
    uint clamped;
	uint autoExposure;
	float avgLuminance;
	float autoExposureLumValueMin;
	float autoExposureLumValueMax;
    float3x4 colorTransform;
};


#endif // TONEMAPPING_CB_H