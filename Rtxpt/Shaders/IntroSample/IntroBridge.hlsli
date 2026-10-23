/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_BRIDGE_HLSLI__
#define __INTRO_BRIDGE_HLSLI__

#include "IntroLightBindings.hlsli"
#include "../DonutBridgeCommon.hlsli"

#if (__SHADER_TARGET_MAJOR > 6) || (__SHADER_TARGET_MAJOR == 6 && __SHADER_TARGET_MINOR >= 9)
#define RTXPT_FLAG_ALLOW_OPACITY_MICROMAPS RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS
#else
#define RTXPT_FLAG_ALLOW_OPACITY_MICROMAPS 0
#endif

#endif // __INTRO_BRIDGE_HLSLI__
