/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTRO_LIGHT_BINDINGS_HLSLI__
#define __INTRO_LIGHT_BINDINGS_HLSLI__

#include "../PathTracer/Lighting/PolymorphicLight.h"

TextureCube<float4>                         t_EnvironmentMap : register(t10);
StructuredBuffer<PolymorphicLightInfo>      t_Lights         : register(t13);
StructuredBuffer<PolymorphicLightInfoEx>    t_LightsEx       : register(t14);

#endif // __INTRO_LIGHT_BINDINGS_HLSLI__
