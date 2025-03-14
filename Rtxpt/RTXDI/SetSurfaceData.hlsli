/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef SET_SURFACE_DATA_HLSLI
#define SET_SURFACE_DATA_HLSLI

#include "SurfaceData.hlsli"
#include "ShaderParameters.h"

ConstantBuffer<RtxdiBridgeConstants> g_RtxdiBridgeConst     : register(b2 VK_DESCRIPTOR_SET(2));


#endif // SET_SURFACE_DATA_HLSLI
