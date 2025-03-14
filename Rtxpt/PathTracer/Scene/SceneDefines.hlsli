/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SCENE_DEFINES_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SCENE_DEFINES_HLSLI__

#include "../Config.h"    

#define GEOMETRY_TYPE_NONE                      0
#define GEOMETRY_TYPE_TRIANGLE_MESH             1
#define GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH   2
#define GEOMETRY_TYPE_CURVE                     3
#define GEOMETRY_TYPE_SDF_GRID                  4
#define GEOMETRY_TYPE_CUSTOM                    5

#define SCENE_HAS_GEOMETRY_TYPE(_type_) ((SCENE_GEOMETRY_TYPES & (1u << _type_)) != 0u)
#define SCENE_HAS_PROCEDURAL_GEOMETRY() ((SCENE_GEOMETRY_TYPES & ~(1u << GEOMETRY_TYPE_TRIANGLE_MESH)) != 0u)

#endif // __SCENE_DEFINES_HLSLI__