/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __LIGHTING_CONFIG_H__
#define __LIGHTING_CONFIG_H__

#ifndef POLYLIGHT_OVERRIDE_CONFIG

// Polymorphic light config - RTXDI will also need ENV 
#define POLYLIGHT_SPHERE_ENABLE         1
#define POLYLIGHT_POINT_ENABLE          0   // handled by sphere
#define POLYLIGHT_TRIANGLE_ENABLE       1
#define POLYLIGHT_DIRECTIONAL_ENABLE    0   // baked into envmap (Distant lighting code)
#define POLYLIGHT_ENV_ENABLE            0   // handled by Distant lighting code, not polymorphic light
#define POLYLIGHT_QT_ENV_ENABLE         1   // environment map quad tree in equal area octahedral mapping

#define POLYLIGHT_CONFIGURED

#endif

#endif // #define __LIGHTING_CONFIG_H__