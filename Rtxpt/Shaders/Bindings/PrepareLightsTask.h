/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PREPARE_LIGHTS_TASK_H__
#define __PREPARE_LIGHTS_TASK_H__

// RTXDI light-prep task, shared by RTXDI/ShaderParameters.h and the IntroSample's runtime-compiled shaders, which can only
// include what is deployed with Rtxpt/Shaders
#define TASK_PRIMITIVE_LIGHT_BIT 0x80000000u

struct PrepareLightsTask
{
    uint instanceAndGeometryIndex; // low 12 bits are geometryIndex, mid 19 bits are instanceIndex, high bit is TASK_PRIMITIVE_LIGHT_BIT
    uint triangleCount;
    uint lightBufferOffset;
    int previousLightBufferOffset; // -1 means no previous data
};

#endif // __PREPARE_LIGHTS_TASK_H__
