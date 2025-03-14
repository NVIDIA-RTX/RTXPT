/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Bindings/ShaderResourceBindings.hlsli"

// This define is meant to trigger a compile error if any resources are declared in any PathTracer files
#define register

#include "PathTracerBridgeNull.hlsli"
#include "PathTracer/PathTracer.hlsli"

[shader("raygeneration")]
void RayGen()
{
}