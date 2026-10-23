/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// The file MaterialsBaker compiles every material permutation from. Which set a pipeline gets is the
// sample's choice, passed as RTXPT_HIT_SHADER_SET in its PTPipelineBaker's hitGroupMacros. The two
// implementations use different ray payloads, so they cannot share a compile.
//
// The set names are defined below, so only files that include this one can test RTXPT_HIT_SHADER_SET;
// in anything else the name is undefined and an #if would silently see 0.

#define RTXPT_HIT_SET_MAIN          0
#define RTXPT_HIT_SET_SIMPLIFIED    1

#ifndef RTXPT_HIT_SHADER_SET
#define RTXPT_HIT_SHADER_SET RTXPT_HIT_SET_MAIN
#endif

#if RTXPT_HIT_SHADER_SET == RTXPT_HIT_SET_SIMPLIFIED
#include "FastHitShaders.hlsl"
#else
// Carries the raygen entry point too; it is compiled into the hit-group libraries and left unused.
#include "IntroPathTracer.hlsl"
#endif
