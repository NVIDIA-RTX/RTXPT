/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __BxDF_CONFIG_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __BxDF_CONFIG_HLSLI__

/** Static configuration of the BxDF models.

    The defaults can be overridden by passing in defines from the host.

    TODO: This file will be removed when we've settled on a new standard material definition.
*/

#define DiffuseBrdfLambert      0
#define DiffuseBrdfDisney       1
#define DiffuseBrdfFrostbite    2

#ifndef DiffuseBrdf
#define DiffuseBrdf DiffuseBrdfFrostbite
#endif

#define SpecularMaskingFunctionSmithGGXSeparable    0       ///< Used by UE4.
#define SpecularMaskingFunctionSmithGGXCorrelated   1       ///< Used by Frostbite. This is the more accurate form (default).

#ifndef SpecularMaskingFunction
#define SpecularMaskingFunction SpecularMaskingFunctionSmithGGXCorrelated
#endif

#endif // __BxDF_CONFIG_HLSLI__