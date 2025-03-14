/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "MathConstants.hlsli"

#ifndef __QUATERNION_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __QUATERNION_HLSLI__

#include "../../Config.h"    

 /**************************************************************************************************************
  # All the quaternion code was copied from: https://gist.github.com/mattatz/40a91588d5fb38240403f198a938a593
  **************************************************************************************************************/

#define QUATERNION_IDENTITY float4(0, 0, 0, 1)

// Quaternion multiplication
// http://mathworld.wolfram.com/Quaternion.html
float4 qmul(float4 q1, float4 q2)
{
    return float4(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}

// Vector rotation with a quaternion
// http://mathworld.wolfram.com/Quaternion.html
float3 rotate_vector(float3 v, float4 r)
{
    float4 r_c = r * float4(-1, -1, -1, 1);
    return qmul(r, qmul(float4(v, 0), r_c)).xyz;
}

// A given angle of rotation about a given axis
float4 rotate_angle_axis(float angle, float3 axis)
{
    float sn = sin(angle * 0.5);
    float cs = cos(angle * 0.5);
    return float4(axis * sn, cs);
}

// https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
float4 from_to_rotation(float3 v1, float3 v2)
{
    float4 q;
    float d = dot(v1, v2);
    if (d < -0.999999)
    {
        float3 right = float3(1, 0, 0);
        float3 up = float3(0, 1, 0);
        float3 tmp = cross(right, v1);
        if (length(tmp) < 0.000001)
        {
            tmp = cross(up, v1);
        }
        tmp = normalize(tmp);
        q = rotate_angle_axis(K_PI, tmp);
    }
    else if (d > 0.999999) {
        q = QUATERNION_IDENTITY;
    }
    else {
        q.xyz = cross(v1, v2);
        q.w = 1 + d;
        q = normalize(q);
    }
    return q;
}

#endif // __QUATERNION_HLSLI__
