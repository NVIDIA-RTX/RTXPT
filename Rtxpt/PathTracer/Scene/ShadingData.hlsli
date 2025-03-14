/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SHADING_DATA_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SHADING_DATA_HLSLI__

#include "../Config.h"    

#include "../Scene/Material/MaterialData.hlsli"

#include "../PathTracerHelpers.hlsli"

// import Rendering.Materials.IBxDF;
// import Scene.Material.MaterialData;
// import Utils.Geometry.GeometryHelpers;

/** This struct holds information needed for shading a hit point.

    This includes:
    - Geometric properties of the surface.
    - Texture coordinates.
    - Material ID and header.
    - Opacity value for alpha testing.
    - Index of refraction of the surrounding medium.

    Based on a ShadingData struct, the material system can be queried
    for a BSDF instance at the hit using `gScene.materials.getBSDF()`.
    The BSDF has interfaces for sampling and evaluation, as well as for
    querying its properties at the hit.
*/
struct ShadingData
{
    // Geometry data
    float3      posW;                   ///< Shading hit position in world space. CANNOT be optimized to fp16.
    float3      faceNCorrected;         ///< Face normal in world space, corrected for front-facing flag (which also means always facing the viewer). CANNOT be optimized to fp16.
    float3      V;                      ///< Direction to the eye at shading hit. CANNOT be optimized to fp16 (enough only for very low screen res).
    float3      N;                      ///< Shading normal at shading hit. Can MAYBE be optimized to fp16.
    float3      T;                      ///< Shading tangent at shading hit. Can MAYBE be optimized to fp16.
    float3      B;                      ///< Shading bitangent at shading hit. Can MAYBE be optimized to fp16.
    lpfloat2    uv;                     ///< Texture mapping coordinates. CAN be optimized to fp16 without any issues.
    float3      vertexN;                ///< A.k.a. "geometry normal" - interpolated vertex normal, corrected for front-facing flag but not corrected for the ray view direction (can point away since interpolated). Could LIKELY be optimized to fp16.
    bool        frontFacing;            ///< True if primitive seen from the front-facing side, as determined by dot product between view direction and triangle face normal (faceN - not corrected). Should be packed into flags.

    // Material data
    MaterialHeader mtl;                 ///< Material header data.
    uint        materialID;             ///< Material ID at shading location - equivalent to MaterialPTData buffer index. Can be optimized to uint16 depending on RTXPT_MATERIAL_MAX_COUNT
    lpfloat     opacity;                ///< Opacity value in [0,1]. This is used for alpha testing. Can be optimized to fp16 without any issues.
    lpfloat     IoR;                    ///< Index of refraction for the medium on the front-facing side (i.e. "outside" the material at the hit). Can be optimized to fp16 without any issues.
    lpfloat     shadowNoLFadeout;       ///< See corresponding material setting. Can be optimized to fp16 without any issues.

    static ShadingData make()
    {
        ShadingData shadingData;
        shadingData.posW = 0;
        shadingData.V = 0;
        shadingData.N = 0;
        shadingData.T = 0;
        shadingData.B = 0;
        shadingData.uv = 0;
        shadingData.faceNCorrected = 0;
        shadingData.vertexN = 0;
        //shadingData.frontFacing = 0;
        // shadingData.curveRadius = 0;

        shadingData.mtl = MaterialHeader::make();
        shadingData.materialID = 0;
        shadingData.opacity = 0;
        shadingData.IoR = 0;
        shadingData.shadowNoLFadeout = 0;
        return shadingData;
    }

    // Utility functions

    /** Computes new ray origin based on the hit point to avoid self-intersection.
        The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
        Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.
        \param[in] viewside True if the origin should be on the view side (reflection) or false otherwise (transmission).
        \return Ray origin of the new ray.
    */
    float3 computeNewRayOrigin(bool viewside = true)
    {
        return ComputeRayOrigin(posW, viewside ? faceNCorrected : -faceNCorrected);
    }

    /** Transform vector from the local surface frame to world space.
        \param[in] v Vector in local space.
        \return Vector in world space.
    */
    float3 fromLocal(float3 v)
    {
        return T * v.x + B * v.y + N * v.z;
        //return mul( v, float3x3(T,B,N) );
    }

    /** Transform vector from world space to the local surface frame.
        \param[in] v Vector in world space.
        \return Vector in local space.
    */
    float3 toLocal(float3 v)
    {
        return float3(dot(v, T), dot(v, B), dot(v, N));
        //return mul( float3x3(T,B,N), v );
    }

    /** Returns the oriented face normal.
        \return Face normal flipped to the same side as the view vector.
    */
    float3 getOrientedFaceNormal()
    {
        return faceNCorrected; //frontFacing ? faceN : -faceN;
    }
};

#endif // __SHADING_DATA_HLSLI__