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

#include "../Surface/ShadingFlags.hlsli"

#include "../PathTracerHelpers.hlsli"

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
    float3      toPrevPosW;             ///< prevPosW - posW    <- can likely be packed to fp16 safely.

    float3      faceNCorrected;         ///< Face normal in world space, corrected for front-facing flag (which also means always facing the viewer). CANNOT be optimized to fp16.
    float3      V;                      ///< Direction to the eye at shading hit. CANNOT be optimized to fp16 (enough only for very low screen res).
    float3      N;                      ///< Shading normal at shading hit.
    bool        frontFacing;            ///< True if primitive seen from the front-facing side, as determined by dot product between view direction and triangle face normal (faceN - not corrected).

    // Material data
    ShadingFlags shadingFlags;
    uint        materialIDPosErrorScale;    ///< Material ID (MaterialPTData buffer index, low 16 bits as RTXPT_MATERIAL_MAX_COUNT is 32768) and posErrorScale as bfloat16 (high 16 bits, see ComputeRayOrigin)
    // posErrorScale rounds up to bfloat16 (fp32's top 16 bits): full fp32 range, never below the computed scale, at most 2^-7 above
    void        SetMaterialIDAndPosErrorScale(uint materialID, float posErrorScale) { materialIDPosErrorScale = (materialID & 0xFFFF) | ((asuint(posErrorScale) + 0xFFFF) & 0xFFFF0000); }
    uint        MaterialID()            { return materialIDPosErrorScale & 0xFFFF; }
    float       PosErrorScale()         { return asfloat(materialIDPosErrorScale & 0xFFFF0000); }
    
#if 0   // Unpacked
    lpfloat3    emission;               ///< Emissive radiance. Move to Surface and compress.
    void        SetEmission(float3 e)   { emission = e; }
    float3      GetEmission()           { return emission; }
#else   // Packed
    uint        emissionPacked;         ///< Emission packed as RGB10 log-encoded.
    void        SetEmission(float3 e)   { emissionPacked = PackEmissionRGB10Log(e); }
    float3      GetEmission()           { return UnpackEmissionRGB10Log(emissionPacked); }
#endif

#if 0   // Unpacked
    lpfloat     exteriorIoR;            ///< Index of refraction for the medium on the front-facing side (i.e. "outside" the material at the hit).
    lpfloat     interiorIoR;            ///< Index of refraction of the material interior (a.k.a. material IoR).
    void        SetIoR(float exterior, float interior)  { exteriorIoR = exterior; interiorIoR = interior; }
    void        SetExteriorIoR(float exterior)          { exteriorIoR = exterior; }
    float2      GetIoR()                                { return float2(exteriorIoR, interiorIoR); }
#else   // Packed
    uint        iorPacked;              ///< Packed fp16 exterior (x) and interior (y) IoR.
    void        SetIoR(float exterior, float interior)  { iorPacked = Fp32ToFp16(float2(exterior, interior)); }
    void        SetExteriorIoR(float exterior)          { iorPacked = Fp32ToFp16(float2(exterior, Fp16ToFp32(iorPacked).y)); }
    float2      GetIoR()                                { return Fp16ToFp32(iorPacked); }
#endif

#if 0   // Unpacked
    float4      Tw;             ///< Shading tangent at shading hit, with last value holding handedness for computing B
    void        SetTangentW(float4 tw)  { Tw = tw; }
    float4      GetTangentW()           { return Tw; }
#else   // Packed
    uint2       tanwPacked;             ///< Shading tangent at shading hit, with last value holding handedness for computing B
    void        SetTangentW(float4 tw)  { tanwPacked = Fp32ToFp16(tw); }
    float4      GetTangentW()           { return Fp16ToFp32(tanwPacked); }
#endif

#if 0   // Unpacked
    float3      _geometryNormal;
    float       _shadowTerminatorK;       ///<move to ShadingFlags if possible           < See corresponding material setting. Can be optimized to fp16 without any issues.
    void        SetGeometryNormalAndShadowTerminatorK(float3 geometryNormal, float shadowTerminatorK) { _geometryNormal = geometryNormal; _shadowTerminatorK = shadowTerminatorK; }
    float4      GetGeometryNormalAndShadowTerminatorK() { return float4( _geometryNormal, _shadowTerminatorK ); }
#else   // Packed
    uint2       gnstPacked;             
    void        SetGeometryNormalAndShadowTerminatorK(float3 geometryNormal, float shadowTerminatorK) { gnstPacked = Fp32ToFp16(float4(geometryNormal, shadowTerminatorK)); }
    float4      GetGeometryNormalAndShadowTerminatorK() { return Fp16ToFp32(gnstPacked); }
#endif

    // Tangent space
    float3      T()                     { return GetTangentW().xyz; }
    float3      B()                     { float4 tangentW = GetTangentW(); return cross(N, tangentW.xyz) * tangentW.w; }

    // Geometry normal - interpolated vertex normal, corrected for front-facing flag but not corrected for the ray view direction (can point away since interpolated). 
    float3      GeometryNormal()        { return GetGeometryNormalAndShadowTerminatorK().xyz; }
    float       ShadowTerminatorK()     { return GetGeometryNormalAndShadowTerminatorK().w; }

    float       GetEta()                { float2 ior = GetIoR(); return (!shadingFlags.isThinSurface() && !frontFacing) ? (ior.y / ior.x) : (ior.x / ior.y); }


    static ShadingData make()
    {
        ShadingData shadingData;
        shadingData.posW = 0;
        shadingData.toPrevPosW = 0;
        shadingData.V = 0;
        shadingData.N = 0;
        shadingData.SetTangentW(float4(0,0,0,0));
        shadingData.faceNCorrected = 0;
        shadingData.SetGeometryNormalAndShadowTerminatorK(float3(0,0,0),0);
        //shadingData.curveRadius = 0;

        shadingData.shadingFlags = ShadingFlags::make();
        shadingData.materialIDPosErrorScale = 0;
        shadingData.SetIoR(0, 0);
        shadingData.SetEmission(0);
        return shadingData;
    }

    // Utility functions

    // Computes new ray origin based on the hit point to avoid self-intersection, see ComputeRayOrigin.
    // \param[in] viewside True if the origin should be on the view side (reflection) or false otherwise (transmission).
    // \return Ray origin of the new ray.
    float3 ComputeNewRayOrigin(bool viewside = true)
    {
        return ComputeRayOrigin(posW, viewside ? faceNCorrected : -faceNCorrected, PosErrorScale());
    }

    // Transform vector from the local surface frame to world space.
    // \param[in] v Vector in local space.
    // \return Vector in world space.
    float3 FromLocal(float3 v)
    {
        return T() * v.x + B() * v.y + N * v.z;
        //return mul( v, float3x3(T,B,N) );
    }

    // Transform vector from world space to the local surface frame.
    // \param[in] v Vector in world space.
    // \return Vector in local space.
    float3 ToLocal(float3 v)
    {
        return float3(dot(v, T()), dot(v, B()), dot(v, N));
        //return mul( float3x3(T,B,N), v );
    }

    // Returns the oriented face normal.
    // \return Face normal flipped to the same side as the view vector.
    float3 GetOrientedFaceNormal()
    {
        return faceNCorrected; //frontFacing ? faceN : -faceN;
    }

    // Computes an orthonormal tangent space based on the normal and given tangent.
    // \param[in] tangent Interpolated tangent in world space (xyz) and bitangent sign (w). The tangent is *only* valid when w is != 0.
    // \return True if a valid tangent space was computed based on the supplied tangent.
    bool ComputeTangentSpace(const bool ignoreExistingTangent)
    {
        // Check that tangent space exists and can be safely orthonormalized.
        // Otherwise invent a tanget frame based on the normal.
        // We check that:
        //  - Tangent exists, this is indicated by a nonzero sign (w).
        //  - It has nonzero length. Zeros can occur due to interpolation or bad assets.
        //  - It is not parallel to the normal. This can occur due to normal mapping or bad assets.
        //  - It does not have NaNs. These will propagate and trigger the fallback.

        float4 tangentW = GetTangentW();

        float NdotT = dot(tangentW.xyz, N);
        bool nonParallel = abs(NdotT) < 0.9999f;
        bool nonZero = dot(tangentW.xyz, tangentW.xyz) > 0.f;

        bool valid = tangentW.w != 0.f && nonZero && nonParallel;
        if (!ignoreExistingTangent && valid)
        {
            tangentW.xyz = normalize(tangentW.xyz - N * NdotT);
        }
        else
        {
            tangentW.xyz = perp_stark(N);
            tangentW.w = 1;
        }
        SetTangentW(tangentW);

        return valid;
    }

    // Adjusts the normal of the supplied shading frame to reduce black pixels due to back-facing view direction.
    // Note: This breaks the reciprocity of the BSDF!
    // \param[in] tangentW - Shading tangent in world space (normalized). The last component is guaranteed to be +-1.0 or zero if tangents are missing.
    void AdjustShadingNormalAndComputeTangentSpace(uniform bool recomputeTangentSpace, const bool ignoreExistingTangent)
    {
#if 0
        // Note: V and Ng below always lie on the same side (as the front-facing flag is computed based on V).
        // The shading normal sf.N may lie on either side depending on whether we're shading the front or back.
        // We orient Ns to lie on the same side as V and Ng for the computations below.
        // The final adjusted normal is oriented to lie on the same side as the original shading normal.
        float3 Ng = GetOrientedFaceNormal();
        float signN = dot(N, Ng) >= 0.f ? 1.f : -1.f;
        float3 Ns = signN * N;

        // Blend the shading normal towards the geometric normal at grazing angles.
        // This is to avoid the view vector from becoming back-facing.
        const float kCosThetaThreshold = 0.1f;
        float cosTheta = dot(V, Ns);
        if (cosTheta <= kCosThetaThreshold)
        {
            float t = saturate(cosTheta * (1.f / kCosThetaThreshold));
            N = signN * normalize(lerp(Ng, Ns, t));
        }
        if (cosTheta <= kCosThetaThreshold || recomputeTangentSpace)
            ComputeTangentSpace(tangentW, ignoreExistingTangent);
#else
        // much simpler approach that pulls the offending normals back towards view vector - seems to leave more detail in practice
        const float k = 0.05; // start morphing just a tiny bit earlier
        float amountOfBackFacing = (max( -k, dot( -V, N ) + 1e-4f )+k) / (1+k);
        N = normalize( N + amountOfBackFacing * V );
        ComputeTangentSpace(ignoreExistingTangent);
#endif


    }
};

#endif // __SHADING_DATA_HLSLI__