/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Config.h"
#include "Lighting/LightingTypes.h"

#ifndef __PATH_TRACER_SHARED_H__
#define __PATH_TRACER_SHARED_H__

#ifndef __cplusplus
#pragma pack_matrix(row_major)
#endif

#define PATH_TRACER_MAX_PAYLOAD_SIZE     4*4*5    // PathPayload is 80 at the moment

// Condensed version of ..\Falcor\Source\Falcor\Scene\Camera\CameraData.hlsli
struct PathTracerCameraData
{
    float3  PosW;
    float   NearZ;                      ///< Camera near plane.
    float3  DirectionW;                 ///< Camera world direction (same as cameraW, except normalized)
    float   PixelConeSpreadAngle;       ///< For raycones
    float3  CameraU;                    ///< Camera base vector U. Normalized it indicates the right image plane vector. The length is dependent on the FOV.
    float   FarZ;                       ///< Camera far plane.
    float3  CameraV;                    ///< Camera base vector V. Normalized it indicates the up image plane vector. The length is dependent on the FOV.
    float   FocalDistance;              ///< Camera focal distance in scene units.
    float3  CameraW;                    ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float   AspectRatio;                ///< viewport.w / viewport.h
    uint2   ViewportSize;               ///< Viewport size
    float   ApertureRadius;             ///< Camera aperture radius in scene units.
    float   _padding0;
    float2  Jitter;
    float   _padding1;
    float   _padding2;
};

// path tracer main constants
struct PathTracerConstants
{
    uint    imageWidth;
    uint    imageHeight;
    uint    sampleBaseIndex;                // sampleIndex != frameIndex since we can have multiple samples per frame
    float   perPixelJitterAAScale;          // this is for reference capture, and it's also used by DLSS-RR in a small amount

    uint    bounceCount;
    uint    diffuseBounceCount;
    float   EnvironmentMapDiffuseSampleMIPLevel;
    float   texLODBias;

    float   invSubSampleCount;              // used to attenuate noisy radiance during multi-sampling (non-noisy stuff like direct sky does not need attenuation!); always 1 for reference mode
    float   fireflyFilterThreshold;         //< if 0, firefly filter disabled
    float   preExposedGrayLuminance;
    uint    denoisingEnabled;

    uint    frameIndex;                     // sampleIndex != frameIndex since we can have multiple samples per frame
    uint    useReSTIRDI;
    uint    useReSTIRGI;
    uint    _padding5;

    float   stablePlanesSplitStopThreshold;
    float   _padding3;
    uint    _padding4;
    float   stablePlanesSuppressPrimaryIndirectSpecularK;

    float   denoiserRadianceClampK;
    uint    _padding0;
    float   stablePlanesAntiAliasingFallthrough;
    uint    activeStablePlaneCount;

    uint    maxStablePlaneVertexDepth;
    uint    allowPrimarySurfaceReplacement;
    uint    genericTSLineStride;  // used for u_SurfaceData - might be a candidate for macro optimization
    uint    genericTSPlaneStride; // used for u_SurfaceData - might be a candidate for macro optimization

    uint    NEEEnabled;
    uint    NEEType;
    uint    NEECandidateSamples;
    uint    NEEFullSamples;
  
    uint    _padding6;
    uint    STFMagnificationMethod;
    uint    STFFilterMode;
    float   STFGaussianSigma;

    PathTracerCameraData camera;
    PathTracerCameraData prevCamera;
};

#ifdef __cplusplus
inline PathTracerCameraData BridgeCamera( uint viewportWidth, uint viewportHeight, float aspectRatio, float3 camPos, float3 camDir, float3 camUp, float fovY, float nearZ, float farZ, 
    float focalDistance, float apertureRadius, float2 jitter )
{
    PathTracerCameraData data;

    data.FocalDistance  = focalDistance;
    data.PosW           = camPos;
    data.NearZ          = nearZ;
    data.FarZ           = farZ;
    data.AspectRatio    = aspectRatio;
    data.ViewportSize   = {viewportWidth, viewportHeight};

    // Ray tracing related vectors
    data.DirectionW = donut::math::normalize( camDir );
    data.CameraW = donut::math::normalize( camDir ) * data.FocalDistance;
    data.CameraU = donut::math::normalize( donut::math::cross( data.CameraW, camUp ) );
    data.CameraV = donut::math::normalize( donut::math::cross( data.CameraU, data.CameraW ) );
    const float ulen = data.FocalDistance * std::tan( fovY * 0.5f ) * data.AspectRatio;
    data.CameraU *= ulen;
    const float vlen = data.FocalDistance * std::tan( fovY * 0.5f );
    data.CameraV *= vlen;
    data.ApertureRadius = apertureRadius;

    // Note: spread angle is the whole (not half) cone angle!
    data.PixelConeSpreadAngle = std::atan(2.0f * std::tan(fovY * 0.5f) / viewportHeight);

    data.Jitter = jitter * float2(1, -1);
    data._padding0 = 0;
    data._padding1 = 0;
    data._padding2 = 0;

    return data;
}
#endif // __cplusplus

inline float3  DbgShowNormalSRGB(float3 normal)
{
    return pow(abs(normal * 0.5f + 0.5f), 2.2f);
}

#endif // __PATH_TRACER_SHARED_H__
