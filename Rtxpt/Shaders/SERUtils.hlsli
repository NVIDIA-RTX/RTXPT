/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SER_UTILS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __SER_UTILS_HLSLI__

#include "PathTracer/Config.h" // must always be included first

// this sets up various approaches - different combinations are possible
// this will get significantly simplified once SER API is part of DirectX - we'll then be able to default to "TraceRayInline" version in all variants
#if USE_NVAPI_HIT_OBJECT_EXTENSION
#define NV_HITOBJECT_USE_MACRO_API 1
#include <NVAPI/nvHLSLExtns.h>

#define SER_HIT_OBJECT  NvHitObject
#define SER_HIT_OBJECT_INIT_FROM_MISS( _outHit, _rayQuery ) \
        RayDesc __rayDesc; __rayDesc.Origin = _rayQuery.WorldRayOrigin(); __rayDesc.TMin = _rayQuery.RayTMin(); __rayDesc.Direction = _rayQuery.WorldRayDirection(); __rayDesc.TMax = kMaxRayTravel; \
        _outHit = NvMakeMiss( 0, __rayDesc );
#define SER_HIT_OBJECT_INIT_FROM_RAYQ( _outHit, _rayQuery ) \
        BuiltInTriangleIntersectionAttributes __attrib; \
        __attrib.barycentrics = _rayQuery.CommittedTriangleBarycentrics(); \
        uint __shaderTableIndex = _rayQuery.CommittedInstanceContributionToHitGroupIndex()+_rayQuery.CommittedGeometryIndex(); \
        RayDesc __rayDesc; __rayDesc.Origin = _rayQuery.WorldRayOrigin(); __rayDesc.TMin = _rayQuery.RayTMin(); __rayDesc.Direction = _rayQuery.WorldRayDirection(); __rayDesc.TMax = _rayQuery.CommittedRayT(); \
        NvMakeHitWithRecordIndex( __shaderTableIndex, SceneBVH, _rayQuery.CommittedInstanceIndex(), _rayQuery.CommittedGeometryIndex(), _rayQuery.CommittedPrimitiveIndex(), 0, __rayDesc, __attrib, _outHit );
#define SER_REORDER_HIT( _hit, _sortBits, _numberOfBits ) \
        NvReorderThread( _hit, _sortBits, _numberOfBits );
#define SER_INVOKE_HIT( _hit, _payload ) \
        NvInvokeHitObject(SceneBVH, _hit, _payload);
#define SER_SORT_ENABLED USE_NVAPI_REORDER_THREADS

#elif USE_DX_HIT_OBJECT_EXTENSION

#define SER_HIT_OBJECT  dx::HitObject
#define SER_HIT_OBJECT_INIT_FROM_MISS( _outHit, _rayQuery ) \
        RayDesc __rayDesc; __rayDesc.Origin = _rayQuery.WorldRayOrigin(); __rayDesc.TMin = _rayQuery.RayTMin(); __rayDesc.Direction = _rayQuery.WorldRayDirection(); __rayDesc.TMax = kMaxRayTravel; \
        _outHit = dx::HitObject::MakeMiss(RAY_FLAG_NONE, 0, __rayDesc);
#define SER_HIT_OBJECT_INIT_FROM_RAYQ( _outHit, _rayQuery ) \
        BuiltInTriangleIntersectionAttributes __attrib; \
        __attrib.barycentrics = _rayQuery.CommittedTriangleBarycentrics(); \
        _outHit = dx::HitObject::FromRayQuery(_rayQuery); \
        _outHit.SetShaderTableIndex(_rayQuery.CommittedInstanceContributionToHitGroupIndex()+_rayQuery.CommittedGeometryIndex());
#define SER_REORDER_HIT( _hit, _sortBits, _numberOfBits ) \
        dx::MaybeReorderThread(_hit, _sortBits, _numberOfBits );
#define SER_INVOKE_HIT( _hit, _payload ) \
        dx::HitObject::Invoke( _hit, _payload );
#define SER_SORT_ENABLED USE_DX_MAYBE_REORDER_THREADS

#endif

#endif // __SER_UTILS_HLSLI__