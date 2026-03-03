/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "PathTracer/Config.h" // must always be included first

#include "PathTracer/PathTracerTypes.hlsli"

#include "Bindings/ShaderResourceBindings.hlsli"
#if PT_USE_RESTIR_GI
#include "Bindings/ReSTIRBindings.hlsli"
#endif

#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"

[shader("closesthit")] 
void CLOSESTHIT_ENTRY(inout PathPayload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib)
{
    PathState path = PathPayload::unpack( payload );
    PathTracer::HandleHit(path, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), attrib.barycentrics, GetWorkingContext());
    payload = PathPayload::pack( path );
}

[shader("anyhit")]
void ANYHIT_ENTRY(inout PathPayload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib/* : SV_IntersectionAttributes*/)
{
    PathState path = PathPayload::unpack( payload );
    if (!Bridge::AlphaTest(InstanceID(), InstanceIndex(), GeometryIndex(), PrimitiveIndex(), attrib.barycentrics/*, GetWorkingContext( ).debug*/ ))
        IgnoreHit();
    
    // AnyHit doesn't currently modify the payload - but it could!
    // payload = PathPayload::pack( path );
}
