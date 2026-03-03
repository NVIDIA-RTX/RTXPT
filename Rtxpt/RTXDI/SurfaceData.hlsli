/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef SURFACE_DATA_HLSLI
#define SURFACE_DATA_HLSLI

#include "../Shaders/Bindings/ShaderResourceBindings.hlsli"

#include "../Shaders/Misc/GBufferHelpers.hlsli"

#include "../Shaders/PathTracerBridgeDonut.hlsli"

#include "ShaderParameters.h"
#include "HelperFunctions.hlsli"

PathTracerCollectedSurfaceData RunCompressDecompress(PathTracerCollectedSurfaceData input)
{
	PackedPathTracerSurfaceData c = RunCompress(input);
	PathTracerCollectedSurfaceData d = RunDecompress(c);
	return d;
}

bool isValidPixelPosition(int2 pixelPosition)
{
    return all(pixelPosition >= 0) && pixelPosition.x < g_Const.ptConsts.imageWidth && pixelPosition.y < g_Const.ptConsts.imageHeight;
}

// // Load a surface from the current vbuffer at the specified pixel position.
// // Pixel positions may be out of bounds or negative, in which case the function is supposed to return an invalid surface.
// PathTracerCollectedSurfaceData getGBufferSurfaceImpl(int2 pixelPosition)
// {
// 	//Return invalid surface data if pixel is out of bounds
// 	if (!isValidPixelPosition(pixelPosition))
// 	    return PathTracerCollectedSurfaceData::makeEmpty();
// 
//     // Init globals
//     StablePlanesContext stablePlanes = StablePlanesContext::make(u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, g_Const.ptConsts);
// 
//     // Figure out the shading plane
//     uint dominantStablePlaneIndex = stablePlanes.LoadDominantIndex(pixelPosition);
//     uint stableBranchID = stablePlanes.GetBranchID(pixelPosition, dominantStablePlaneIndex);
//     return getGBufferSurfaceImpl(pixelPosition, stablePlanes.LoadStablePlane(pixelPosition, dominantStablePlaneIndex), dominantStablePlaneIndex, stableBranchID);
// }

// Load a surface from the current or previous GBuffer at the specified pixel position.
// Pixel positions may be out of bounds or negative, in which case the function is supposed to return an invalid surface.
PathTracerCollectedSurfaceData getGBufferSurface(int2 pixelPosition, bool previousFrame)
{
	if (!isValidPixelPosition(pixelPosition))
		return PathTracerCollectedSurfaceData::makeEmpty();

    // the current/history ping pongs each frame - compute the required offset!
	// see ExportVisibilityBuffer.hlsl for idxPingPong computation
    const uint idxPingPong = (g_Const.ptConsts.frameIndex % 2) == (uint)previousFrame;
    const uint idx = GenericTSPixelToAddress(pixelPosition, idxPingPong, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);

	PackedPathTracerSurfaceData packed = u_SurfaceData[idx];
	return RunDecompress(packed);

#if 0 // for testing the above, but only if previousFrame == false, use this
		PathTracerCollectedSurfaceData surface = getGBufferSurfaceImpl(pixelPosition);
		const bool debugCompression = true; // for testing compression/decompression, enable this
		if (debugCompression)
			surface = RunCompressDecompress(surface);
		return surface;
#endif
}

// PackedPathTracerSurfaceData ExtractPackedGbufferSurfaceData(uint2 pixelPosition, StablePlane sp, uint dominantStablePlaneIndex, uint stableBranchID)
// {
// 	const PathTracerCollectedSurfaceData data = getGBufferSurfaceImpl(pixelPosition, sp, dominantStablePlaneIndex, stableBranchID);
// 	return RunCompress(data);
// }

#endif //SURFACE_DATA_HLSLI
