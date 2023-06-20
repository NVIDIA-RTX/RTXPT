/*
* Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef SURFACE_DATA_HLSLI
#define SURFACE_DATA_HLSLI

#include "../PathTracerBridgeDonut.hlsli"
#include "../PathTracer/PathTracer.hlsli"

#include "ShaderParameters.h"
#include "HelperFunctions.hlsli"

struct PathTracerSurfaceData
{
	// Only store these (plus more, potentially)
	MaterialHeader _mtl; // (2x uint)

	// misc (mostly subset of struct ShadingData)
	float3 _T;
	float3 _B;
	float3 _N;
	float3 _V;
	float3 _posW;
	float3 _faceN;
	bool _frontFacing;
	bool _isEmpty;
	float _viewDepth;
	uint _planeHash;

	// StandardBSDFData (All fields nessesary)
	// float3 diffuse;
	// float3 specular;
	// float roughness;
	// float metallic;
	// float eta;
	// float3 transmission;
	// float diffuseTransmission;
	// float specularTransmission;

	StandardBSDFData _data;      ///< BSDF parameters.

	float3 GetDiffuse()
	{
		return _data.diffuse;
	}

	float3 GetSpecular()
	{
		return _data.specular;
	}

	float GetRoughness()
	{
		return _data.roughness;
	}

	float3 GetNormal()
	{
		// TODO: this can come from FalcorBSDF
		return _N;
	}

	float3 GetView()
	{
		// TODO: this can come from FalcorBSDF
		return _V;
	}

	float3 GetPosW()
	{
		return _posW;
	}

	float3 GetFaceN()
	{
		return _faceN;
	}

	float	GetViewDepth()
	{
		return _viewDepth;
	}

	uint GetPlaneHash()
	{
		return _planeHash;
	}

    void RemoveTransmission()
    {
        _data.transmission = 0;
        _data.diffuseTransmission = 0;
        _data.specularTransmission = 0;
    }

#if PTSDK_DIFFUSE_SPECULAR_SPLIT
	void Eval(const float3 wo, inout SampleGenerator sg, out float3 diffuse, out float3 specular)
	{
		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = _ToLocal(wo);

		FalcorBSDF bsdf = FalcorBSDF::make(_mtl, _N, _V, _data);

		bsdf.eval(wiLocal, woLocal, sg, diffuse, specular);
	}

	void EvalRoughnessClamp(float minRoughness, const float3 wo, inout SampleGenerator sg, out float3 diffuse, out float3 specular)
	{
		StandardBSDFData roughBsdf = _data;
		roughBsdf.roughness = max(roughBsdf.roughness, minRoughness);

		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = _ToLocal(wo);

		FalcorBSDF bsdf = FalcorBSDF::make(_mtl, _N, _V, roughBsdf);

		bsdf.eval(wiLocal, woLocal, sg, diffuse, specular);
	}
#endif

	float3 Eval(const float3 wo, inout SampleGenerator sg)
	{
		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = _ToLocal(wo);

		FalcorBSDF bsdf = FalcorBSDF::make(_mtl, _N, _V, _data);

#if PTSDK_DIFFUSE_SPECULAR_SPLIT
		float3 diffuse, specular;
		bsdf.eval(wiLocal, woLocal, sg, diffuse, specular);
		return diffuse + specular;
#else
		return bsdf.eval(wiLocal, woLocal, sg);
#endif
	}

	float evalPdfReference(const float3 wo)
	{
		uint lobes = FalcorBSDF::getLobes(_data);

		const bool isTransmissive = (lobes & (uint)LobeType::Transmission) != 0;

		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = _ToLocal(wo);

		if (isTransmissive)
		{
			if (min(abs(wiLocal.z), abs(woLocal.z)) < kMinCosTheta) return 0.f;
			return 0.5f * woLocal.z * M_1_PI; // pdf = 0.5 * cos(theta) / pi
		}
		else
		{
			if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;
			return woLocal.z * M_1_PI; // pdf = cos(theta) / pi
		}
	}

	float EvalPdf(const float3 wo, bool useImportanceSampling)
	{
		if (!useImportanceSampling) return evalPdfReference(wo);

		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = _ToLocal(wo);

		FalcorBSDF bsdf = FalcorBSDF::make(_mtl, _N, _V, _data);

		return bsdf.evalPdf(wiLocal, woLocal);
	}

	bool sampleReference(inout SampleGenerator sg, out BSDFSample result)
	{
		uint lobes = FalcorBSDF::getLobes(_data);

		const bool isTransmissive = (lobes & (uint)LobeType::Transmission) != 0;

		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi

		if (isTransmissive)
		{
			if (sampleNext1D(sg) < 0.5f)
			{
				woLocal.z = -woLocal.z;
			}
			result.pdf *= 0.5f;
			if (min(abs(wiLocal.z), abs(woLocal.z)) < kMinCosTheta || result.pdf == 0.f) return false;
		}
		else
		{
			if (min(wiLocal.z, woLocal.z) < kMinCosTheta || result.pdf == 0.f) return false;
		}

		FalcorBSDF bsdf = FalcorBSDF::make(_mtl, _N, _V, _data);

		result.wo = _FromLocal(woLocal);
#if PTSDK_DIFFUSE_SPECULAR_SPLIT
		float3 diffuse, specular;
		bsdf.eval(wiLocal, woLocal, sg, diffuse, specular);
		result.weight = (diffuse + specular) / result.pdf;
#else
		result.weight = bsdf.eval(wiLocal, woLocal, sg) / result.pdf;
#endif
		result.lobe = (uint)(woLocal.z > 0.f ? (uint)LobeType::DiffuseReflection : (uint)LobeType::DiffuseTransmission);

		return true;
	}

	bool Sample(inout SampleGenerator sg, out BSDFSample result, bool useImportanceSampling)
	{
		if (!useImportanceSampling) return sampleReference(sg, result);

		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = float3(0, 0, 0);

		FalcorBSDF bsdf = FalcorBSDF::make(_mtl, _N, _V, _data);
		bool valid = bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobe, result.lobeP, sg);
		result.wo = _FromLocal(woLocal);

		return valid;
	}

	float3 ComputeNewRayOrigin(bool viewside = true)
	{
		return computeRayOrigin(_posW, (_frontFacing == viewside) ? _faceN : -_faceN);
	}

	float3 _ToLocal(float3 v)
	{
		return float3(dot(v, _T), dot(v, _B), dot(v, _N));
	}

	float3 _FromLocal(float3 v)
	{
		return _T * v.x + _B * v.y + _N * v.z;
	}

	static PathTracerSurfaceData create
	(
		 // FalcorBSDF
		const MaterialHeader mtl,
		float3 T,
		float3 B,
		float3 N,
		float3 V,
		float3 posW,
		float3 faceN,
		bool frontFacing,

		float viewDepth,
		const uint planeHash,

		const StandardBSDFData data
	)
	{
		PathTracerSurfaceData surface;

		surface._mtl = mtl;

		surface._T = T;
		surface._B = B;
		surface._N = N;
		surface._V = V;

		surface._data = data;

		surface._posW = posW;
		surface._faceN = faceN;
		surface._frontFacing = frontFacing;
		surface._viewDepth = viewDepth;
		surface._isEmpty = false;
		surface._planeHash = planeHash;
		//surface.dummy = 0;

		return surface;
	}

	static PathTracerSurfaceData makeEmpty()
	{
		PathTracerSurfaceData surface = (PathTracerSurfaceData)0;
		//surface._viewDepth = BACKGROUND_DEPTH;
		surface._isEmpty = true;
		return surface;
	}

	bool isEmpty()
	{
		return _isEmpty;
	    //return _viewDepth == BACKGROUND_DEPTH;
	}
};

// TODO: browse/merge/move these to Utils.hlsli

uint Fp32ToFp16(float2 v)
{
	const uint2 r = f32tof16(v);
	return (r.y << 16) | (r.x & 0xFFFF);
}

float2 Fp16ToFp32(uint r)
{
	uint2 v;
	v.x = (r & 0xFFFF);
	v.y = (r >> 16);
	return f16tof32(v);
}

uint2 Fp32ToFp16(float3 v)
{
	const uint d0 = Fp32ToFp16(v.xy);
	const uint d1 = f32tof16(v.z);
	return uint2(d0, d1);
}

float3 Fp16ToFp32(uint2 d)
{
	const float2 d0 = Fp16ToFp32(d.x);
	const float d1 = f16tof32(d.y);
	return float3(d0.xy, d1);
}

uint3 Fp32ToFp16(float3 a, float3 b)
{
	const uint d0 = Fp32ToFp16(a.xy);
	const uint d1 = Fp32ToFp16(float2(a.z, b.x));
	const uint d2 = Fp32ToFp16(b.yz);

	return uint3(d0, d1, d2);
}

void Fp16ToFp32(uint3 d, out float3 a, out float3 b)
{
	const float2 d0 = Fp16ToFp32(d.x);
	const float2 d1 = Fp16ToFp32(d.y);
	const float2 d2 = Fp16ToFp32(d.z);

	a = float3(d0, d1.x);
	b = float3(d1.y, d2);
}

float3 ReconstructOrthonormal(float3 a, float3 b) {
	return normalize(cross(a, b));
}

// Hash function from H. Schechter & R. Bridson, goo.gl/RXiKaH
uint Hash(uint s)
{
	s ^= 2747636419u;
	s *= 2654435769u;
	s ^= s >> 16;
	s *= 2654435769u;
	s ^= s >> 16;
	s *= 2654435769u;
	return s;
}

// TODO: check the miniengine variant - I think this does rounding down so over time loses energy
uint Encode_R11G11B10_FLOAT(float3 rgb)
{
	uint r = (f32tof16(rgb.x) << 17) & 0xFFE00000;
	uint g = (f32tof16(rgb.y) << 6) & 0x001FFC00;
	uint b = (f32tof16(rgb.z) >> 5) & 0x000003FF;
	return r | g | b;
}

float3 Decode_R11G11B10_FLOAT(uint rgb)
{
	float r = f16tof32((rgb >> 17) & 0x7FF0);
	float g = f16tof32((rgb >> 6) & 0x7FF0);
	float b = f16tof32((rgb << 5) & 0x7FE0);
	return float3(r, g, b);
}

PackedPathTracerSurfaceData RunCompress(PathTracerSurfaceData d)
{
	PackedPathTracerSurfaceData c;

	c._mtl = d._mtl.packedData;
	c._T = Fp32ToFp16(Encode_Oct(d._T));
	c._N = Fp32ToFp16(Encode_Oct(d._N));
	c._V = Fp32ToFp16(d._V);
	c._posW = d._posW;
	c._faceN = Fp32ToFp16(Encode_Oct(d._faceN));

	c._viewDepth_planeHash_isEmpty_frontFacing = (f32tof16(d._viewDepth) << 16u) | (Hash(d._planeHash) & 0xFFFC) | (d._isEmpty << 1u) | (d._frontFacing & 0x1);

	c._diffuse = Encode_R11G11B10_FLOAT(d._data.diffuse);
	c._specular = Encode_R11G11B10_FLOAT(d._data.specular);
	c._roughnessMetallicEta = Encode_R11G11B10_FLOAT(float3(d._data.roughness, d._data.metallic, d._data.eta));
	c._transmission = Encode_R11G11B10_FLOAT(d._data.transmission);
	c._diffuseSpecularTransmission = Fp32ToFp16(float2(d._data.diffuseTransmission, d._data.specularTransmission));

	return c;
}

PathTracerSurfaceData RunDecompress(PackedPathTracerSurfaceData c)
{
	PathTracerSurfaceData d;

	d._mtl.packedData = c._mtl;
	d._T = Decode_Oct(Fp16ToFp32(c._T));
	d._N = Decode_Oct(Fp16ToFp32(c._N));
	d._B = ReconstructOrthonormal(d._N, d._T);
	
	// Fp16ToFp32(c._V_posW, d._V, d._posW);
	
	d._V = Fp16ToFp32(c._V);
	d._posW = c._posW;

	d._faceN = Decode_Oct(Fp16ToFp32(c._faceN));

	d._viewDepth	= f16tof32(c._viewDepth_planeHash_isEmpty_frontFacing >> 16u);
	d._planeHash	= c._viewDepth_planeHash_isEmpty_frontFacing & 0xFFFC;  // perhaps we can just keep the front facing bit as part of the plane hash? 
	d._isEmpty		= (c._viewDepth_planeHash_isEmpty_frontFacing >> 1u) & 0x1;
	d._frontFacing	= c._viewDepth_planeHash_isEmpty_frontFacing & 0x1;

	d._data.diffuse = Decode_R11G11B10_FLOAT(c._diffuse);
	d._data.specular = Decode_R11G11B10_FLOAT(c._specular);
	
	const float3 roughnessMetallicEta = Decode_R11G11B10_FLOAT(c._roughnessMetallicEta);
	
	d._data.roughness = roughnessMetallicEta.x;
	d._data.metallic = roughnessMetallicEta.y;
	d._data.eta = roughnessMetallicEta.z;
	d._data.transmission = Decode_R11G11B10_FLOAT(c._transmission);

	const float2 diffuseSpeculartransmission = Fp16ToFp32(c._diffuseSpecularTransmission);

	d._data.diffuseTransmission = diffuseSpeculartransmission.x;
	d._data.specularTransmission = diffuseSpeculartransmission.y;

	return d;
}

PathTracerSurfaceData RunCompressDecompress(PathTracerSurfaceData input)
{
	PackedPathTracerSurfaceData c = RunCompress(input);
	PathTracerSurfaceData d = RunDecompress(c);
	return d;
}

bool isValidPixelPosition(int2 pixelPosition)
{
    return all(pixelPosition >= 0) && pixelPosition.x < g_Const.ptConsts.imageWidth && pixelPosition.y < g_Const.ptConsts.imageHeight;
}

// Load a surface from the current or previous vbuffer at the specified pixel postion 
// Pixel positions may be out of bounds or negative, in which case the function is supposed to 
// return an invalid surface
PathTracerSurfaceData getGBufferSurfaceImpl(uint2 pixelPosition, StablePlane sp, uint dominantStablePlaneIndex, uint stableBranchID)
{
	DebugContext debug;
	debug.Init(pixelPosition, Bridge::getSampleIndex(), g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack, u_DebugVizOutput);

    PackedHitInfo packedHitInfo; float3 rayDir; uint vertexIndex; uint SERSortKey; float sceneLength; float3 pathThp; float3 motionVectors;
    StablePlanesContext::UnpackStablePlane(sp, vertexIndex, packedHitInfo, SERSortKey, rayDir, sceneLength, pathThp, motionVectors);
	
    uint planeHash = (dominantStablePlaneIndex << 16) | vertexIndex;    // consider replacing with stableBranchID - that will cover vertexIndex and planes but also separate delta paths within a plane

    const HitInfo hit = HitInfo(packedHitInfo);
    if ((hit.isValid() && hit.getType() == HitType::Triangle))
    {
        // Load shading surface
        const Ray cameraRay = Bridge::computeCameraRay( pixelPosition );
        RayCone rayCone = RayCone::make(0, g_Const.ptConsts.camera.pixelConeSpreadAngle).propagateDistance(sceneLength);
        const SurfaceData bridgedData = Bridge::loadSurface(OptimizationHints::NoHints(), TriangleHit::make(packedHitInfo), rayDir, rayCone, Bridge::getPathTracerParams(), vertexIndex, debug);
        const ShadingData sd    = bridgedData.sd;
        const ActiveBSDF bsdf   = bridgedData.bsdf;
        BSDFProperties bsdfProperties = bsdf.getProperties(sd);

#if 1
        float viewDepth = g_Const.ptConsts.camera.nearZ+dot( cameraRay.dir * sceneLength, normalize(g_Const.ptConsts.camera.directionW) );
#else // same as above - useful for testing - for viz use `debug.DrawDebugViz( pixelPosition, float4( frac(viewDepth).xxx, 1) );`
        float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sceneLength;
        float viewDepth = mul(float4(virtualWorldPos, 1), g_Const.view.matWorldToView).z;
#endif

		uint lobes = bsdf.getLobes(sd);
		if ((lobes & (uint)LobeType::NonDeltaReflection) != 0)
		{
			PathTracerSurfaceData surface = PathTracerSurfaceData::create(
				sd.mtl,
				sd.T,
				sd.B,
				sd.N,
				sd.V,
				sd.posW,
				sd.faceN,
				sd.frontFacing,
				
				viewDepth,
				planeHash,
			
				bsdf.data
			);


			return surface;
		}
    }

	return PathTracerSurfaceData::makeEmpty();
}

// Load a surface from the current vbuffer at the specified pixel position.
// Pixel positions may be out of bounds or negative, in which case the function is supposed to return an invalid surface.
PathTracerSurfaceData getGBufferSurfaceImpl(int2 pixelPosition)
{
	//Return invalid surface data if pixel is out of bounds
	if (!isValidPixelPosition(pixelPosition))
	    return PathTracerSurfaceData::makeEmpty();

    // Init globals
    StablePlanesContext stablePlanes = StablePlanesContext::make(pixelPosition, u_StablePlanesHeader, u_StablePlanesBuffer, u_StableRadiance, u_SecondarySurfaceRadiance, g_Const.ptConsts);

    // Figure out the shading plane
    uint dominantStablePlaneIndex = stablePlanes.LoadDominantIndexCenter();
    uint stableBranchID = stablePlanes.GetBranchID(pixelPosition, dominantStablePlaneIndex);
    return getGBufferSurfaceImpl(pixelPosition, stablePlanes.LoadStablePlane(pixelPosition, dominantStablePlaneIndex), dominantStablePlaneIndex, stableBranchID);
}

// Load a surface from the current or previous GBuffer at the specified pixel position.
// Pixel positions may be out of bounds or negative, in which case the function is supposed to return an invalid surface.
PathTracerSurfaceData getGBufferSurface(int2 pixelPosition, bool previousFrame)
{
	if (!isValidPixelPosition(pixelPosition))
		return PathTracerSurfaceData::makeEmpty();

    // the current/history ping pongs each frame - compute the required offset!
	// see ExportVisibilityBuffer.hlsl for idxPingPong computation
    const uint idxPingPong = (g_Const.ptConsts.frameIndex % 2) == (uint)previousFrame;
    const uint idx = GenericTSPixelToAddress(pixelPosition, idxPingPong, g_Const.ptConsts.genericTSLineStride, g_Const.ptConsts.genericTSPlaneStride);

	PackedPathTracerSurfaceData packed = u_SurfaceData[idx];
	return RunDecompress(packed);

#if 0 // for testing the above, but only if previousFrame == false, use this
		PathTracerSurfaceData surface = getGBufferSurfaceImpl(pixelPosition);
		const bool debugCompression = true; // for testing compression/decompression, enable this
		if (debugCompression)
			surface = RunCompressDecompress(surface);
		return surface;
#endif
}

PackedPathTracerSurfaceData ExtractPackedGbufferSurfaceData(uint2 pixelPosition, StablePlane sp, uint dominantStablePlaneIndex, uint stableBranchID)
{
	const PathTracerSurfaceData data = getGBufferSurfaceImpl(pixelPosition, sp, dominantStablePlaneIndex, stableBranchID);
	return RunCompress(data);
}

#endif //SURFACE_DATA_HLSLI
