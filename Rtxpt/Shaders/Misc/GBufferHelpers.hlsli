/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __GBUFFER_HELPERS_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __GBUFFER_HELPERS_HLSLI__

#include "../PathTracer/PathTracerTypes.hlsli"

struct PathTracerCollectedSurfaceData
{
	// Only store these (plus more, potentially)
	MaterialHeader _mtl; // (2x uint)

	// misc (mostly subset of struct ShadingData)
	float3 _T;
	float3 _B;
	float3 _N;
	float3 _V;
	float3 _posW;
	float3 _faceNCorrected;
	bool _frontFacing;
	bool _isEmpty;
	float _viewDepth;
	uint _planeHash;

	StandardBSDFData _data;      ///< BSDF parameters.

	float3 GetDiffuse()
	{
		return _data.Diffuse();
	}

	float3 GetSpecular()
	{
		return _data.Specular();
	}

	float GetRoughness()
	{
		return _data.Roughness();
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

	float3 GetFaceNCorrected()
	{
		return _faceNCorrected;
	}

	float	GetViewDepth()
	{
		return _viewDepth;
	}

	uint GetPlaneHash()
	{
		return _planeHash;
	}

	float4 Eval(const float3 wo)
	{
		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = _ToLocal(wo);

		FalcorBSDF bsdf = FalcorBSDF::make(_mtl, _N, _V, _data);

		return bsdf.eval(wiLocal, woLocal);
	}

	float4 EvalRoughnessClamp(lpfloat minRoughness, const float3 wo)
	{
		StandardBSDFData roughBsdf = _data;
		roughBsdf.SetRoughness( max(roughBsdf.Roughness(), minRoughness) );

		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = _ToLocal(wo);

		FalcorBSDF bsdf = FalcorBSDF::make(_mtl, _N, _V, roughBsdf);

		return bsdf.eval(wiLocal, woLocal);
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
			return 0.5f * woLocal.z * K_1_PI; // pdf = 0.5 * cos(theta) / pi
		}
		else
		{
			if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return 0.f;
			return woLocal.z * K_1_PI; // pdf = cos(theta) / pi
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

	bool sampleReference(inout SampleGenerator sampleGenerator, out BSDFSample result)
	{
		uint lobes = FalcorBSDF::getLobes(_data);

		const bool isTransmissive = (lobes & (uint)LobeType::Transmission) != 0;

		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sampleGenerator), result.pdf); // pdf = cos(theta) / pi

		if (isTransmissive)
		{
			if (sampleNext1D(sampleGenerator) < 0.5f)
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
        result.weight = bsdf.eval(wiLocal, woLocal).rgb / result.pdf;
		result.lobe = (uint)(woLocal.z > 0.f ? (uint)LobeType::DiffuseReflection : (uint)LobeType::DiffuseTransmission);

		return true;
	}
    
	bool Sample(inout SampleGenerator sampleGenerator, out BSDFSample result, bool useImportanceSampling)
	{
		if (!useImportanceSampling) return sampleReference(sampleGenerator, result);

		float3 wiLocal = _ToLocal(_V);
		float3 woLocal = float3(0, 0, 0);

		FalcorBSDF bsdf = FalcorBSDF::make(_mtl, _N, _V, _data);
#if RecycleSelectSamples
        bool valid = bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobe, result.lobeP, sampleNext3D(sampleGenerator));
#else
		bool valid = bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobe, result.lobeP, sampleNext4D(sampleGenerator));
#endif
		result.wo = _FromLocal(woLocal);

		return valid;
	}

	float3 ComputeNewRayOrigin(bool viewside = true)
	{
		return ComputeRayOrigin(_posW, (viewside) ? _faceNCorrected : -_faceNCorrected);
	}

	float3 _ToLocal(float3 v)
	{
		return float3(dot(v, _T), dot(v, _B), dot(v, _N));
	}

	float3 _FromLocal(float3 v)
	{
		return _T * v.x + _B * v.y + _N * v.z;
	}

	static PathTracerCollectedSurfaceData create
	(
		 // FalcorBSDF
		const MaterialHeader mtl,
		float3 T,
		float3 B,
		float3 N,
		float3 V,
		float3 posW,
		float3 faceNCorrected,
		bool frontFacing,

		float viewDepth,
		const uint planeHash,

		const StandardBSDFData data
	)
	{
		PathTracerCollectedSurfaceData surface;

		surface._mtl = mtl;

		surface._T = T;
		surface._B = B;
		surface._N = N;
		surface._V = V;

		surface._data = data;

		surface._posW = posW;
		surface._faceNCorrected = faceNCorrected;
		surface._frontFacing = frontFacing;
		surface._viewDepth = viewDepth;
		surface._isEmpty = false;
		surface._planeHash = planeHash;
		//surface.dummy = 0;

		return surface;
	}

	static PathTracerCollectedSurfaceData makeEmpty()
	{
		PathTracerCollectedSurfaceData surface = (PathTracerCollectedSurfaceData)0;
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

PackedPathTracerSurfaceData RunCompress(PathTracerCollectedSurfaceData d)
{
	PackedPathTracerSurfaceData c;

	c._mtl = uint2(d._mtl.packedData, 0);
	c._T = Fp32ToFp16(Encode_Oct(d._T));
	c._N = Fp32ToFp16(Encode_Oct(d._N));
    float btNormal = -sign(dot( cross(d._T, d._N), d._B ));
	c._V = Fp32ToFp16(float4(d._V, btNormal));
	c._posW = d._posW;
	c._faceNCorrected = Fp32ToFp16(Encode_Oct(d._faceNCorrected));

	c._viewDepth_planeHash_isEmpty_frontFacing = (f32tof16(d._viewDepth) << 16u) | (Hash(d._planeHash) & 0xFFFC) | (d._isEmpty << 1u) | (d._frontFacing & 0x1);

	c._diffuse = Encode_R11G11B10_FLOAT(d._data.Diffuse());
	c._specular = Encode_R11G11B10_FLOAT(d._data.Specular());
	c._roughnessMetallicEta = Encode_R11G11B10_FLOAT(float3(d._data.Roughness(), d._data.Metallic(), d._data.Eta()));
	c._transmission = Encode_R11G11B10_FLOAT(d._data.Transmission());
	c._diffuseSpecularTransmission = Fp32ToFp16(float2(d._data.DiffuseTransmission(), d._data.SpecularTransmission()));

	return c;
}

PathTracerCollectedSurfaceData RunDecompress(PackedPathTracerSurfaceData c)
{
	PathTracerCollectedSurfaceData d;

	d._mtl.packedData = c._mtl.x;
    float4 VandTW = Fp16ToFp32(c._V);
	d._T = Decode_Oct(Fp16ToFp32(c._T));
	d._N = Decode_Oct(Fp16ToFp32(c._N));
	d._B = ReconstructOrthonormal(d._N, d._T*VandTW.w);  // I'm not 100% sure this is correct - it looks like we need to store the winding bit as well somehere. But it looks ok in all tests
	
	// Fp16ToFp32(c._V_posW, d._V, d._posW);
	
	d._V = VandTW.xyz;
	d._posW = c._posW;

	d._faceNCorrected = Decode_Oct(Fp16ToFp32(c._faceNCorrected));

	d._viewDepth	= f16tof32(c._viewDepth_planeHash_isEmpty_frontFacing >> 16u);
	d._planeHash	= c._viewDepth_planeHash_isEmpty_frontFacing & 0xFFFC;  // perhaps we can just keep the front facing bit as part of the plane hash? 
	d._isEmpty		= (c._viewDepth_planeHash_isEmpty_frontFacing >> 1u) & 0x1;
	d._frontFacing	= c._viewDepth_planeHash_isEmpty_frontFacing & 0x1;

    lpfloat3    bsdfDataDiffuse              = 0;
    lpfloat     bsdfDataRoughness            = 0;
    lpfloat3    bsdfDataSpecular             = 0;
    lpfloat     bsdfDataMetallic             = 0;
    lpfloat3    bsdfDataTransmission         = 0;
    lpfloat     bsdfDataEta                  = 0;
    lpfloat     bsdfDataDiffuseTransmission  = 0;
    lpfloat     bsdfDataSpecularTransmission = 0;

    // I'm unsure if ReSTIR GI needs transmission or not; I can't see any difference between below on/off for ReSTIR GI but this needs a follow-up
#if RAB_SURFACE_REMOVE_TRANSMISSION
#else
	bsdfDataTransmission = lpfloat3(Decode_R11G11B10_FLOAT(c._transmission));
	const lpfloat2 diffuseSpeculartransmission = lpfloat2(Fp16ToFp32(c._diffuseSpecularTransmission));
	bsdfDataDiffuseTransmission = diffuseSpeculartransmission.x;
	bsdfDataSpecularTransmission = diffuseSpeculartransmission.y;
#endif

	bsdfDataDiffuse = lpfloat3(Decode_R11G11B10_FLOAT(c._diffuse));
	bsdfDataSpecular = lpfloat3(Decode_R11G11B10_FLOAT(c._specular));
	
	const lpfloat3 roughnessMetallicEta = lpfloat3(Decode_R11G11B10_FLOAT(c._roughnessMetallicEta));
	
	bsdfDataRoughness = roughnessMetallicEta.x;
	bsdfDataMetallic = roughnessMetallicEta.y;
	bsdfDataEta = roughnessMetallicEta.z;

    d._data = StandardBSDFData::make( bsdfDataDiffuse, bsdfDataSpecular, bsdfDataRoughness, bsdfDataMetallic, bsdfDataEta, bsdfDataTransmission, bsdfDataDiffuseTransmission, bsdfDataSpecularTransmission );

	return d;
}

PathTracerCollectedSurfaceData CollectGBufferSurface(PathState path, PathTracer::SurfaceData surfaceData, const Ray cameraRay)
{
	DebugContext debug;
	debug.Init(g_Const.debug, u_FeedbackBuffer, u_DebugLinesBuffer, u_DebugDeltaPathTree, u_DeltaPathSearchStack);

    // float3 rayOrigin = path.origin;
	// float3 rayDir = path.dir;
	uint vertexIndex = path.getVertexIndex();
	float sceneLength = path.GetSceneLength();
	
#if 1
        float viewDepth = g_Const.ptConsts.camera.NearZ+dot( cameraRay.dir * sceneLength, normalize(g_Const.ptConsts.camera.DirectionW) );
#else // same as above - useful for testing - for viz use `debug.DrawDebugViz( pixelPosition, float4( frac(viewDepth).xxx, 1) );`
        float3 virtualWorldPos = cameraRay.origin + cameraRay.dir * sceneLength;
        float viewDepth = mul(float4(virtualWorldPos, 1), g_Const.view.matWorldToView).z;
#endif

    const ShadingData shadingData   = surfaceData.shadingData;
    const ActiveBSDF bsdf           = surfaceData.bsdf;

	uint lobes = bsdf.getLobes(shadingData);
	if ((lobes & (uint)LobeType::NonDeltaReflection) != 0)
	{
		PathTracerCollectedSurfaceData surface = PathTracerCollectedSurfaceData::create(
			shadingData.mtl,
			shadingData.T,
			shadingData.B,
			shadingData.N,
			shadingData.V,
			shadingData.posW,
			shadingData.faceNCorrected,
			shadingData.frontFacing,
				
			viewDepth,
			0,
			
			bsdf.data
		);


		return surface;
	}
    
	return PathTracerCollectedSurfaceData::makeEmpty();
}


#endif // __GBUFFER_HELPERS_HLSLI__
