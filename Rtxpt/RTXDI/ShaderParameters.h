/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef SHADER_PARAMETERS_H
#define SHADER_PARAMETERS_H

// Need to define this early so that RtxdiParameters.h defines the necessary structures
#define RTXDI_WITH_RESTIR_GI

#include <donut/shaders/view_cb.h>
#include <donut/shaders/sky_cb.h>


#include <rtxdi/ReSTIRDIParameters.h>
#include <rtxdi/ReGIRParameters.h>
#include <rtxdi/ReSTIRGIParameters.h>
#include "../PathTracer/PathTracerShared.h"
#include "../PathTracer/Lighting/PolymorphicLight.h"

#define TASK_PRIMITIVE_LIGHT_BIT 0x80000000u

#define RTXDI_PRESAMPLING_GROUP_SIZE 256
#define RTXDI_GRID_BUILD_GROUP_SIZE 256
#define RTXDI_SCREEN_SPACE_GROUP_SIZE 8
#define RTXDI_GRAD_FACTOR 3
#define RTXDI_GRAD_STORAGE_SCALE 256.0f
#define RTXDI_GRAD_MAX_VALUE 65504.0f

#define INSTANCE_MASK_OPAQUE 0x01
#define INSTANCE_MASK_ALPHA_TESTED 0x02
#define INSTANCE_MASK_TRANSPARENT 0x04
#define INSTANCE_MASK_ALL 0xFF

#define DENOISER_MODE_OFF 0
#define DENOISER_MODE_REBLUR 1
#define DENOISER_MODE_RELAX 2

#define VIS_MODE_NONE                0
#define VIS_MODE_COMPOSITED_COLOR    1
#define VIS_MODE_RESOLVED_COLOR      2
#define VIS_MODE_DIFFUSE             3
#define VIS_MODE_SPECULAR            4
#define VIS_MODE_DENOISED_DIFFUSE    5
#define VIS_MODE_DENOISED_SPECULAR   6
#define VIS_MODE_RESERVOIR_WEIGHT    7
#define VIS_MODE_RESERVOIR_M         8
#define VIS_MODE_DIFFUSE_GRADIENT    9
#define VIS_MODE_SPECULAR_GRADIENT   10
#define VIS_MODE_DIFFUSE_CONFIDENCE  11
#define VIS_MODE_SPECULAR_CONFIDENCE 12

#define BACKGROUND_DEPTH 65504.f

#define RAY_COUNT_TRACED(index) ((index) * 2)
#define RAY_COUNT_HITS(index) ((index) * 2 + 1)

#define REPORT_RAY(hit) if (g_PerPassConstants.rayCountBufferIndex >= 0) { \
    InterlockedAdd(u_RayCountBuffer[RAY_COUNT_TRACED(g_PerPassConstants.rayCountBufferIndex)], 1); \
    if (hit) InterlockedAdd(u_RayCountBuffer[RAY_COUNT_HITS(g_PerPassConstants.rayCountBufferIndex)], 1); }

struct BrdfRayTracingConstants
{
    PlanarViewConstants view;

    uint frameIndex;
};

struct PrepareLightsConstants
{
    uint numTasks;
    uint currentFrameLightOffset;
    uint previousFrameLightOffset;
    uint _padding;
    EnvMapSceneParams envMapSceneParams;
    EnvMapImportanceSamplingParams envMapImportanceSamplingParams;
};

struct PrepareLightsTask
{
    uint instanceAndGeometryIndex; // low 12 bits are geometryIndex, mid 19 bits are instanceIndex, high bit is TASK_PRIMITIVE_LIGHT_BIT
    uint triangleCount;
    uint lightBufferOffset;
    int previousLightBufferOffset; // -1 means no previous data
};

struct RenderEnvironmentMapConstants
{
    ProceduralSkyShaderParameters params;

    float2 invTextureSize;
};

struct PreprocessEnvironmentMapConstants
{
    uint2 sourceSize;
    uint sourceMipLevel;
    uint numDestMipLevels;
};

struct GBufferConstants
{
    PlanarViewConstants view;
    PlanarViewConstants viewPrev;

    float roughnessOverride;
    float metalnessOverride;
    float normalMapScale;
    uint enableAlphaTestedGeometry;

    int2 materialReadbackPosition;
    uint materialReadbackBufferIndex;
    uint enableTransparentGeometry;

    float textureLodBias;
    float textureGradientScale; // 2^textureLodBias
};

struct GlassConstants
{
    PlanarViewConstants view;

    uint enableEnvironmentMap;
    uint environmentMapTextureIndex;
    float environmentScale;
    float environmentRotation;

    int2 materialReadbackPosition;
    uint materialReadbackBufferIndex;
    float normalMapScale;
};

struct CompositingConstants
{
    PlanarViewConstants view;
    PlanarViewConstants viewPrev;

    uint enableTextures;
    uint denoiserMode;
    uint enableEnvironmentMap;
    uint environmentMapTextureIndex;

    float environmentScale;
    float environmentRotation;
    float noiseMix;
    float noiseClampLow;

    float noiseClampHigh;
    uint checkerboard;
    uint numRtxgiVolumes;
};

struct AccumulationConstants
{
    float2 outputSize;
    float2 inputSize;
    float2 inputTextureSizeInv;
    float2 pixelOffset;
    float blendFactor;
};

struct ProbeDebugConstants
{
    PlanarViewConstants view;
    uint blasDeviceAddressLow;
    uint blasDeviceAddressHigh;
    uint volumeIndex;
};

struct DDGIVolumeResourceIndices
{
    uint irradianceTextureSRV;
    uint distanceTextureSRV;
    uint probeDataTextureSRV;
    uint rayDataTextureUAV;
};

struct FilterGradientsConstants
{
    uint2 viewportSize;
    int passIndex;
    uint checkerboard;
};

struct ConfidenceConstants
{
    uint2 viewportSize;
    float2 invGradientTextureSize;

    float darknessBias;
    float sensitivity;
    uint checkerboard;
    int inputBufferIndex;

    float blendFactor;
};

struct VisualizationConstants
{
    RTXDI_RuntimeParameters runtimeParams;

    int2 outputSize;
    float2 resolutionScale;

    uint visualizationMode;
    uint inputBufferIndex;
    uint enableAccumulation;
};

struct ReGirIndirectConstants
{
    uint numIndirectSamples;
	uint _padding0;
	uint _padding1;
	uint _padding2;
};

struct RtxdiBridgeConstants 
{
    RTXDI_RuntimeParameters runtimeParams;

	// Common buffer parameters
	RTXDI_LightBufferParameters lightBufferParams;
	RTXDI_RISBufferSegmentParameters localLightsRISBufferSegmentParams;
	RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams;

	// Algorithm specific parameters
	ReSTIRDI_Parameters restirDI;
	ReGIR_Parameters regir;
	ReSTIRGI_Parameters restirGI;

    ReGirIndirectConstants regirIndirect;
	
    // Application specific parameters
    uint frameIndex;
	uint environmentMapImportanceSampling;
	uint maxLights;
	float rayEpsilon;

	uint2 _padding3;
	uint2 localLightPdfTextureSize;

	uint2 frameDim;
    uint environmentPdfLastMipLevel;
    uint localLightPdfLastMipLevel;

    uint reStirGIEnableTemporalResampling;
    uint reStirGIVaryAgeThreshold;
    uint _padding1;
    uint _padding2;
};

struct SecondarySurface
{
    float3 worldPos;
    uint normal;

    uint2 throughput;
    uint diffuseAlbedo;
    uint specularAndRoughness;
};

struct PackedPathTracerSurfaceData
{
    float3 _posW;
    uint _faceNCorrected;                   // fp16[3]
    uint2 _mtl;                             // Falcor::MaterialDefinition
    uint2 _V;                               // fp16[3]

    // misc (mostly subset of struct ShadingData)
    uint _T;                                // octFp16
    uint _N;                                // octFp16
    uint _viewDepth_planeHash_isEmpty_frontFacing;	// (fp16) | u15 | u1

    // StandardBSDFData (All fields nessesary)
    uint _diffuse;							// R11G11B10_FLOAT
    uint _specular;							// R11G11B10_FLOAT
    uint _roughnessMetallicEta;				// R11G11B10_FLOAT
    uint _transmission;						// R11G11B10_FLOAT
    uint _diffuseSpecularTransmission;		// fp16 | fp16
};

#endif // SHADER_PARAMETERS_H
