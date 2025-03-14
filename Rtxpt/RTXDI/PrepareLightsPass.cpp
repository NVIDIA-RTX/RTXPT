/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "PrepareLightsPass.h"
#include "RtxdiResources.h"
#include "../ExtendedScene.h"
#include <donut/engine/Scene.h>

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/log.h>
#include <nvrhi/utils.h>

#include <algorithm>
#include <utility>

using namespace donut::math;
#include "ShaderParameters.h"
#include "../Lighting/Distant/EnvMapBaker.h"
#include "../Lighting/Distant/EnvMapImportanceSamplingBaker.h"
#include "../RenderTargets.h"

#include "../ShaderDebug.h"

#include "../Materials/MaterialsBaker.h"
#include "../OpacityMicroMap/OmmBaker.h"

using namespace donut::engine;


PrepareLightsPass::PrepareLightsPass(
    nvrhi::IDevice* device, 
    std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, 
    std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
    std::shared_ptr<donut::engine::ExtendedScene> scene,
    std::shared_ptr<MaterialsBaker> materialsBaker,
    std::shared_ptr<OmmBaker> ommBaker,

    nvrhi::BufferHandle subInstanceData,
    nvrhi::IBindingLayout* bindlessLayout,
    std::shared_ptr<ShaderDebug> shaderDebug)
    : m_device(device)
    , m_bindlessLayout(bindlessLayout)
    , m_shaderFactory(std::move(shaderFactory))
    , m_commonPasses(std::move(commonPasses))
    , m_Scene(std::move(scene))
    , m_materialsBaker(materialsBaker)
    , m_ommBaker(ommBaker)
    , m_subInstanceData(subInstanceData)
    , m_shaderDebug(shaderDebug)
{
    nvrhi::BindingLayoutDesc bindingLayoutDesc;
    bindingLayoutDesc.visibility = nvrhi::ShaderType::Compute;
    bindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0), //PushConstants(0, sizeof(PrepareLightsConstants)),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0),
        nvrhi::BindingLayoutItem::TypedBuffer_UAV(1),
        nvrhi::BindingLayoutItem::Texture_UAV(2),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5),
        nvrhi::BindingLayoutItem::Texture_SRV(6),
        nvrhi::BindingLayoutItem::Texture_SRV(7),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(8),
        nvrhi::BindingLayoutItem::Texture_UAV(50),

        nvrhi::BindingLayoutItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX),

        nvrhi::BindingLayoutItem::Sampler(0),
        nvrhi::BindingLayoutItem::Sampler(1),
        nvrhi::BindingLayoutItem::Sampler(2)
    };

    m_bindingLayout = m_device->createBindingLayout(bindingLayoutDesc);

    nvrhi::BufferDesc constBufferDesc;
    constBufferDesc.byteSize = sizeof(PrepareLightsConstants);
    constBufferDesc.debugName = "PrepareLightsConstants";
    constBufferDesc.isConstantBuffer = true;
    constBufferDesc.isVolatile = true;
    constBufferDesc.maxVersions = 16;
    m_constantBuffer = device->createBuffer(constBufferDesc);
}


void PrepareLightsPass::SetScene(std::shared_ptr<donut::engine::ExtendedScene> scene,
    std::shared_ptr<EnvMapBaker> environmentMap, EnvMapSceneParams envMapSceneParams)
{
    m_Scene = scene;
    m_EnvironmentMap = environmentMap;
    m_EnvironmentMapSceneParams = envMapSceneParams;
}

void PrepareLightsPass::CreatePipeline()
{
    donut::log::debug("Initializing PrepareLightsPass...");

    m_computeShader = m_shaderFactory->CreateShader("app/RTXDI/PrepareLights.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

    nvrhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.bindingLayouts = { m_bindingLayout, m_bindlessLayout };
    pipelineDesc.CS = m_computeShader;
    m_computePipeline = m_device->createComputePipeline(pipelineDesc);
}

void PrepareLightsPass::CreateBindingSet(RtxdiResources& resources, const RenderTargets& renderTargets)
{
    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),// PushConstants(0, sizeof(PrepareLightsConstants)),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(0, resources.LightDataBuffer),
        nvrhi::BindingSetItem::TypedBuffer_UAV(1, resources.LightIndexMappingBuffer),
        nvrhi::BindingSetItem::Texture_UAV(2, resources.LocalLightPdfTexture),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, resources.TaskBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_subInstanceData),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_Scene->GetInstanceBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_Scene->GetGeometryBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_ommBaker->GetGeometryDebugBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(5, m_materialsBaker->GetMaterialDataBuffer()),
        nvrhi::BindingSetItem::Texture_SRV(6, m_EnvironmentMap ? m_EnvironmentMap->GetEnvMapCube() : m_commonPasses->m_BlackCubeMapArray),
        nvrhi::BindingSetItem::Texture_SRV(7, m_EnvironmentMap ? m_EnvironmentMap->GetImportanceSampling()->GetImportanceMap() : m_commonPasses->m_BlackTexture),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(8, resources.PrimitiveLightBuffer),
        nvrhi::BindingSetItem::Texture_UAV(50, m_shaderDebug->GetDebugVizTexture()), // TODO: move to shader debug uav

        nvrhi::BindingSetItem::RawBuffer_UAV(SHADER_DEBUG_BUFFER_UAV_INDEX, m_shaderDebug->GetGPUWriteBuffer()),

        nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_AnisotropicWrapSampler),
        nvrhi::BindingSetItem::Sampler(1, m_EnvironmentMap ? m_EnvironmentMap->GetEnvMapCubeSampler() : m_commonPasses->m_AnisotropicWrapSampler),
        nvrhi::BindingSetItem::Sampler(2, m_EnvironmentMap ? m_EnvironmentMap->GetImportanceSampling()->GetImportanceMapSampler() : m_commonPasses->m_AnisotropicWrapSampler)
    };

    m_bindingSet = m_device->createBindingSet(bindingSetDesc, m_bindingLayout);
    m_TaskBuffer = resources.TaskBuffer;
    m_PrimitiveLightBuffer = resources.PrimitiveLightBuffer;
    m_LightIndexMappingBuffer = resources.LightIndexMappingBuffer;
    m_GeometryInstanceToLightBuffer = resources.GeometryInstanceToLightBuffer;
    m_LocalLightPdfTexture = resources.LocalLightPdfTexture;
    m_MaxLightsInBuffer = uint32_t(resources.LightDataBuffer->getDesc().byteSize / (sizeof(PolymorphicLightInfoFull) * 2));
}

void PrepareLightsPass::CountLightsInScene(uint32_t& numEmissiveMeshes, uint32_t& numEmissiveTriangles)
{
    numEmissiveMeshes = 0;
    numEmissiveTriangles = 0;

    const auto& instances = m_Scene->GetSceneGraph()->GetMeshInstances();
    for (const auto& instance : instances)
    {
        for (const auto& geometry : instance->GetMesh()->geometries)
        {
            MaterialPT & materialPT = *MaterialPT::FromDonut(geometry->material);
            if (materialPT.IsEmissive())
            {
                numEmissiveMeshes += 1;
                numEmissiveTriangles += geometry->numIndices / 3;
            }
        }
    }
}

static inline uint floatToUInt(float _V, float _Scale)
{
    return (uint)floor(_V * _Scale + 0.5f);
}

static inline uint FLOAT3_to_R8G8B8_UNORM(float unpackedInputX, float unpackedInputY, float unpackedInputZ)
{
    return (floatToUInt(saturate(unpackedInputX), 0xFF) & 0xFF) |
        ((floatToUInt(saturate(unpackedInputY), 0xFF) & 0xFF) << 8) |
        ((floatToUInt(saturate(unpackedInputZ), 0xFF) & 0xFF) << 16);
}

static void packLightColor(const float3& color, PolymorphicLightInfoFull& lightInfo)
{
    float maxRadiance = std::max(color.x, std::max(color.y, color.z));

    if (maxRadiance <= 0.f)
        return;

    float logRadiance = (::log2f(maxRadiance) - kPolymorphicLightMinLog2Radiance) / (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance);
    logRadiance = saturate(logRadiance);
    uint32_t packedRadiance = std::min(uint32_t(ceilf(logRadiance * 65534.f)) + 1, 0xffffu);
    float unpackedRadiance = ::exp2f((float(packedRadiance - 1) / 65534.f) * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance);

    lightInfo.Base.ColorTypeAndFlags |= FLOAT3_to_R8G8B8_UNORM(color.x / unpackedRadiance, color.y / unpackedRadiance, color.z / unpackedRadiance);
    lightInfo.Base.LogRadiance |= packedRadiance;
}

static float2 OctWrap(float2 v)
{
    return float2( ( 1.0f - abs(v.y)) * ((v.x >= 0.0)?1.0f:-1.0f), 
                   ( 1.0f - abs(v.x)) * ((v.y >= 0.0)?1.0f:-1.0f) );
}

static float2 Encode_Oct(float3 n3)
{
    n3 /= (abs(n3.x) + abs(n3.y) + abs(n3.z));
    float2 n = n3.xy();
    n = n3.z >= 0.0 ? n : OctWrap(n);
    n = n * 0.5f + 0.5f;
    return n;
}

static uint NDirToOctUnorm32(float3 n)
{
    float2 p = Encode_Oct(n);
    p = saturate(p * 0.5f + 0.5f);
    return uint(p.x * 0xfffe) | (uint(p.y * 0xfffe) << 16);
}

// Modified from original, based on the method from the DX fallback layer sample
static uint16_t fp32ToFp16(float v)
{
    // Multiplying by 2^-112 causes exponents below -14 to denormalize
    static const union FU {
        uint ui;
        float f;
    } multiple = { 0x07800000 }; // 2**-112

    FU BiasedFloat;
    BiasedFloat.f = v * multiple.f;
    const uint u = BiasedFloat.ui;

    const uint sign = u & 0x80000000;
    uint body = u & 0x0fffffff;

    return (uint16_t)(sign >> 16 | body >> 13) & 0xFFFF;
}

static bool ConvertLight(const donut::engine::Light& light, PolymorphicLightInfoFull& polymorphic, bool enableImportanceSampledEnvironmentLight, EnvMapBaker* environmentMap)
{
    switch (light.GetLightType())
    {
#if 0 // we're now baking these into the environment map!
    case LightType_Directional: {
        auto& directional = static_cast<const donut::engine::DirectionalLight&>(light);
        float clampedAngularSize = clamp(directional.angularSize, 0.f, 90.f);
        float halfAngularSizeRad = 0.5f * dm::radians(clampedAngularSize);
        float solidAngle = float(2 * dm::PI_d * (1.0 - cos(halfAngularSizeRad)));
        float3 radiance = directional.color * directional.irradiance / solidAngle;

        polymorphic.Base.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kDirectional << kPolymorphicLightTypeShift;
        packLightColor(radiance, polymorphic);
        polymorphic.Base.Direction1 = NDirToOctUnorm32(float3(normalize(directional.GetDirection())));
        // Can't pass cosines of small angles reliably with fp16
        polymorphic.Base.Scalars = fp32ToFp16(halfAngularSizeRad) | (fp32ToFp16(solidAngle) << 16);
        return true;
    }
#endif
 
	case LightType_Spot: {
		auto& spot = static_cast<const SpotLight&>(light);

        // Sphere lights not supported in the RTXPT currently
        if (spot.radius == 0.f)
        {
			float3 flux = spot.color * spot.intensity;

			polymorphic.Base.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kPoint << kPolymorphicLightTypeShift;
			packLightColor(flux, polymorphic);
			polymorphic.Base.Center = float3(spot.GetPosition());
            polymorphic.Base.Direction1 = NDirToOctUnorm32(float3(normalize(spot.GetDirection())));
            polymorphic.Base.Direction2 = fp32ToFp16(dm::radians(spot.outerAngle));
			polymorphic.Base.Direction2 |= fp32ToFp16(dm::radians(spot.innerAngle)) << 16;
        }
        else
        {

            float projectedArea = dm::PI_f * square(spot.radius);
            float3 radiance = spot.color * spot.intensity / projectedArea;
            float softness = saturate(1.f - spot.innerAngle / spot.outerAngle);

            polymorphic.Base.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift;
            polymorphic.Base.ColorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
            packLightColor(radiance, polymorphic);
            polymorphic.Base.Center = float3(spot.GetPosition());
            polymorphic.Base.Scalars = fp32ToFp16(spot.radius);
            polymorphic.Extended.PrimaryAxis = NDirToOctUnorm32(float3(normalize(spot.GetDirection())));
            polymorphic.Extended.CosConeAngleAndSoftness = fp32ToFp16(cosf(dm::radians(spot.outerAngle)));
            polymorphic.Extended.CosConeAngleAndSoftness |= fp32ToFp16(softness) << 16;
        }

		return true;
	}
   /* case LightType_Spot: {
   *    // Spot Light with ies profile
        auto& spot = static_cast<const SpotLightWithProfile&>(light);
        float projectedArea = dm::PI_f * square(spot.radius);
        float3 radiance = spot.color * spot.intensity / projectedArea;
        float softness = saturate(1.f - spot.innerAngle / spot.outerAngle);

        polymorphic.Base.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift;
        polymorphic.Base.ColorTypeAndFlags |= kPolymorphicLightShapingEnableBit;
        packLightColor(radiance, polymorphic);
        polymorphic.Base.Center = float3(spot.GetPosition());
        polymorphic.Base.Scalars = fp32ToFp16(spot.radius);
        polymorphic.Extended.PrimaryAxis = packNormalizedVector(float3(normalize(spot.GetDirection())));
        polymorphic.Extended.CosConeAngleAndSoftness = fp32ToFp16(cosf(dm::radians(spot.outerAngle)));
        polymorphic.Extended.CosConeAngleAndSoftness |= fp32ToFp16(softness) << 16;

        if (spot.profileTextureIndex >= 0)
        {
            polymorphic.Extended.iesProfileIndex = spot.profileTextureIndex;
            polymorphic.Extended.ColorTypeAndFlags |= kPolymorphicLightIesProfileEnableBit;
        }

        return true;
    }*/
    case LightType_Point: {
        auto& point = static_cast<const donut::engine::PointLight&>(light);
     
        if (point.radius == 0.f)
        {
            float3 flux = point.color * point.intensity;

            polymorphic.Base.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kPoint << kPolymorphicLightTypeShift;
            packLightColor(flux, polymorphic);
            polymorphic.Base.Center = float3(point.GetPosition());
            // Set the default values so we can use the same path for spot lights 
            polymorphic.Base.Direction2 = fp32ToFp16(dm::PI_f) | fp32ToFp16(0.0f) << 16;
        }
        else
        {
            float projectedArea = dm::PI_f * square(point.radius);
            float3 radiance = point.color * point.intensity / projectedArea;

            polymorphic.Base.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kSphere << kPolymorphicLightTypeShift;
            packLightColor(radiance, polymorphic);
            polymorphic.Base.Center = float3(point.GetPosition());
            polymorphic.Base.Scalars = fp32ToFp16(point.radius);
        }

        return true;
    }
    case LightType_Environment: {
        // do not add an environment light to the light list if the option is disabled 
        if (enableImportanceSampledEnvironmentLight)
        {
            auto& env = static_cast<const EnvironmentLight&>(light);

            polymorphic.Base.ColorTypeAndFlags = (uint32_t)PolymorphicLightType::kEnvironment << kPolymorphicLightTypeShift;
        
            polymorphic.Base.Direction2 = 0;

            return true;
        }
        return false;
    }
    default:
        return false;
    }
}

static int isInfiniteLight(const donut::engine::Light& light)
{
    switch (light.GetLightType())
    {
    case LightType_Directional:
        return 1;

    case LightType_Environment:
        return 2;

    default:
        return 0;
    }
}

RTXDI_LightBufferParameters PrepareLightsPass::Process(nvrhi::ICommandList* commandList)
{
    RTXDI_LightBufferParameters lightBufferParams = {};
    
    commandList->beginMarker("PrepareLights");

    std::vector<PrepareLightsTask> tasks;
    std::vector<PolymorphicLightInfoFull> primitiveLightInfos;
    uint32_t lightBufferOffset = 0;
    std::vector<uint32_t> geometryInstanceToLight(m_Scene->GetSceneGraph()->GetGeometryInstancesCount(), RTXDI_INVALID_LIGHT_INDEX);

    const auto& instances = m_Scene->GetSceneGraph()->GetMeshInstances();
    for (const auto& instance : instances)
    {
        const auto& mesh = instance->GetMesh();

        assert(instance->GetGeometryInstanceIndex() < geometryInstanceToLight.size());
        uint32_t firstGeometryInstanceIndex = instance->GetGeometryInstanceIndex();

        for (size_t geometryIndex = 0; geometryIndex < mesh->geometries.size(); ++geometryIndex)
        {
            const auto& geometry = mesh->geometries[geometryIndex];

            size_t instanceHash = 0;
            nvrhi::hash_combine(instanceHash, instance.get());
            nvrhi::hash_combine(instanceHash, geometryIndex);

            MaterialPT & materialPT = *MaterialPT::FromDonut(geometry->material);
            if (!materialPT.IsEmissive())
            {
                // remove the info about this instance, just in case it was emissive and now it's not
                m_InstanceLightBufferOffsets.erase(instanceHash);
                continue;
            }

            geometryInstanceToLight[firstGeometryInstanceIndex + geometryIndex] = lightBufferOffset;

            // find the previous offset of this instance in the light buffer
            auto pOffset = m_InstanceLightBufferOffsets.find(instanceHash);

            assert(geometryIndex < 0xfff);

            PrepareLightsTask task;
            task.instanceAndGeometryIndex = (instance->GetInstanceIndex() << 12) | uint32_t(geometryIndex & 0xfff);
            task.lightBufferOffset = lightBufferOffset;
            task.triangleCount = geometry->numIndices / 3;
            task.previousLightBufferOffset = (pOffset != m_InstanceLightBufferOffsets.end()) ? int(pOffset->second) : -1;

            // record the current offset of this instance for use on the next frame
            m_InstanceLightBufferOffsets[instanceHash] = lightBufferOffset;

            lightBufferOffset += task.triangleCount;

            tasks.push_back(task);
        }
    }

    commandList->writeBuffer(m_GeometryInstanceToLightBuffer, geometryInstanceToLight.data(), geometryInstanceToLight.size() * sizeof(uint32_t));

	lightBufferParams.localLightBufferRegion.firstLightIndex = 0;
	lightBufferParams.localLightBufferRegion.numLights = lightBufferOffset;

    auto sortedLights = m_Scene->GetSceneGraph()->GetLights();
    std::sort(sortedLights.begin(), sortedLights.end(), [](const auto& a, const auto& b) 
        { return isInfiniteLight(*a) < isInfiniteLight(*b); });

    uint32_t numFinitePrimLights = 0;
    uint32_t numInfinitePrimLights = 0;

    bool enableImportanceSampledEnvironmentLight = m_EnvironmentMap ? true : false;

    for (const std::shared_ptr<Light>& pLight : sortedLights)
    {
        PolymorphicLightInfoFull polymorphicLight = {};
       
        if (!ConvertLight(*pLight, polymorphicLight, enableImportanceSampledEnvironmentLight, m_EnvironmentMap.get()))
            continue;

        // find the previous offset of this instance in the light buffer
        auto pOffset = m_PrimitiveLightBufferOffsets.find(pLight.get());

        PrepareLightsTask task;
        task.instanceAndGeometryIndex = TASK_PRIMITIVE_LIGHT_BIT | uint32_t(primitiveLightInfos.size());
        task.lightBufferOffset = lightBufferOffset;
        task.triangleCount = 1; // technically zero, but we need to allocate 1 thread in the grid to process this light
        task.previousLightBufferOffset = (pOffset != m_PrimitiveLightBufferOffsets.end()) ? pOffset->second : -1;

        // record the current offset of this instance for use on the next frame
        m_PrimitiveLightBufferOffsets[pLight.get()] = lightBufferOffset;

        lightBufferOffset += task.triangleCount;

        tasks.push_back(task);
        primitiveLightInfos.push_back(polymorphicLight);

        if (isInfiniteLight(*pLight))
            numInfinitePrimLights++;
        else
            numFinitePrimLights++;
    }

	lightBufferParams.localLightBufferRegion.numLights += numFinitePrimLights;
	lightBufferParams.infiniteLightBufferRegion.firstLightIndex = lightBufferParams.localLightBufferRegion.numLights;
    // Note we do not include the environment map in numInfiniteLights
	lightBufferParams.infiniteLightBufferRegion.numLights = numInfinitePrimLights - enableImportanceSampledEnvironmentLight;;
	lightBufferParams.environmentLightParams.lightIndex = lightBufferParams.infiniteLightBufferRegion.firstLightIndex + lightBufferParams.infiniteLightBufferRegion.numLights;
	lightBufferParams.environmentLightParams.lightPresent = enableImportanceSampledEnvironmentLight ? 1u : 0u;
    
    commandList->writeBuffer(m_TaskBuffer, tasks.data(), tasks.size() * sizeof(PrepareLightsTask));

    if (!primitiveLightInfos.empty())
    {
        commandList->writeBuffer(m_PrimitiveLightBuffer, primitiveLightInfos.data(), primitiveLightInfos.size() * sizeof(PolymorphicLightInfoFull));
    }

    // clear the mapping buffer - value of 0 means all mappings are invalid
    commandList->clearBufferUInt(m_LightIndexMappingBuffer, 0);

    // Clear the PDF texture mip 0 - not all of it might be written by this shader
    commandList->clearTextureFloat(m_LocalLightPdfTexture, 
        nvrhi::TextureSubresourceSet(0, 1, 0, 1), 
        nvrhi::Color(0.f));

    nvrhi::ComputeState state;
    state.pipeline = m_computePipeline;
    state.bindings = { m_bindingSet, m_Scene->GetDescriptorTable() };

    PrepareLightsConstants constants;
    constants.numTasks = uint32_t(tasks.size());
    constants.currentFrameLightOffset = m_MaxLightsInBuffer * m_OddFrame;
    constants.previousFrameLightOffset = m_MaxLightsInBuffer * !m_OddFrame;
    constants._padding = 0;
    constants.envMapSceneParams = {};
    if (enableImportanceSampledEnvironmentLight)
    {
        constants.envMapSceneParams = m_EnvironmentMapSceneParams;
        constants.envMapImportanceSamplingParams = m_EnvironmentMap->GetImportanceSampling()->GetShaderParams( );
    }

    commandList->writeBuffer(m_constantBuffer, &constants, sizeof(PrepareLightsConstants));
    //commandList->setPushConstants(&constants, sizeof(constants));

    commandList->setComputeState(state);
    
    //Skip the prepare lights dispatch if there are no lights. Note the Environment map is handled in another pass
    if (lightBufferOffset > 0)
        commandList->dispatch(dm::div_ceil(lightBufferOffset, 256));

    commandList->endMarker();

	lightBufferParams.localLightBufferRegion.firstLightIndex += constants.currentFrameLightOffset;
	lightBufferParams.infiniteLightBufferRegion.firstLightIndex += constants.currentFrameLightOffset;
	lightBufferParams.environmentLightParams.lightIndex += constants.currentFrameLightOffset;
    
    m_OddFrame = !m_OddFrame;

    return lightBufferParams;
}

nvrhi::TextureHandle PrepareLightsPass::GetEnvironmentMapTexture()
{
    return m_EnvironmentMap ? m_EnvironmentMap->GetEnvMapCube() : nullptr;
}
