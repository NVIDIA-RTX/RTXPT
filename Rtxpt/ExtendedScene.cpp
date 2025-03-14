/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ExtendedScene.h"
#include <donut/core/json.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/log.h>
#include <json/value.h>
#include <nvrhi/utils.h>
#include <nvrhi/common/misc.h>

using namespace donut::math;
#include <donut/shaders/light_cb.h>

#include "Materials/MaterialsBaker.h"
#include "OpacityMicroMap/OmmBaker.h"

#include "LocalConfig.h"

using namespace donut;
using namespace donut::engine;

std::shared_ptr<engine::SceneGraphLeaf> EnvironmentLight::Clone()
{
    auto copy = std::make_shared<EnvironmentLight>();
    copy->color = color;
    copy->radianceScale = radianceScale;
    copy->textureIndex = textureIndex;
    copy->rotation = rotation;
    copy->path = path;
    return std::static_pointer_cast<SceneGraphLeaf>(copy);
}

void EnvironmentLight::Load(const Json::Value& node)
{
    node["radianceScale"] >> radianceScale;
    node["textureIndex"] >> textureIndex;
    node["rotation"] >> rotation;
    node["path"] >> path;
}
std::shared_ptr<donut::engine::SceneGraphLeaf> ExtendedSceneTypeFactory::CreateLeaf(const std::string& type)
{
    if (type == "EnvironmentLight")
    {
        return std::make_shared<EnvironmentLight>();
    } else
    if (type == "PerspectiveCamera" || type == "PerspectiveCameraEx")
    {
        return std::make_shared<PerspectiveCameraEx>();
    } else
    if (type == "MaterialPatch")
    {
        assert(false && "Your .scene.json file is out of date, this codepath is no longer supported. Please update your media folder. Loading will continue but some material properties will be missing.");
        return nullptr;
    } else
    if (type == "SampleSettings")
    {
        return std::make_shared<SampleSettings>();
    }
    return SceneTypeFactory::CreateLeaf(type);
}

std::shared_ptr<MeshInfo> ExtendedSceneTypeFactory::CreateMesh()
{
    return std::make_shared<MeshInfoEx>();
}

std::shared_ptr<MeshGeometry> ExtendedSceneTypeFactory::CreateMeshGeometry()
{
    return std::make_shared<MeshGeometryEx>();
}

std::shared_ptr<Material> ExtendedSceneTypeFactory::CreateMaterial()
{
    return std::static_pointer_cast<Material>(std::make_shared<MaterialEx>());
}

void ExtendedScene::ProcessNodesRecursive(donut::engine::SceneGraphNode* node)
{
    // std::find_if doesn't compile on linux.
    auto _find_if = [](
        ResourceTracker<Material>::ConstIterator begin,
        ResourceTracker<Material>::ConstIterator end,
        std::function<bool(const std::shared_ptr<Material>& mat)> fn)->ResourceTracker<Material>::ConstIterator
    {
        for (ResourceTracker<Material>::ConstIterator it = begin; it != end; it++)
        {
            if (fn(*it)) {
                return it;
            }
        }
        return end;
    };

    if (node->GetLeaf() != nullptr)
    {
#if 0 // material patching no longer supported - leaving this as a future reference
        std::shared_ptr<MaterialPatch> materialPatch = std::dynamic_pointer_cast<MaterialPatch>(node->GetLeaf());
        if (materialPatch != nullptr)
        {
            const std::string name = node->GetName();

            auto & materials = m_SceneGraph->GetMaterials();
            auto it = _find_if( materials.begin(), materials.end(), [&name](const std::shared_ptr<Material> & mat) { return mat->name == name; });
            if (it == materials.end())
            {
                log::warning("Material patch '%s' can't find material to patch!", name.c_str() );
                assert( false );
            }
            else
            {
                materialPatch->Patch(**it);
            }
        }
#endif
        std::shared_ptr<SampleSettings> sampleSettings = std::dynamic_pointer_cast<SampleSettings>(node->GetLeaf());
        if (sampleSettings != nullptr)
        {
            assert(m_loadedSettings == nullptr);    // multiple settings nodes? only last one will be loaded
            m_loadedSettings = sampleSettings;
        }
    }

    for( int i = (int)node->GetNumChildren()-1; i >= 0; i-- )
        ProcessNodesRecursive(node->GetChild(i));
}

bool ExtendedScene::LoadWithExecutor(const std::filesystem::path& jsonFileName, tf::Executor* executor)
{
	if (!Scene::LoadWithExecutor(jsonFileName, executor))
		return false;

    ProcessNodesRecursive( GetSceneGraph()->GetRootNode().get() );

#if 1 // example of modifying all materials after scene loading; this is the ideal place to do material modification without worrying about resetting relevant caches/dependencies
    auto& materials = m_SceneGraph->GetMaterials();
    for( auto it : materials )
    {
        Material & mat = *it;
        LocalConfig::PostMaterialLoad(mat);
    }
#endif

    return true;
}

std::shared_ptr<EnvironmentLight> donut::engine::FindEnvironmentLight(std::vector <std::shared_ptr<Light>> lights)
{
    for (auto light : lights)
    {
        if (light->GetLightType() == LightType_Environment)
        {
            return std::dynamic_pointer_cast<EnvironmentLight>(light);
        }
    }
    return nullptr;
}

void EnvironmentLight::FillLightConstants(LightConstants& lightConstants) const
{
    Light::FillLightConstants(lightConstants);
    lightConstants.intensity = 0.0f;
    lightConstants.color = { 0,0,0 };
}

std::shared_ptr<SceneGraphLeaf> PerspectiveCameraEx::Clone()
{
    auto copy = std::make_shared<PerspectiveCameraEx>();
    copy->zNear = zNear;
    copy->zFar = zFar;
    copy->verticalFov = verticalFov;
    copy->aspectRatio = aspectRatio;
    copy->enableAutoExposure = enableAutoExposure;
    copy->exposureCompensation = exposureCompensation;
    copy->exposureValue = exposureValue;
    copy->exposureValueMin = exposureValueMin;
    copy->exposureValueMax = exposureValueMax;
    return copy;
}

void PerspectiveCameraEx::Load(const Json::Value& node)
{
    node["enableAutoExposure"] >> enableAutoExposure;
    node["exposureCompensation"] >> exposureCompensation;
    node["exposureValue"] >> exposureValue;
    node["exposureValueMin"] >> exposureValueMin;
    node["exposureValueMax"] >> exposureValueMax;
    
    PerspectiveCamera::Load(node);
}

bool PerspectiveCameraEx::SetProperty(const std::string& name, const dm::float4& value)
{
    assert(false); // not implemented
    return PerspectiveCamera::SetProperty(name, value);
}

std::shared_ptr<SceneGraphLeaf> SampleSettings::Clone()
{
    auto copy = std::make_shared<SampleSettings>();
    assert(false); // not properly implemented
    return copy;
}

void SampleSettings::Load(const Json::Value& node)
{
    node["realtimeMode"] >> realtimeMode;
    node["enableAnimations"] >> enableAnimations;
    node["enableReSTIRDI"] >> enableReSTIRDI;
    node["enableReSTIRGI"] >> enableReSTIRGI;
    node["startingCamera"] >> startingCamera;
    node["realtimeFireflyFilter"] >> realtimeFireflyFilter;
    node["maxBounces"] >> maxBounces;
    node["realtimeMaxDiffuseBounces"] >> realtimeMaxDiffuseBounces;
    node["referenceMaxDiffuseBounces"] >> referenceMaxDiffuseBounces;
    node["textureMIPBias"] >> textureMIPBias;
}

