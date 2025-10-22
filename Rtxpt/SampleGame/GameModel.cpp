/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "GameScene.h"
#include "GameModel.h"

#include <donut/core/log.h>
#include <donut/core/json.h>
#include <donut/core/math/math.h>
#include <donut/app/Camera.h>
#include <cmath>

#include "../ExtendedScene.h"

#include "../Misc/Korgi.h"

#include <fstream>
#include <iostream>
#include <thread>

using namespace donut;
using namespace donut::math;
using namespace donut::app;
using namespace donut::vfs;
using namespace donut::engine;
using namespace donut::render;

using namespace game;

void SpecialFixupsRec(const std::string & modelName, SceneGraphNode* node, dm::daffine3 globalTransform)
{
    dm::daffine3 transform = dm::scaling(node->GetScaling());
    transform *= node->GetRotation().toAffine();
    transform *= dm::translation(node->GetTranslation());
    globalTransform = transform * globalTransform;

    double3 scale;
    dm::decomposeAffine<double>(globalTransform, nullptr, nullptr, &scale);
    float reallyAverageScale = float((scale.x + scale.y + scale.z) / 3.0);

//    SpotLight* spotLight = dynamic_cast<SpotLight*>(node->GetLeaf().get());
//
//    if (spotLight != nullptr)
//    {
//        //spotLight->range = 10000.0f;
//        spotLight->intensity = 50.0f;   // TODO: fix on blender side
//        spotLight->innerAngle;
//        spotLight->outerAngle;
//        spotLight->radius = reallyAverageScale;
//
//        // if (spotLight->GetName() == "SpotLeft")
//        spotLight->outerAngle = -abs(spotLight->outerAngle); // this is a special flag to indicate that, instead of falling off to 0 when angle >= outerAngle, we should clamp to kMinSpotlightFalloff
//    }
//
//    PointLight* pointLight = dynamic_cast<PointLight*>(node->GetLeaf().get());
//
//    if (pointLight != nullptr)
//    {
//        pointLight->radius = reallyAverageScale;
//    }

    // if (modelName == "RX6SpaceShip")
    // if (modelName == "OrbLight")

    for (int i = 0; i < node->GetNumChildren(); i++)
        SpecialFixupsRec(modelName, node->GetChild(i), globalTransform);
}

// For stuff where it's easier to fix in code vs json
static void SpecialFixups(const std::string & modelName, SceneGraphNode* rootNode)
{
    dm::daffine3 transform = dm::daffine3::identity();
    SpecialFixupsRec(modelName, rootNode, transform);
}

ModelType::ModelType(class GameScene & game, const std::string & name, const Json::Value & node)
    : m_game(game)
{
    m_valid = false;
    const std::shared_ptr<ExtendedScene> & scene = game.GetScene();

    // node["name"] >> m_name;
    m_name = name;

    int modelIndex = -1;
    std::string modelName = "";
    node["sceneModelIndex"] >> modelIndex;
    node["sceneModelName"] >> modelName;

    node["modelPose"] >> m_modelPose;

    auto lights = node["lights"];
    if (lights.isArray())
    {
        for( auto & lightData : lights )
        {
            std::string lightName;
            lightData["name"] >> lightName;
            if (lightName == "" || m_lightsInfos.find(lightName) != m_lightsInfos.end())
            { assert( false && "malformed or repeated light name in .model.json" );  continue; }
            m_lightsInfos.insert( {lightName, SaveJsonToString(lightData) } );
        }
        ///m_lightsInfoJson = SaveJsonToString(lights);
    }

    const std::vector<donut::engine::SceneImportResult>& models = scene->GetModels();

    if (modelIndex==-1 && modelName != "")
        for( int i = 0; i < models.size(); i++ )
            if ( FindSubStringIgnoreCase(models[i].rootNode->GetName(), modelName) != std::string::npos )
            {
                modelIndex = i;
                break;
            }

    if (modelIndex<0 || modelIndex>=models.size())
    {
        log::warning("Referenced model %d is not defined in the model array.", modelIndex);
        return; 
    }

    const auto& loadedModel = models[modelIndex];
    if (!loadedModel.rootNode)
    { assert( false ); return; }

    loadedModel.rootNode->SetTransform(&m_modelPose.Translation, &m_modelPose.Rotation, &m_modelPose.Scaling);

    m_node = loadedModel.rootNode;

    SpecialFixups( m_name, loadedModel.rootNode.get() );

    m_valid = m_name != "" && modelIndex != -1;
}

std::string ModelType::FindLightControllerInfo( const std::string & nodeName )
{
    auto it = m_lightsInfos.find(nodeName);
    if (it == m_lightsInfos.end())
        return "";
    else
        return it->second;
}

ModelInstance::ModelInstance( const std::string & name, const std::shared_ptr<ModelType> & modelType, const std::shared_ptr<donut::engine::SceneGraphNode> & parentNode )
    : m_modelType(modelType)
{
    assert( modelType != nullptr );
    const std::shared_ptr<class ExtendedScene> & scene = modelType->GetGame().GetScene();
    m_node = std::make_shared<SceneGraphNode>();

    m_node = scene->GetSceneGraph()->Attach(parentNode, m_node);
    m_node->SetName(name);

    scene->GetSceneGraph()->Attach(m_node, modelType->GetNode());   // each model type has its own root node

    MapLightControllers( m_node.get() );
}

void ModelInstance::MapLightControllers( SceneGraphNode* node )
{
    if (node->GetLeaf() != nullptr && (node->GetLeaf()->GetContentFlags() & SceneContentFlags::Lights) != 0)
    {
        std::string data = m_modelType->FindLightControllerInfo(node->GetName());
        Json::Value jsonData;
        if (data != "" && LoadJsonFromString(data, jsonData))
        {
            auto lightController = std::make_shared<LightController>();
            if (!lightController->Read(jsonData))
            {
                assert(false && "Error reading LightController data");
            }
            else
            {
                lightController->Node = node;
                m_lightControllers.push_back(lightController);
            }
        }
        else
        {
            log::warning( "Model instance '%s', light '%s' has no controller", m_node->GetName().c_str(), node->GetName().c_str() );
        }
    }
    for (int i = 0; i < node->GetNumChildren(); i++)
        MapLightControllers(node->GetChild(i));
}

void ModelInstance::UpdateLightFromControllers(double gameTime)
{
    for (const auto & controller : m_lightControllers)
    {
        auto node = controller->Node;

        // NOTE - this gets frame old world transform - if changing scale at runtime, this won't work and instead it has to be fully updated
        float3 scale;
        dm::decomposeAffine<float>(node->GetLocalToWorldTransformFloat(), nullptr, nullptr, &scale);
        float reallyAverageScale = (scale.x + scale.y + scale.z) / 3.0f;

        Light* light = dynamic_cast<Light*>(node->GetLeaf().get());
        assert( light != nullptr );
        SpotLight* spotLight = dynamic_cast<SpotLight*>(node->GetLeaf().get());
        PointLight* pointLight = dynamic_cast<PointLight*>(node->GetLeaf().get());

        light->color = controller->Color;

        bool enabled = controller->Enabled;

        if (controller->AutoOffTime != 0 && controller->AutoOnTime != 0)
        {
            double periodLength = controller->AutoOffTime + controller->AutoOnTime;
            double scaledTime = (gameTime+controller->AutoOnOffTimeOffset) / periodLength;
            double remainder = dm::saturate<double>(scaledTime - floor(scaledTime));
            enabled &= remainder < (controller->AutoOffTime / periodLength);
        }

        float intensity = (enabled)?(controller->Intensity):(0);
        
        if (spotLight != nullptr)
        {
            spotLight->radius = reallyAverageScale;
            spotLight->intensity = intensity;
            
            spotLight->outerAngle = abs(controller->OuterAngle);
            spotLight->innerAngle = controller->InnerAngle;
            
            const bool kUseMinSpotlightFalloff = true;
            if (kUseMinSpotlightFalloff)
                spotLight->outerAngle = -spotLight->outerAngle; // this is a special flag to indicate that, instead of falling off to 0 when angle >= outerAngle, we should clamp to kMinSpotlightFalloff

        }
        if (pointLight != nullptr)
        {
            pointLight->radius = reallyAverageScale;
            pointLight->intensity = intensity;
        }
    }
}

void ModelInstance::SetTransform(const dm::double3& translation, const dm::dquat& rotation, const dm::double3& scaling)
{
    m_node->SetTransform(&translation, &rotation, &scaling);
}

void ModelInstance::SetTransform(const dm::float3 & translation, const dm::quat & rotation, const dm::float3 & scaling)
{
    dm::double3 transD = dm::double3(translation);
    dm::dquat rotD = dm::dquat(rotation);
    dm::double3 scalD = dm::double3(scaling);
    m_node->SetTransform(&transD, &rotD, &scalD);
}