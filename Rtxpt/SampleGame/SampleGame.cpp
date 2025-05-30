/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SampleGame.h"

#include <donut/core/log.h>
#include <donut/core/json.h>
#include <donut/core/math/math.h>
#include <donut/app/Camera.h>
#include <cmath>

#include "../ExtendedScene.h"

#include "../Misc/Korgi.h"
#include <json/json.h>

using namespace donut;
using namespace donut::math;
using namespace donut::app;
using namespace donut::vfs;
using namespace donut::engine;
using namespace donut::render;

#include <fstream>
#include <iostream>
#include <thread>


SampleGame::SampleGame(Sample & sample)
    : m_sample(sample) // NOTE: at this point, Sample is being constructed - beware of accessing incompletely constructed object
{
}

void SampleGame::Deinitialize()
{
    m_scene = nullptr;
    m_active = false;
    m_propInstances.clear();
    m_propTypes.clear();
    m_gameStoragePath = std::filesystem::path();
}

void SampleGame::ResetGame()
{
    m_recordCamera = false;
    for( auto & vehicle : m_propInstances )
        vehicle->Reset();
}

void SampleGame::SetActive(bool active)
{
    if (m_active == active)
        return;

    m_active = active;
    if (m_active == false)
    {
        m_gameTime = 0.0;
    }
}

void SampleGame::LoadPropTypes(const Json::Value & propTypeArray)
{
    if (!propTypeArray.isArray())
    { assert(false); return; }

    assert( m_scene != nullptr );
    for( int i = 0; i < propTypeArray.size(); i++ )
        m_propTypes.push_back( std::make_shared<SampleGamePropType>(*this, propTypeArray[i]));
}

void SampleGame::SceneLoaded(const std::shared_ptr<ExtendedScene>& scene, const std::filesystem::path& sceneFilePath, const std::filesystem::path & mediaPath)
{
    Deinitialize();

    auto gameSettings = scene->GetGameSettingsNode();
    if (gameSettings == nullptr)
        return;

    std::filesystem::path mediaGamePath = mediaPath / std::string(c_SampleGameSubFolder);
    if (!EnsureDirectoryExists(mediaGamePath))
        return;

    std::filesystem::path sceneName = sceneFilePath.filename().stem();
    m_gameStoragePath = mediaGamePath / sceneName;
    if (!EnsureDirectoryExists(m_gameStoragePath))
        return;

    Json::Value node;
    bool parsingSuccessful = LoadJsonFromString(gameSettings->GetJsonData(), node);
    if (!parsingSuccessful) 
    {
        log::warning( "Unable to load game settings" );
        return;
    }

    m_scene = scene;

    Json::Value propTypeArray = node["propTypes"];
    if (!propTypeArray.empty())
        LoadPropTypes(propTypeArray);

    int modelShip = -1;
    node["modelShip"] >> modelShip;
    if (modelShip==-1) 
        return;

    const std::vector<donut::engine::SceneImportResult>& models = scene->GetModels();

    if (modelShip < 0 || modelShip >= int( models.size()))
    {
        log::warning("Referenced model %d is not defined in the model array.", modelShip);
        return;
    }

    SampleGamePose initPose, modelPose;
    if (!initPose.Read(node["startPose"]))
        return;

    auto instanceFiles = EnumerateFilesWithWildcard(m_gameStoragePath, "*.instance.json");

    for( auto instancePath : instanceFiles )
    {
        auto fileNoExt = instancePath.filename();
        fileNoExt.replace_extension();
        fileNoExt.replace_extension();

        Json::Value instanceRoot;
        if (!LoadJsonFromFile(instancePath, instanceRoot) || instanceRoot.empty() || !instanceRoot.isObject() )
            continue;
        std::string propTypeName; 
        instanceRoot["typeName"] >> propTypeName;
        if (propTypeName == "")
            continue;
        auto it = std::find_if(m_propTypes.begin(), m_propTypes.end(), [&propTypeName](const std::shared_ptr<SampleGamePropType> & pt) { return pt->GetTypeName() == propTypeName; } );
        if (it == m_propTypes.end())
            continue;
        
        std::shared_ptr<SampleGamePropInstance> newInstance = SampleGamePropType::CreateInstance(*it, fileNoExt.string(), instancePath, instanceRoot);
        if (newInstance != nullptr)
            m_propInstances.push_back( newInstance );
    }

    // std::srand(0);
    // for (int i = 0; i < (int)m_propInstances.size(); i++)
    // {
    //     SampleGamePropInstance& vehicle = *m_propInstances[i];
    //     std::string fileName = vehicle.GetName() + ".vehicle.json";
    //     vehicle.SetStoragePath(m_gameStoragePath/fileName);
    //     vehicle.Load();
    //     //vehicle.SetAnimOffset(std::rand() / (float)RAND_MAX * 110.0f);
    // }

    // m_cameraNode = scene->GetSceneGraph()->Attach(scene->GetSceneGraph()->GetRootNode(), std::make_shared<SceneGraphNode>());
    // m_cameraNode->SetName("SampleGameCameraNode");
    // m_cameraNode->SetLeaf( m_camera = std::make_shared<PerspectiveCameraEx>() );
    // m_camera->SetName("SampleGameCamera");
}

void SampleGame::SceneUnloading()
{
    Deinitialize();
}

bool SampleGame::DebugGUI(float indent)
{
    if (!m_active)
    {
        if (ImGui::Button("ACTIVATE", ImVec2(-FLT_MIN,0.0f)))
            SetActive(true);
        // m_gameTime = 0;
    }

    if (m_active)
    {
        ImGui::Text("Active, time %f", m_gameTime);
        if (ImGui::Button("DEACTIVATE", ImVec2(-FLT_MIN, 0.0f)))
            SetActive(false);
    }

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
        RAII_SCOPE(ImGui::Indent(indent);, ImGui::Unindent(indent); );
        if (ImGui::Checkbox("Game camera enabled", &m_gameCameraEnabled))
        {
            if (m_gameCameraEnabled)
                m_gameCameraInitialInit = true;
            if (m_gameCameraAttached.lock() != nullptr && ImGui::Button("Detach from object"))
                m_gameCameraAttached.reset();
        }
    }

    if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen))
    {
        RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
        for (int i = 0; i < (int)m_propInstances.size(); i++)
        {
            SampleGamePropInstance & object = *m_propInstances[i];
            ImGui::Text("Obj %d - %s ", i, object.GetName().c_str());
            {
                RAII_SCOPE(ImGui::Indent(indent); ImGui::PushID(i);, ImGui::PopID(); ImGui::Unindent(indent); );

                if (ImGui::Button("Attach camera"))
                    m_gameCameraAttached = m_propInstances[i];
            }
            ImGui::Separator();
        }
    }
    ImGui::Separator();
    ImGui::Text("Recorded poses: %d", m_recordedCameraPoses.size());
    if (m_recordedCameraPoses.size() > 0 && ImGui::Button("SAVE RECORDING"))
    {
        Json::Value rootJ;
            
        for (auto & pose : m_recordedCameraPoses)
            rootJ.append(pose.Write());

        SaveJsonToFile( m_gameStoragePath / "EXPORT_POSES.json", rootJ );
    }

    return false;
}

bool SampleGame::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    if (m_gameCameraEnabled)
        m_gameCamera.KeyboardUpdate(key, scancode, action, mods);

    if (!m_active)
        return false;

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL )
    {
        m_recordCamera = true;
    }

    return false;
}
void SampleGame::MousePosUpdate(double xpos, double ypos)
{
    if (m_gameCameraEnabled)
        m_gameCamera.MousePosUpdate(xpos, ypos);
}
void SampleGame::MouseButtonUpdate(int button, int action, int mods)
{
    if (m_gameCameraEnabled)
        m_gameCamera.MouseButtonUpdate(button, action, mods);
}

void SampleGame::Tick(float deltaTime)
{
    if (!m_active)
        return;

    deltaTime = min(deltaTime, 0.5f);

    m_gameTime += deltaTime;

    for (auto& vehicle : m_propInstances)
        vehicle->Tick(m_gameTime, deltaTime);
}

void SampleGame::TickCamera(float deltaTime, donut::app::FirstPersonCamera & renderCamera)
{
    if (m_gameCameraEnabled)
    {
        if (m_gameCameraInitialInit)
            m_gameCamera.LookTo(renderCamera.GetPosition(), renderCamera.GetDir(), renderCamera.GetUp());
        else
        {
            auto attachedObject = m_gameCameraAttached.lock();
            affine3 transform = affine3::identity();
            if (attachedObject != nullptr)
            {
                // transform = attachedObject->GetRootNode()->GetLocalToWorldTransformFloat(); // we can't do this because these have not yet been updated
                dm::daffine3 transformD = dm::scaling(attachedObject->GetRootNode()->GetScaling());
                transformD *= attachedObject->GetRootNode()->GetRotation().toAffine();
                transformD *= dm::translation(attachedObject->GetRootNode()->GetTranslation());
                transform = dm::affine3(transformD);
            }

            renderCamera.LookTo( transform.transformPoint(m_gameCamera.GetPosition()), transform.transformVector(m_gameCamera.GetDir()), transform.transformVector(m_gameCamera.GetUp()));
        }
        m_gameCameraInitialInit = false;

        m_gameCamera.Animate(deltaTime);
    }

    if (!m_active)
        return;
    
    if (m_recordCamera)
    {
        SampleGamePose pose;
        pose.KeyTime = m_gameTime;
        pose.Translation = double3(renderCamera.GetPosition());

        dm::dquat rotation;
        dm::affine3 sceneWorldToView = dm::scaling(dm::float3(1.f, 1.f, -1.f)) * dm::inverse(renderCamera.GetWorldToViewMatrix());
        dm::decomposeAffine<double>(daffine3(sceneWorldToView), nullptr, &rotation, nullptr);
        pose.Rotation = rotation;
        m_recordedCameraPoses.push_back(pose);
        m_recordCamera = false;
    }

}
