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
#include "GameProps.h"
#include "GamePropComponent.h"

#include <donut/core/log.h>
#include <donut/core/json.h>
#include <donut/core/math/math.h>
#include <donut/app/Camera.h>
#include <cmath>

#include "../ExtendedScene.h"

#include "../Misc/Korgi.h"
#include <json/json.h>

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

PropBase::PropBase(GameScene& gameScene, const std::string & name)
    : m_gameScene(gameScene) 
{
    const std::shared_ptr<class ExtendedScene>& scene = gameScene.GetScene();
    m_node = std::make_shared<SceneGraphNode>();

    m_node = scene->GetSceneGraph()->Attach(scene->GetSceneGraph()->GetRootNode(), m_node);
    m_node->SetName(name);
}

std::shared_ptr<ModelInstance> PropBase::CreateAndAttachModel(const std::shared_ptr<game::ModelType> & modelType, const std::string & instanceName, const dm::float3& translation, const dm::quat& rotation, const dm::float3& scaling )
{
    auto ret = std::make_shared<game::ModelInstance>(instanceName, modelType, m_node);
    ret->SetTransform(translation, rotation, scaling);
    return ret;
}

void PropBase::SetTransform(const dm::double3& translation, const dm::dquat& rotation, const dm::double3& scaling)
{
    m_node->SetTransform(&translation, &rotation, &scaling);
}

void PropBase::SetTransform(const dm::float3& translation, const dm::quat& rotation, const dm::float3& scaling)
{
    dm::double3 transD = dm::double3(translation);
    dm::dquat rotD = dm::dquat(rotation);
    dm::double3 scalD = dm::double3(scaling);
    m_node->SetTransform(&transD, &rotD, &scalD);
}

template<typename T>
quaternion<T> fixedRotationQuat(const vector<T, 3>& axis, T radians)
{
    // Note: assumes axis is normalized
    T sinHalfTheta = std::sin(T(0.5) * radians);
    T cosHalfTheta = std::cos(T(0.5) * radians);
    const vector<T, 3>& axisLocal = axis * sinHalfTheta;

    return quaternion<T>(cosHalfTheta, axisLocal.x, axisLocal.y, axisLocal.z);
}

void PropBase::Tick(double gameTime, float deltaTime)
{
#if 1
    if (m_allowKeyMoveIfSelected && m_gameScene.GetSelectedProp().get() == this) // yuck?
    {
        float forward = 0.0f;
        float right = 0.0f;
        float up = 0.0f;
        float rotateAroundForward = 0.0;
        float rotateAroundRight = 0.0;
        float rotateAroundUp = 0.0;

        auto IsDown = [&](int key) { return glfwGetKey( m_gameScene.GetGLFWWindow(), key ) == GLFW_PRESS; };

        float moveSpeed = deltaTime;
        if (IsDown(GLFW_KEY_LEFT_CONTROL) || IsDown(GLFW_KEY_RIGHT_CONTROL))
            moveSpeed *= 0.1f;
        if (IsDown(GLFW_KEY_LEFT_SHIFT) || IsDown(GLFW_KEY_RIGHT_SHIFT))
            moveSpeed *= 10.0f;
        float rotateSpeed = moveSpeed;

        bool rotateInsteadOfMove = IsDown(GLFW_KEY_RIGHT_ALT);
        if (!rotateInsteadOfMove)
        {
            if (IsDown(GLFW_KEY_KP_8))  forward += moveSpeed;
            if (IsDown(GLFW_KEY_KP_2))  forward -= moveSpeed;
            if (IsDown(GLFW_KEY_KP_4))  right -= moveSpeed;
            if (IsDown(GLFW_KEY_KP_6))  right += moveSpeed;
            if (IsDown(GLFW_KEY_KP_9))  up += moveSpeed;
            if (IsDown(GLFW_KEY_KP_3))  up -= moveSpeed;
        }
        else
        {
            if (IsDown(GLFW_KEY_KP_8))  rotateAroundRight += rotateSpeed;
            if (IsDown(GLFW_KEY_KP_2))  rotateAroundRight -= rotateSpeed;
            if (IsDown(GLFW_KEY_KP_4))  rotateAroundUp += rotateSpeed;
            if (IsDown(GLFW_KEY_KP_6))  rotateAroundUp -= rotateSpeed;
            if (IsDown(GLFW_KEY_KP_7))  rotateAroundForward -= rotateSpeed;
            if (IsDown(GLFW_KEY_KP_9))  rotateAroundForward += rotateSpeed;
        }

        if (forward != 0 || right != 0 || up != 0)
        {
            double3 trans = m_node->GetTranslation();
            
#if 1 // use rotation 
            auto rot = quat(m_node->GetRotation()).toAffine();
            float3 forwardVec   = rot.transformVector(m_referenceForward);
            float3 upVec        = rot.transformVector(m_referenceUp);
            float3 rightVec     = rot.transformVector(m_referenceRight);
#else
            float3 forwardVec   = m_referenceForward;
            float3 upVec        = m_referenceUp;
            float3 rightVec     = m_referenceRight;
#endif

            trans += double3( forwardVec * forward + upVec * up + rightVec * right );
            m_node->SetTranslation(trans);
        }
        if (rotateAroundForward != 0 || rotateAroundRight != 0 || rotateAroundUp != 0)
        {
            dquat rot = m_node->GetRotation();
            rot *= dquat( fixedRotationQuat<float>(m_referenceForward, rotateAroundForward) * fixedRotationQuat<float>(m_referenceUp, rotateAroundUp) * fixedRotationQuat<float>(m_referenceRight, rotateAroundRight) );
            m_node->SetRotation(rot);
        }
    }
#endif

    // if (!m_gameScene.IsActive())
    //     return;

    Pose animPose;
    float animTime = 0.0f;
    if (m_animation.GetAt(m_animPlaybackSpeed*gameTime + m_animOffset, true, animPose, animTime))
        SetTransform(animPose.Translation, animPose.Rotation, animPose.Scaling);

    for (auto & comp : m_components)
        comp->Tick(gameTime, animTime, deltaTime);

    // transfer all changes to lights - especially necessary if scaling used
    for (auto& model : m_models)
        model->UpdateLightFromControllers(gameTime);
}

void PropBase::Reset()
{
    SetTransform(m_startPose.Translation, m_startPose.Rotation, m_startPose.Scaling);
    m_animOffset = 0;
}

void PropBase::Load(const Json::Value& jsonRoot)
{
    jsonRoot["propType"] >> m_propType;

    m_startPose.Read(jsonRoot["startPose"]);

    auto jsonDefaultCameraPose = jsonRoot["defaultCameraPose"];
    if (jsonDefaultCameraPose.empty())
        m_defaultCameraPose.SetTransformFromCamera( {0,0,0}, {1, 0, 0}, {0, 1, 0} );
    else
        m_defaultCameraPose.Read(jsonDefaultCameraPose);

    jsonRoot["animPlaybackSpeed"] >> m_animPlaybackSpeed;

    Json::Value animationRecording = jsonRoot["animation"];

    if (!animationRecording.empty() && animationRecording.isArray())
    {
        m_animation.Read(animationRecording);
        m_animating = true;
    }

    Json::Value modelsLightsOverrides = jsonRoot["modelsLightsOverrides"];
    if (!modelsLightsOverrides.empty())
        m_modelsLightsOverrides = SaveJsonToString(modelsLightsOverrides);

    Json::Value components = jsonRoot["components"];
    if (!components.empty())
        m_componentsData = SaveJsonToString(components);
}

void PropBase::PostLoadSetup()
{
    if (m_modelsLightsOverrides != "")
    {
        Json::Value modelsLightsOverrides;
        LoadJsonFromString(m_modelsLightsOverrides, modelsLightsOverrides);
        for( Json::Value & m : modelsLightsOverrides )
        {
            std::string modelName; m["modelInstanceName"] >> modelName;
            
            auto it = std::find_if(m_models.begin(), m_models.end(), [&modelName](auto & model) { return model->GetInstanceName() == modelName; });
            if (it != m_models.end())
            {
                const std::vector<std::shared_ptr<LightController>> & modelLights = (*it)->GetLights();
                Json::Value lightOverrides = m["lightOverrides"];
                for (Json::Value& lo : lightOverrides)
                {
                    std::string lightName; lo["name"] >> lightName;

                    auto lit = std::find_if(modelLights.begin(), modelLights.end(), [ &lightName ](auto& light) { return light->Node->GetName() == lightName; });
                    if (lit != modelLights.end())
                    {
                        (*lit)->Read(lo);
                    }
                    else { donut::log::warning("Bad light override, prop %s, model %s, light name %s", GetName().c_str(), modelName.c_str(), lightName.c_str()); }
                }
            }
            else { donut::log::warning("Bad light override, prop %s, model %s", GetName().c_str(), modelName.c_str()); }
        }
    }
    if (m_componentsData != "")
    {
        Json::Value components;
        LoadJsonFromString(m_componentsData, components);
        for (Json::Value& m : components)
        {
            std::shared_ptr<PropComponentBase> comp = PropComponentBase::Create(*this, m);
            if (comp != nullptr)
                m_components.push_back(comp);
        }
    }
}

Json::Value PropBase::Save()
{
    Json::Value jsonRoot;
    if (m_animation.Keys.size() > 0)
        jsonRoot["animation"] = m_animation.Write();

    jsonRoot["defaultCameraPose"] = m_defaultCameraPose.Write();
    jsonRoot["startPose"] = m_startPose.Write();
    jsonRoot["propType"] = m_propType;
    jsonRoot["animPlaybackSpeed"] = m_animPlaybackSpeed;
    
    if (m_modelsLightsOverrides!="")
    {
        Json::Value outRootNode;
        if (LoadJsonFromString(m_modelsLightsOverrides, outRootNode))
            jsonRoot["modelsLightsOverrides"] = outRootNode;
    }
    if (m_componentsData != "")
    {
        Json::Value outRootNode;
        if (LoadJsonFromString(m_componentsData, outRootNode))
            jsonRoot["components"] = outRootNode;
    }
    return jsonRoot;
}

void PropBase::GUI(float indent, bool & gameCameraAttached, donut::app::FirstPersonCamera & gameCamera)
{
    RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );
    ImGui::Text("Properties for %s", GetName().c_str());

    if (gameCameraAttached)
    {
        RAII_SCOPE(ImGui::PushID("CamPose");, ImGui::PopID(););
        ImGui::Text("Camera ATTACHED!"); ImGui::SameLine(); 
        if (ImGui::Button("Detach")) 
            gameCameraAttached = false;
#ifdef SAMPLE_GAME_DEVELOPER_SETTINGS
        ImGui::Text("Camera pose: "); ImGui::SameLine();
        if (ImGui::Button("Save"))
        {
            m_defaultCameraPose.SetTransformFromCamera(gameCamera.GetPosition(), gameCamera.GetDir(), gameCamera.GetUp());
        }
        ImGui::SameLine();
        if (ImGui::Button("Load"))
        {
            auto [pos, dir, up] = m_defaultCameraPose.GetPosDirUp();
            gameCamera.LookTo( pos, dir, up );
        }
#endif
    }
    else
    {
        ImGui::Text("Camera not attached"); ImGui::SameLine();
        if (ImGui::Button("Attach"))
            gameCameraAttached = true;
#ifdef SAMPLE_GAME_DEVELOPER_SETTINGS
        if (ImGui::Button("Move prop to camera pose"))
        {
            auto& pose = m_gameScene.GetLastRenderCameraPose();
            m_node->SetTransform(&pose.Translation, &pose.Rotation, nullptr/*&pose.Scaling*/);
        }
#endif
    }
#ifdef SAMPLE_GAME_DEVELOPER_SETTINGS
    if (ImGui::Button("Save current as start pose"))
    {
        m_startPose.Translation = m_node->GetTranslation();
        m_startPose.Rotation = m_node->GetRotation();
        m_startPose.Scaling = m_node->GetScaling();
    }
    if (m_gameScene.GetCamRecAnimation().size() > 0 && ImGui::Button("Copy current cam animation"))
    {
        m_animation.Keys.clear();
        m_animation.FromKeys(m_gameScene.GetCamRecAnimation());
    }
    ImGui::Separator();
    {
        RAII_SCOPE(ImGui::PushID("Storage"); , ImGui::PopID(););
        //ImGui::Text("Storage: '%s'", m_storagePath.string().c_str());
        ImGui::Text("Storage: ");
        ImGui::SameLine();
        if (ImGui::Button("Save"))
            SaveJsonToFile(m_storagePath, Save());

        if (ImGui::CollapsingHeader("Lights"))
        {
            int counter = 0;
            for (const auto & model : m_models)
                for (const auto & light : model->GetLights())
                {
                    counter++; RAII_SCOPE(ImGui::PushID(counter);, ImGui::PopID(););
                    ImGui::Checkbox( std::format("Light {} - {}", model->GetInstanceName().c_str(), light->Node->GetName().c_str()).c_str(), &light->Enabled );
                }
        }
        // ImGui::SameLine();
        // if (ImGui::Button("Load"))
        // {
        //     Json::Value jread;
        //     LoadJsonFromFile( m_storagePath, jread );
        //     Load(jread);
        //     Reset();
        // }
    }
#endif
}

ScreenGUISel PropBase::StandaloneGUI(const std::shared_ptr<donut::engine::PlanarView> & view, const float2 & mousePos, const float2 & displaySize)
{
    ScreenGUISel sel{};
    for (auto& comp : m_components)
    {
        ScreenGUISel selC = comp->StandaloneGUI(view, mousePos, displaySize);
        if (selC.Selected && selC.RangeToCamera < sel.RangeToCamera)
            sel = selC;
    }
    return sel;
}

SimpleProp::SimpleProp(class GameScene & gameScene, const std::string & name)
    : PropBase(gameScene, name)
{
}

void SimpleProp::Tick(double gameTime, float deltaTime)
{
    PropBase::Tick(gameTime, deltaTime);
}

void SimpleProp::Reset()
{
    PropBase::Reset();
}

void SimpleProp::Load(const Json::Value& jsonRoot)
{
    PropBase::Load(jsonRoot);

    jsonRoot["modelName"] >> m_modelName;
    auto modelType = m_gameScene.FindModelType(m_modelName);
    if (modelType == nullptr)
        { assert( false ); return; }
    m_model = CreateAndAttachModel(modelType, "SimplePropOnlyModel", float3(0,0,0) );
    m_models.push_back(m_model);
}

Json::Value SimpleProp::Save()
{
    Json::Value ret = PropBase::Save();

    ret["modelName"] << m_modelName;
    return ret;
}
