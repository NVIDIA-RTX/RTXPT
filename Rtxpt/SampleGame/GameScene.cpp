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

#include <donut/core/log.h>
#include <donut/core/json.h>
#include <donut/core/math/math.h>
#include <donut/app/Camera.h>
#include <cmath>

#include "../ExtendedScene.h"

#include "../Sample.h"

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


GameScene::GameScene(Sample & sample)
    : m_sample(sample) // NOTE: at this point, Sample is being constructed - beware of accessing incompletely constructed object
{
}

GLFWwindow* GameScene::GetGLFWWindow() const
{ 
    return m_sample.GetGLFWWindow(); 
}

void GameScene::Deinitialize()
{
    m_scene = nullptr;
    m_props.clear();
    m_modelTypes.clear();
    m_gameStoragePath = std::filesystem::path();
    m_gameTime = 0.0;
    m_timeLoopEnable = false;
    m_timeLoopFrom = 0.0f;
    m_timeLoopTo = 0.0f;
    m_lastTickGlobalAnimationEnabled = false;
    m_camRecEnabled = false;
    m_recordedCameraPoses.clear();
    m_wasGameCameraActive = false;
    m_selectedProp.reset();
    m_playSpeed = 3;
}

void GameScene::ResetGame()
{
    m_camRecEnabled = false;
    for( auto & prop : m_props )
        prop->Reset();
}

// void GameScene::SetActive(bool active)
// {
//     if (m_active == active)
//         return;
// 
//     m_active = active;
//     if (m_active == false)
//     {
//         m_gameTime = 0.0;
//     }
// }

std::shared_ptr<game::ModelType> GameScene::FindModelType(const std::string& modelTypeName)
{
    if (modelTypeName == "")
        return nullptr;
    auto it = std::find_if(m_modelTypes.begin(), m_modelTypes.end(), [ &modelTypeName ](const std::shared_ptr<game::ModelType>& pt) { return pt->GetModelName() == modelTypeName; });
    if (it == m_modelTypes.end())
        return nullptr;
    return *it;
}

std::shared_ptr<game::PropBase> GameScene::CreatePropFromFile(const std::string& name, const std::filesystem::path& storagePath, const Json::Value& jsonRoot)
{
    std::string propType;
    jsonRoot["propType"] >> propType;

    std::shared_ptr<game::PropBase> prop = nullptr;
    if (propType == "SimpleProp")
        prop = std::make_shared<game::SimpleProp>(*this, name);
    if (prop == nullptr)
        { assert( false ); return nullptr; }
    prop->SetStoragePath(storagePath);
    prop->Load(jsonRoot);
    prop->PostLoadSetup();
    prop->Reset();
    return prop;
}

void GameScene::SceneLoaded(const std::shared_ptr<ExtendedScene>& scene, const std::filesystem::path& sceneFilePath, const std::filesystem::path & mediaPath)
{
    Deinitialize();

    auto gameSettings = scene->GetGameSettingsNode();
    if (gameSettings == nullptr)
        return;

    std::filesystem::path mediaGamePath = mediaPath / std::string(c_SampleGameSubFolder);
    if (!EnsureDirectoryExists(mediaGamePath))
        { assert(false); return; }

    std::filesystem::path sceneName = sceneFilePath.filename().stem();
    m_gameStoragePath = mediaGamePath / sceneName;
    if (!EnsureDirectoryExists(m_gameStoragePath))
        { assert(false); return; }

    Json::Value node;
    bool parsingSuccessful = LoadJsonFromString(gameSettings->GetJsonData(), node);
    if (!parsingSuccessful) 
    {
        log::warning( "Unable to load game settings" );
        assert( false );
        return;
    }

    m_scene = scene;

    auto modelFiles = EnumerateFilesWithWildcard(m_gameStoragePath / "models", "*.model.json");

    for (auto modelPath : modelFiles)
    {
        auto fileNoExt = modelPath.filename();
        fileNoExt.replace_extension();
        fileNoExt.replace_extension();

        Json::Value modelRoot;
        if (!LoadJsonFromFile(modelPath, modelRoot) || modelRoot.empty() || !modelRoot.isObject())
            continue;
        m_modelTypes.push_back( std::make_shared<game::ModelType>(*this, fileNoExt.string(), modelRoot) );
    }

    auto propFiles = EnumerateFilesWithWildcard(m_gameStoragePath / "props", "*.prop.json");

    for( auto propPath : propFiles )
    {
        auto fileNoExt = propPath.filename();
        fileNoExt.replace_extension();
        fileNoExt.replace_extension();

        Json::Value propRoot;
        if (!LoadJsonFromFile(propPath, propRoot) || propRoot.empty() || !propRoot.isObject() )
            continue;
        
        std::shared_ptr<game::PropBase> newProp = CreatePropFromFile(fileNoExt.string(), propPath, propRoot);
        if (newProp != nullptr)
            m_props.push_back( newProp );
    }

    // std::srand(0);
    // for (int i = 0; i < (int)m_modelInstances.size(); i++)
    // {
    //     ModelInstance& vehicle = *m_modelInstances[i];
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

void GameScene::SceneUnloading()
{
    Deinitialize();
}

static float GetPlaySpeedK(int playSpeed)
{
    float playSpeedK = 0.0f;
    switch (playSpeed) {
    case(1): playSpeedK = 0.1f; break;
    case(2): playSpeedK = 0.5f; break;
    case(3): playSpeedK = 1.0f; break;
    case(4): playSpeedK = 2.0f; break;
    case(5): playSpeedK =10.0f; break;
    }
    return playSpeedK;
}

void GameScene::AttachCamera(const std::shared_ptr<game::PropBase> & prop)
{
    if (prop == nullptr)
        m_gameCameraAttached.reset();
    else
    {
        auto [pos, dir, up] = prop->GetDefaultCameraPose().GetPosDirUp();
        m_gameCamera.LookTo(pos, dir, up);
        m_gameCameraAttached = prop;
    }
}

bool GameScene::DebugGUI(float indent)
{
    if (!m_lastTickGlobalAnimationEnabled)
        ImGui::Text("Note: global animations disabled, game world not updating!");

    {
        UI_SCOPED_DISABLE(!m_lastTickGlobalAnimationEnabled);

        float playSpeedK = GetPlaySpeedK(m_playSpeed);
        if (ImGui::Button("[-2s]"))
            m_gameTime = std::max(0.0, m_gameTime-2.0);
        ImGui::SameLine();
        if (ImGui::Button("<slower<"))
            m_playSpeed--;
        ImGui::SameLine();
        if (m_playSpeed == 0 && ImGui::Button("[PLAY]"))
            m_playSpeed = 3;
        else if (m_playSpeed != 0 && ImGui::Button("[PAUSE]"))
            m_playSpeed = 0;
        ImGui::SameLine();
        if (ImGui::Button(">faster>"))
            m_playSpeed++;
        ImGui::SameLine();
        if (ImGui::Button("[+2s]"))
            m_gameTime = m_gameTime + 2.0;

        ImGui::Text("Time %05.2f, play speed %.2fx", m_gameTime, playSpeedK);
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
            m_gameTime = 0.0f;

        ImGui::Checkbox("Loop", &m_timeLoopEnable);
        if (m_timeLoopEnable)
        {
            ImGui::SameLine();
            float2 fromTo(m_timeLoopFrom, m_timeLoopTo); ImGui::InputFloat2("from<->to", &fromTo.x, "%.2f"); m_timeLoopFrom = fromTo.x; m_timeLoopTo = fromTo.y;
        }
    }

    {
        auto cameraAttached = m_gameCameraAttached.lock();
        if (cameraAttached == nullptr)
            ImGui::Text("Game camera not active");
        else
        {
            ImGui::Text("Game camera attached to %s", cameraAttached->GetName().c_str());
            if (ImGui::Button("Detach"))
                m_gameCameraAttached.reset();
        }
    }

    if (ImGui::CollapsingHeader("Props", ImGuiTreeNodeFlags_DefaultOpen))
    {
        RAII_SCOPE(ImGui::Indent(indent); , ImGui::Unindent(indent); );

        // ImGui::Text("Props:");
        // ImGui::Separator();

        int itemDisplaySize = 6;
        ImGui::BeginChild("ItemList", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * itemDisplaySize), ImGuiChildFlags_None, ImGuiWindowFlags_AlwaysVerticalScrollbar);

        for (int i = 0; i < (int)m_props.size(); i++)
        {
            const std::shared_ptr<game::PropBase> & prop = m_props[i];

            bool selected = m_selectedProp.lock() == prop;
            if (ImGui::Selectable(prop->GetName().c_str(), &selected, ImGuiSelectableFlags_None))
                m_selectedProp = (selected)?(prop):(nullptr);
        }
        ImGui::EndChild();

        ImGui::Separator();
        //ImGui::Text("Selected:");
        ImGui::BeginChild("Properties", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * itemDisplaySize), ImGuiChildFlags_None, ImGuiWindowFlags_AlwaysVerticalScrollbar);
        
        std::shared_ptr<game::PropBase> selectedProp = m_selectedProp.lock();
        if (selectedProp)
        {
            bool cameraAttached = m_gameCameraAttached.lock() == selectedProp;
            selectedProp->GUI(indent, cameraAttached, m_gameCamera);
            if (cameraAttached && m_gameCameraAttached.lock() != selectedProp)
                AttachCamera(selectedProp);
            if (!cameraAttached && m_gameCameraAttached.lock() == selectedProp)
                AttachCamera(nullptr); 
        }
        else
        {
            ImGui::Text("No selected prop");
        }

        ImGui::EndChild();
    }

#ifdef SAMPLE_GAME_DEVELOPER_SETTINGS
    ImGui::Separator();

    if(!m_camRecEnabled)
    {
        if (ImGui::Button("Start camera rec (1 sec delay)"))
        {
            m_camRecEnabled = true;
            m_camRecTimeToNextKeyframe = 1;
            m_gameTime = -1.0f;
        }
    }
    if (m_camRecEnabled)
    {
        if (ImGui::Button("Stop recording"))
            m_camRecEnabled = false;
    }
    ImGui::Text("Recorded poses: %d", m_recordedCameraPoses.size());
    if (!m_camRecEnabled && m_recordedCameraPoses.size() > 0)
    {
        if (ImGui::Button("Copy first to last for looping"))
        {
            game::Pose term = m_recordedCameraPoses[0];
            term.KeyTime = m_recordedCameraPoses.back().KeyTime+m_camRecKeyframeStep;
            m_recordedCameraPoses.push_back(term);
        }

        if (ImGui::Button("SAVE REC TO EXPORT_POSES.json"))
        {
            Json::Value animsJ;
            for (auto& pose : m_recordedCameraPoses)
                animsJ.append(pose.Write());
            Json::Value rootJ;
            rootJ["animation"] = animsJ;

            SaveJsonToFile(m_gameStoragePath / "EXPORT_POSES.json", rootJ);
            m_recordedCameraPoses.clear();
        }
    }
#endif

    return false;
}

bool GameScene::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    if (CameraActive())
        m_gameCamera.KeyboardUpdate(key, scancode, action, mods);

    //if (key == GLFW_KEY_SPACE && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    //    m_recordCamera = true;

    if (!IsActive())
        return false;

    return false;
}
void GameScene::MousePosUpdate(double xpos, double ypos)
{
    if (CameraActive())
        m_gameCamera.MousePosUpdate(xpos, ypos);
}
void GameScene::MouseButtonUpdate(int button, int action, int mods)
{
    if (CameraActive())
        m_gameCamera.MouseButtonUpdate(button, action, mods);
}

void GameScene::Tick(float deltaTime, bool globalAnimationEnabled)
{
    deltaTime = min(deltaTime, 0.5f);

    if (m_timeLoopEnable && m_gameTime < m_timeLoopFrom)
        m_gameTime = m_timeLoopFrom;

    m_playSpeed = dm::clamp(m_playSpeed, 0, 5);
    float playSpeedK = GetPlaySpeedK(m_playSpeed);

    m_lastTickGlobalAnimationEnabled = globalAnimationEnabled;
    if (IsActive() && globalAnimationEnabled)
    {
        m_gameTime += deltaTime * playSpeedK;

        if (m_timeLoopEnable)
        {
            double loopSpan = (double)m_timeLoopTo-(double)m_timeLoopFrom;
            if (loopSpan > 0 )
            {
                double p = (m_gameTime-m_timeLoopFrom)/loopSpan;
                m_gameTime = (p-floor(p))*loopSpan+m_timeLoopFrom;
            }
            else
                m_gameTime = m_timeLoopFrom;
        }

    }

    for (auto& prop : m_props)
        prop->Tick(m_gameTime, deltaTime);
}

void GameScene::TickCamera(float deltaTime, donut::app::FirstPersonCamera & renderCamera)
{
    // in case we're switching from scene camera (renderCamera) to game camera and back, save/restore scene camera
    if (!m_wasGameCameraActive && CameraActive())
    {
        m_sceneCameraLastPos = renderCamera.GetPosition();
        m_sceneCameraLastDir = renderCamera.GetDir();
        m_sceneCameraLastUp  = renderCamera.GetUp();
    }
    if (m_wasGameCameraActive && !CameraActive())
    {
        renderCamera.LookTo(m_sceneCameraLastPos, m_sceneCameraLastDir, m_sceneCameraLastUp);
    }
    m_wasGameCameraActive = CameraActive();

    if (CameraActive())
    {
        // Allow game camera to move in its own reference frame - this should be optional as some props might like to have control of it
        m_gameCamera.Animate(deltaTime);

        // Move game camera into it's parent (attached) prop's reference frame and apply to global renderCamera
        {
            auto attachedProp = m_gameCameraAttached.lock();
            affine3 transform = affine3::identity();
            if (attachedProp != nullptr)
            {
                // transform = attachedObject->GetRootNode()->GetLocalToWorldTransformFloat(); // we can't do this because these have not yet been updated
                dm::daffine3 transformD = dm::scaling(attachedProp->GetNode()->GetScaling());
                transformD *= attachedProp->GetNode()->GetRotation().toAffine();
                transformD *= dm::translation(attachedProp->GetNode()->GetTranslation());
                transform = dm::affine3(transformD);
            }

            renderCamera.LookTo( transform.transformPoint(m_gameCamera.GetPosition()), transform.transformVector(m_gameCamera.GetDir()), transform.transformVector(m_gameCamera.GetUp()));
        }
    }

    m_lastRenderCameraPose.SetTransformFromCamera(renderCamera.GetPosition(), renderCamera.GetDir(), renderCamera.GetUp());

    //if (!m_active)
    //    return;
    
    if (m_camRecEnabled && IsActive())
    {
        m_camRecTimeToNextKeyframe = dm::clamp(m_camRecTimeToNextKeyframe-deltaTime, -m_camRecKeyframeStep, m_camRecKeyframeStep);

        if (m_camRecTimeToNextKeyframe <= 0)
        {
            game::Pose pose;
            pose.SetTransformFromCamera(renderCamera.GetPosition(), renderCamera.GetDir(), renderCamera.GetUp());
            pose.Scaling = { 1,1,1 };
            pose.KeyTime = m_gameTime;

            m_recordedCameraPoses.push_back(pose);

            m_camRecTimeToNextKeyframe += m_camRecKeyframeStep;
        }
    }
}

void GameScene::StandaloneGUI(const std::shared_ptr<donut::engine::PlanarView> & view, const float2 & displaySize)
{
    // collect toggles
    struct BigButton
    {
        std::string                 Name;
        std::optional<std::string>  HoverText;

        std::function<bool(void)>   IsSelected;
        std::function<void(void)>   OnClick;

        bool                        Enabled;

        BigButton(const std::string& name, const std::string& hoverText, std::function<void(void)> onClick, std::function<bool(void)> isSelected) : Name(name), HoverText(hoverText), OnClick(onClick), IsSelected(isSelected), Enabled(true) { }
        std::string                 GetText() const { return Name; }

    };
    std::vector<BigButton> buttons;

    auto gameCameraAttached = m_gameCameraAttached.lock();
    if (gameCameraAttached!=nullptr)
        buttons.push_back( BigButton("Exit prop camera", StringFormat( "Camera attached to %s", gameCameraAttached->GetName().c_str() ), [&]() { AttachCamera(nullptr); }, [ ]() { return true; } ) );

    if (buttons.size() > 0)
    {
        auto& io = ImGui::GetIO();
        float scaledWidth = io.DisplaySize.x;
        float scaledHeight = io.DisplaySize.y;

        // show & 
        ImVec2 texSizeA = ImGui::CalcTextSize("A");
        float buttonWidth = texSizeA.x * 16;
        float windowHeight = texSizeA.y * 3.0f;
        float windowWidth = buttonWidth * buttons.size() + ImGui::GetStyle().ItemSpacing.x * (buttons.size() + 1);
        ImGui::SetNextWindowPos(ImVec2(0.5f * (scaledWidth - windowWidth), scaledHeight - 10.0f - windowHeight ), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(windowWidth, windowHeight), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.0f);
        if (ImGui::Begin("GameUI", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoNav))
        {
            for (int i = 0; i < buttons.size(); i++)
            {
                if (i > 0)
                    ImGui::SameLine();

                UI_SCOPED_DISABLE(!buttons[i].Enabled);

                bool selected = buttons[i].IsSelected();

                ImGui::PushID(i);
                float h = 0.33f;
                float b = selected ? 1.0f : 0.1f;
                ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, 0.6f * b, 0.6f * b));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, 0.7f * b, 0.7f * b));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(h, 0.8f * b, 0.8f * b));
                if (ImGui::Button(buttons[i].GetText().c_str(), ImVec2(buttonWidth, texSizeA.y * 2)))
                {
                    buttons[i].OnClick();
                }
                ImGui::PopStyleColor(3);
                ImGui::PopID();

                if (buttons[i].HoverText.has_value())
                {
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip(buttons[i].HoverText.value().c_str());
                }
            }
        }
        ImGui::End();
    }

    auto currentlyAttachedProp = m_gameCameraAttached.lock();
    game::ScreenGUISel selArea{}; std::shared_ptr<game::PropBase> selProp;
    float2 mousePos = { ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y };
    for (auto& prop : m_props)
    {
        if (prop == currentlyAttachedProp)
            continue;

        game::ScreenGUISel selC = prop->StandaloneGUI(view, mousePos, displaySize);
        if (selC.Selected && selC.RangeToCamera < selArea.RangeToCamera)
        {
            selArea = selC;
            selProp = prop;
        }
    }
    
    if (selArea.Selected && selProp != nullptr)
    {
        ImDrawList* draw_list = ImGui::GetForegroundDrawList();
        draw_list->AddCircle(ImVec2(selArea.ScreenPos.x, selArea.ScreenPos.y), selArea.ScreenRadius, IM_COL32(0, 0, 255, 255), 32);
        std::string info = StringFormat("Press 'F' to lock camera to prop '%s'", selProp->GetName().c_str());
        draw_list->AddText(ImVec2(selArea.ScreenPos.x+1, selArea.ScreenPos.y+1), IM_COL32(0, 0, 0, 192), info.c_str() );
        draw_list->AddText(ImVec2(selArea.ScreenPos.x, selArea.ScreenPos.y), IM_COL32(255, 255, 255, 255), info.c_str());

        if (ImGui::IsKeyDown(ImGuiKey::ImGuiKey_F))
            AttachCamera(selProp);
    }
      
}

