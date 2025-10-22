/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "../Shaders/PathTracer/Config.h"
#include "../SampleCommon.h"
#include "../SampleUI.h"

// #include <donut/core/vfs/VFS.h>
#include <donut/app/Camera.h>

#include "GameModel.h"
#include "GameProps.h"

#ifdef _DEBUG
#define SAMPLE_GAME_DEVELOPER_SETTINGS
#endif

// this is kind of a parallel item to ExtendedScene, but built on top - perhaps a better name is needed; GameStage? GameLevel?
class GameScene
{
public:
    GameScene(class Sample & sample);

    void                    SceneLoaded( const std::shared_ptr<class ExtendedScene> & scene, const std::filesystem::path& sceneFilePath, const std::filesystem::path & mediaPath );
    void                    SceneUnloading( );
    bool                    DebugGUI(float indent);
    void                    StandaloneGUI(const std::shared_ptr<donut::engine::PlanarView> & view, const float2 & displaySize);

    bool                    IsInitialized() const           { return m_scene != nullptr; }
    bool                    CameraActive() const            { return m_gameCameraAttached.lock() != nullptr; }
    const donut::app::FirstPersonCamera &
                            GetCamera() const               { return m_gameCamera; }
    std::shared_ptr<game::PropBase>
                            GetCameraAttached() const       { return m_gameCameraAttached.lock(); }
    void                    AttachCamera(const std::shared_ptr<game::PropBase> & prop);
    const game::Pose &      GetLastRenderCameraPose() const { return m_lastRenderCameraPose; }

    // active means animating / physics is enabled
    //void                    SetActive(bool active);
    bool                    IsActive()                      { return m_playSpeed != 0; }

    bool                    KeyboardUpdate(int key, int scancode, int action, int mods);
    void                    MousePosUpdate(double xpos, double ypos);
    void                    MouseButtonUpdate(int button, int action, int mods);
    void                    Tick(float deltaTime, bool globalAnimationEnabled); // globalAnimationEnabled will be false if not in reference mode or global scene animations not enabled
    void                    TickCamera(float deltaTime, donut::app::FirstPersonCamera & renderCamera);

    const std::shared_ptr<ExtendedScene> &
                            GetScene() const { return m_scene; }

    double                  GetGameTime() const             { return m_gameTime; }
    void                    SetGameTime(double t)           { m_gameTime = t; }

    std::shared_ptr<game::ModelType>
                            FindModelType(const std::string & name);

    std::shared_ptr<game::PropBase>
                            GetSelectedProp() const { return m_selectedProp.lock(); }

    GLFWwindow *            GetGLFWWindow() const;

    const std::vector<game::Pose> & GetCamRecAnimation() const { return m_recordedCameraPoses; }


private:
    std::shared_ptr<game::PropBase> CreatePropFromFile(const std::string& name, const std::filesystem::path& storagePath, const Json::Value& jsonRoot);
    void                    Deinitialize( );
    void                    ResetGame( );

private:
    class Sample &          m_sample;
    std::shared_ptr<class ExtendedScene>
                            m_scene = nullptr;
    int                     m_playSpeed = 3;   // speed: 0 - paused, 1 - 0.1x, 2 - 0.5x, 3 - 1.0x, 4 - 2.0x, 5 - 10.0x

    std::vector<std::shared_ptr<game::ModelType>> m_modelTypes;

    std::vector<std::shared_ptr<game::PropBase>>
                            m_props;
    std::weak_ptr<game::PropBase>
                            m_selectedProp;

    double                  m_gameTime = 0.0;

    bool                    m_timeLoopEnable    = false;
    float                   m_timeLoopFrom      = 0.0f;
    float                   m_timeLoopTo        = 0.0f;

    bool                    m_lastTickGlobalAnimationEnabled = false;

    // std::shared_ptr<donut::engine::SceneGraphNode> m_cameraNode;
    // std::shared_ptr<class PerspectiveCameraEx> m_camera;

    std::filesystem::path   m_gameStoragePath;

    bool                    m_camRecEnabled = false;
    float                   m_camRecKeyframeStep = 1.0f;
    float                   m_camRecTimeToNextKeyframe = 0.0f;
    std::vector<game::Pose> m_recordedCameraPoses;
    game::Pose              m_lastRenderCameraPose; // useful for stuff like "set prop to camera pose"

    donut::app::FirstPersonCamera   m_gameCamera;
    std::weak_ptr<game::PropBase>   m_gameCameraAttached;

    bool                            m_wasGameCameraActive = false;
    float3                          m_sceneCameraLastPos;
    float3                          m_sceneCameraLastDir;
    float3                          m_sceneCameraLastUp;
};

