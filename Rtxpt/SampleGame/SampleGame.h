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

#include "SampleGameProp.h"

class SampleGame
{
public:
    SampleGame(class Sample & sample);

    void                    SceneLoaded( const std::shared_ptr<class ExtendedScene> & scene, const std::filesystem::path& sceneFilePath, const std::filesystem::path & mediaPath );
    void                    SceneUnloading( );
    bool                    DebugGUI(float indent);

    bool                    IsEnabled() const { return m_scene != nullptr; }
    bool                    HasCamera() const { return m_gameCameraEnabled; }
    const donut::app::FirstPersonCamera &
                            Camera() const      { return m_gameCamera; }

    void                    SetActive(bool active);

    bool                    KeyboardUpdate(int key, int scancode, int action, int mods);
    void                    MousePosUpdate(double xpos, double ypos);
    void                    MouseButtonUpdate(int button, int action, int mods);
    void                    Tick(float deltaTime);
    void                    TickCamera(float deltaTime, donut::app::FirstPersonCamera & renderCamera);

    const std::shared_ptr<ExtendedScene> &
                            GetScene() const { return m_scene; }

private:
    void                    LoadPropTypes(const Json::Value & propTypeArray);
    void                    Deinitialize( );
    void                    ResetGame( );

private:
    class Sample &          m_sample;
    std::shared_ptr<class ExtendedScene>
                            m_scene = nullptr;
    bool                    m_active = false;

    std::vector<std::shared_ptr<SampleGamePropType>> m_propTypes;

    std::vector<std::shared_ptr<SampleGamePropInstance>>
                            m_propInstances;

    double                  m_gameTime = 0.0;

    // std::shared_ptr<donut::engine::SceneGraphNode> m_cameraNode;
    // std::shared_ptr<class PerspectiveCameraEx> m_camera;

    std::filesystem::path   m_gameStoragePath;

    bool                    m_recordCamera = false;
    std::vector<SampleGamePose> m_recordedCameraPoses;

    donut::app::FirstPersonCamera   m_gameCamera;
    bool                            m_gameCameraEnabled      = false;
    bool                            m_gameCameraInitialInit  = false;
    std::weak_ptr<SampleGamePropInstance> m_gameCameraAttached;
};

