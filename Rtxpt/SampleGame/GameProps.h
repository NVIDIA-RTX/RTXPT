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

namespace game
{
    class PropComponentBase;

    // base class for other game
    class PropBase
    {
    public:
        PropBase(class GameScene & gameScene, const std::string & name);
        virtual ~PropBase() { }

        virtual void            Tick(double gameTime, float deltaTime);
        virtual void            Reset();
        virtual void            Load(const Json::Value& jsonRoot);
        virtual void            PostLoadSetup();
        virtual Json::Value     Save();

        virtual void            SetTransform(const dm::double3& translation, const dm::dquat& rotation, const dm::double3& scaling);
        virtual void            SetTransform(const dm::float3& translation, const dm::quat& rotation, const dm::float3& scaling);

        const Pose &            GetDefaultCameraPose() const            { return m_defaultCameraPose; }

        void                    SetStoragePath(const std::filesystem::path& path) { m_storagePath = path; }
        void                    SetAnimOffset(double animOffset)        { m_animOffset = animOffset; }
        const std::string &     GetName() const                         { return m_node->GetName(); }
        const std::shared_ptr<donut::engine::SceneGraphNode> & 
                                GetNode() const                         { return m_node; }
        const std::vector<std::shared_ptr<ModelInstance>> &
                                GetModels() const                       { return m_models; }

        virtual void            GUI(float indent, bool & gameCameraAttached, donut::app::FirstPersonCamera & gameCamera);
        virtual ScreenGUISel    StandaloneGUI(const std::shared_ptr<donut::engine::PlanarView> & view, const float2 & mousePos, const float2 & displaySize);

    protected:
        std::shared_ptr<ModelInstance> CreateAndAttachModel( const std::shared_ptr<game::ModelType> & modelType, const std::string & instanceName, const dm::float3& translation, const dm::quat& rotation = dm::quat::identity(), const dm::float3& scaling = dm::float3(1,1,1) );
        
    protected:
        class GameScene & m_gameScene;
        std::string                 m_propType;

        Pose                        m_startPose;
        
        Pose                        m_defaultCameraPose;
        // float3                      m_defaultCameraPos  = 0.0f;
        // float3                      m_defaultCameraDir  = dm::float3(1.f, 0.f, 0.f);
        // float3                      m_defaultCameraUp   = dm::float3(0.f, 1.f, 0.f);

        float3                      m_referenceForward  = dm::float3(1.f, 0.f, 0.f);
        float3                      m_referenceUp       = dm::float3(0.f, 1.f, 0.f);
        float3                      m_referenceRight    = dm::float3(0.f, 0.f, 1.f);

        std::filesystem::path       m_storagePath;

        KeyframeAnimation           m_animation;
        float                       m_animPlaybackSpeed = 1.0f;
        double                      m_animOffset    = 0.0;
        bool                        m_animating     = false;

        bool                        m_allowKeyMoveIfSelected = true;

        std::shared_ptr<donut::engine::SceneGraphNode> m_node;

        std::vector<std::shared_ptr<ModelInstance>>
                                    m_models;
        std::vector<std::shared_ptr<PropComponentBase>>
                                    m_components;

        std::string                 m_modelsLightsOverrides;

        std::string                 m_componentsData;
    };

    // Single model prop
    class SimpleProp : public PropBase
    {
    public:
        SimpleProp(class GameScene & gameScene, const std::string & name);
        virtual ~SimpleProp() {}

    protected:
        virtual void            Tick(double gameTime, float deltaTime) override;
        virtual void            Reset() override;
        virtual void            Load(const Json::Value& jsonRoot) override;
        virtual Json::Value     Save() override;

    protected:
        std::string             m_modelName;
        std::shared_ptr<ModelInstance>
                                m_model;
    };
}