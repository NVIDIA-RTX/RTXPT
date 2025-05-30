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

struct SampleGamePose
{
    double3     Translation     = { 0, 0, 0 };
    dquat       Rotation        = { 0, 0, 0, 1 };
    double3     Scaling         = { 1, 1, 1 };
    double      KeyTime         = 0.0;

    bool Read(const Json::Value& node);
    Json::Value Write();
};

struct SampleGameKeyframeAnimation
{
    std::vector<SampleGamePose>     Keys;
    double                          KeyTimeMin;
    double                          KeyTimeMax;

    bool Read(const Json::Value& node);

    bool GetAt(double time, bool wrap, SampleGamePose & outPose);

private:
    int                             LastFound = -1;
};

namespace donut::app
{
    class FirstPersonCamera;
}

class SampleGamePropInstance;

class SampleGamePropType
{
public:
    SampleGamePropType(class SampleGame & game, const Json::Value & node);

    bool                                                    IsValid() const         { return m_valid; }
    const std::shared_ptr<donut::engine::SceneGraphNode> &  GetModelNode() const    { return m_modelNode; }
    const std::string &                                     GetTypeName() const     { return m_typeName; }
    class SampleGame &                                      GetGame() const         { return m_game; }

    static std::shared_ptr<SampleGamePropInstance> CreateInstance(const std::shared_ptr<SampleGamePropType> & propType, const std::string & name, const std::filesystem::path & storagePath, const Json::Value & jsonRoot);

private:
    class SampleGame &  m_game;
    bool                m_valid = false;
    std::string         m_typeName;
    int                 m_modelIndex = -1;
    SampleGamePose      m_modelPose;
    std::shared_ptr<donut::engine::SceneGraphNode> m_modelNode;
};

class SampleGamePropInstance
{
public:
    SampleGamePropInstance( const std::string & name, const std::shared_ptr<SampleGamePropType> & propType, const std::filesystem::path & storagePath );

    // void                    SetTransform(const dm::double3* translation, const dm::dquat* rotation, const dm::double3* scaling);

    void                    Tick(double gameTime, float deltaTime);

    void                    Reset();

    const std::string &     GetName() const { return m_node->GetName(); }

    void                    SetStoragePath(const std::filesystem::path & path) { m_storagePath = path; }
    void                    Load(const Json::Value & jsonRoot);

    void                    SetAnimOffset(double animOffset) { m_animOffset = animOffset; }

    const std::shared_ptr<donut::engine::SceneGraphNode> &
                            GetRootNode() const { return m_node; }

private:
    SampleGamePose                  m_startPose;
    std::shared_ptr<donut::engine::SceneGraphNode> m_node;  // weak_ptr?
    std::filesystem::path           m_storagePath;

    SampleGameKeyframeAnimation     m_recording;
    double                          m_animOffset;
    bool                            m_animating = false;
};
