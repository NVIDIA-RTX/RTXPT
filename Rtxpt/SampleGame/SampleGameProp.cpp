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
#include "SampleGameProp.h"

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

bool SampleGamePose::Read( const Json::Value & node )
{
    if (node.isNull())
        return false;

    const auto& translation = node["translation"];
    if (!translation.isNull())
        translation >> Translation;

    const auto& rotation = node["rotation"];
    if (!rotation.isNull())
    {
        double4 value = double4(0.0, 0.0, 0.0, 1.0);
        rotation >> value;
        Rotation = dm::dquat::fromXYZW(value);
    }
    else
    {
        const auto& euler = node["euler"];
        if (!euler.isNull())
        {
            double3 value = double3::zero();
            euler >> value;
            Rotation = rotationQuat(value);
        }
    }

    const auto& scaling = node["scaling"];
    if (!scaling.isNull())
        scaling >> Scaling;

    const auto& keytime = node["keytime"];
    if (!keytime.isNull())
        keytime >> KeyTime;
        
    return true;
}

Json::Value SampleGamePose::Write()
{
    Json::Value ret;
    ret["translation"] << Translation;
    ret["rotation"] << double4(Rotation.x, Rotation.y, Rotation.z, Rotation.w);
    ret["scaling"] << Scaling;
    ret["keytime"] << KeyTime;
    return ret;
}

bool SampleGameKeyframeAnimation::Read(const Json::Value& node)
{
    assert(node.isArray());
    KeyTimeMin = FLT_MAX;
    KeyTimeMax = -FLT_MAX;
    for( int i = 0; i < node.size(); i++)
    {
        SampleGamePose key;
        key.Read(node[i]);
        Keys.push_back(key);
        KeyTimeMin = min( KeyTimeMin, key.KeyTime );
        KeyTimeMax = max( KeyTimeMax, key.KeyTime );
    }
    std::sort(Keys.begin(), Keys.end(), [ ](const SampleGamePose & a, const SampleGamePose & b) { return a.KeyTime < b.KeyTime; });
    return true;
}

static double3 my_lerp(double3 a, double3 b, double u) { return a + (b - a) * u; }

static SampleGamePose lerp(const SampleGamePose & a, const SampleGamePose & b, double k)
{
    SampleGamePose ret;
    ret.KeyTime = lerp( a.KeyTime, b.KeyTime, k );
    ret.Translation = my_lerp( a.Translation, b.Translation, k );
    ret.Rotation = slerp<double>(a.Rotation, b.Rotation, k);
    ret.Scaling = my_lerp( a.Scaling, b.Scaling, k );
    return ret;
}

bool SampleGameKeyframeAnimation::GetAt(double time, bool wrap, SampleGamePose& outPose)
{
    if (Keys.size()==0)
        return false;

    if (!wrap)
        time = clamp( time, KeyTimeMin, KeyTimeMax );
    else
    {
        time = fmod( abs(time-KeyTimeMin+KeyTimeMax-KeyTimeMin), KeyTimeMax-KeyTimeMin );
        time += KeyTimeMin;
    }

    SampleGamePose a = Keys[0], b = Keys.back();
    if (LastFound>0 && Keys[LastFound-1].KeyTime <= time && Keys[LastFound].KeyTime >= time ) 
    {
        a = Keys[LastFound - 1];
        b = Keys[LastFound];
    }
    else
    {
        int start = 1; 
        if (LastFound>0 && Keys[LastFound-1].KeyTime <= time && Keys[LastFound].KeyTime <= time )
            start = LastFound;
        LastFound = start-1;

        for( int i = start; i < Keys.size(); i++ ) // linear search, awful heh
        {
            if (Keys[i].KeyTime>time)
            {
                a = Keys[i-1];
                b = Keys[i];
                LastFound = i;
                break;
            }
        }
    }
    double span = b.KeyTime - a.KeyTime + 1e-15; assert( span > 0 );
    double off = time - a.KeyTime; assert( off > 0 );
    double lerpK = saturate(off / span);

    outPose = lerp(a, b, lerpK);
    return true;
}

void operator >> (class Json::Value const& h, struct SampleGamePose& p) 
{ 
    p.Read(h); 
}

void FixupSpotlightsRec(SceneGraphNode* node, float avgScale)
{
    SpotLight* spotLight = dynamic_cast<SpotLight*>(node->GetLeaf().get());

    if (spotLight != nullptr)
    {
        //spotLight->range = 10000.0f;
        //spotLight->intensity = 250.0f;
        spotLight->innerAngle;
        spotLight->outerAngle;
        spotLight->radius = 0.1f * avgScale;    // there is no radius in gltf lights as of yet (see KHR_lights_punctual)
    }

    for (int i = 0; i < node->GetNumChildren(); i++)
        FixupSpotlightsRec(node->GetChild(i), avgScale);
}

SampleGamePropType::SampleGamePropType(class SampleGame & game, const Json::Value & node)
    : m_game(game)
{
    m_valid = false;
    const std::shared_ptr<ExtendedScene> & scene = game.GetScene();

    node["typeName"] >> m_typeName;

    node["modelIndex"] >> m_modelIndex;

    node["modelPose"] >> m_modelPose;

    const std::vector<donut::engine::SceneImportResult>& models = scene->GetModels();
    if (m_modelIndex<0 || m_modelIndex>=models.size())
    {
        log::warning("Referenced model %d is not defined in the model array.", m_modelIndex);
        return; 
    }

    const auto& loadedModel = models[m_modelIndex];
    if (!loadedModel.rootNode)
    { assert( false ); return; }

    FixupSpotlightsRec(loadedModel.rootNode.get(), float((m_modelPose.Scaling.x+m_modelPose.Scaling.y+m_modelPose.Scaling.z)/3.0) );
    loadedModel.rootNode->SetTransform(&m_modelPose.Translation, &m_modelPose.Rotation, &m_modelPose.Scaling);

    m_modelNode = loadedModel.rootNode;

    m_valid = m_typeName != "" && m_modelIndex != -1;
}

std::shared_ptr<SampleGamePropInstance> SampleGamePropType::CreateInstance(const std::shared_ptr<SampleGamePropType>& propType, const std::string& name, const std::filesystem::path & storagePath, const Json::Value & jsonRoot)
{
    std::shared_ptr<SampleGamePropInstance> ret = std::make_shared<SampleGamePropInstance>(name, propType, storagePath);
    ret->Load(jsonRoot);
    ret->Reset();
    return ret;
}

SampleGamePropInstance::SampleGamePropInstance( const std::string & name, const std::shared_ptr<SampleGamePropType> & propType, const std::filesystem::path & storagePath )
    : m_storagePath(storagePath)
{
    const std::shared_ptr<class ExtendedScene> & scene = propType->GetGame().GetScene();
    m_node = std::make_shared<SceneGraphNode>();

    m_node = scene->GetSceneGraph()->Attach(scene->GetSceneGraph()->GetRootNode(), m_node);
    m_node->SetName(name);

    scene->GetSceneGraph()->Attach(m_node, propType->GetModelNode());
}

void SampleGamePropInstance::Load(const Json::Value& jsonRoot)
{
    jsonRoot["startPose"] >> m_startPose;

    Json::Value animationRecording = jsonRoot["animation"];

    if (!animationRecording.empty() && animationRecording.isArray())
    {
        m_recording.Read(animationRecording);
        m_animating = true;
    }
}

void SampleGamePropInstance::Reset()
{
    m_node->SetTransform(&m_startPose.Translation, &m_startPose.Rotation, &m_startPose.Scaling);
    m_animOffset = 0;
}

void SampleGamePropInstance::Tick(double gameTime, float deltaTime)
{
    SampleGamePose animPose;
    if (m_recording.GetAt(gameTime+m_animOffset, true, animPose))
        m_node->SetTransform(&animPose.Translation, &animPose.Rotation, &animPose.Scaling);
}

