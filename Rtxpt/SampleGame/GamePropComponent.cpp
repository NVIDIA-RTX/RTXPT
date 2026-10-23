/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "GamePropComponent.h"
#include "GameScene.h"
#include "GameProps.h"
#include "GameModel.h"

#include <donut/core/log.h>
#include <donut/core/json.h>
#include <donut/core/math/math.h>
#include <donut/app/Camera.h>
#include <donut/engine/View.h>
#include <cmath>

#include "../Misc/Korgi.h"

using namespace donut;
using namespace donut::math;
using namespace donut::engine;
using namespace donut::render;

using namespace game;

PropComponentBase::PropComponentBase(class PropBase& prop, const std::string & type)
    : m_prop(prop), m_type(type)
{

}

class PropComponentPoliceLights : public PropComponentBase
{
public:
    PropComponentPoliceLights(class PropBase & prop, const std::string & type) : PropComponentBase(prop, type) {}

protected:
    float   m_start = 0.0f;
    float   m_stop  = 0.0f;

    std::shared_ptr<LightController> m_spotLeft;
    std::shared_ptr<LightController> m_spotRight;
    std::shared_ptr<LightController> m_blobLeft;
    std::shared_ptr<LightController> m_blobRight;
    float m_spotLeftIntensity = 0.0f;
    float m_spotRightIntensity = 0.0f;
    float m_blobLeftIntensity = 0.0f;
    float m_blobRightIntensity = 0.0f;
    dquat               m_blobLeftRot = dquat::identity();
    dquat               m_blobRightRot = dquat::identity();

    virtual void Load(const Json::Value & loadData) override
    {
        loadData["animTimeStart"] >> m_start;
        loadData["animTimeStop"] >> m_stop;

        for (auto & model : m_prop.GetModels())
        {
            for (auto & light : model->GetLights() )
            {
                auto & node = light->Node;
                if (node->GetName()=="SpotLeft")
                { m_spotLeft = light; m_spotLeftIntensity = light->Intensity; }
                if (node->GetName() == "SpotRight")
                { m_spotRight = light; m_spotRightIntensity = light->Intensity; }
                if (node->GetName()=="BlobLeft")
                { m_blobLeft = light; m_blobLeftRot = node->GetRotation(); m_blobLeftIntensity = light->Intensity; }
                if (node->GetName() == "BlobRight")
                { m_blobRight = light; m_blobRightRot = node->GetRotation(); m_blobRightIntensity = light->Intensity; }
            }
        }
    }

    void Tick(double gameTime, float animationTime, float deltaTime) override 
    {
        float lightTime = animationTime - m_start;
        float blobRightI = 0.0f;
        float blobRotAngle = 0.0f;
        if (animationTime > m_start && animationTime < m_stop)
        {
            float spotLeftI = 0.0f;
            float spotRightI = 0.0f;
            float blobLeftI = 0.0f;
            if (animationTime < m_start+2 || animationTime > m_stop-2) // intro / outro
            {
                spotLeftI   = m_spotLeftIntensity   * 0.1f;
                spotRightI  = m_spotRightIntensity  * 0.1f;
                blobLeftI    = m_blobLeftIntensity  * 0.01f;
                blobRightI   = m_blobRightIntensity * 0.01f;
            }
            else
            {
                float spotPeriod = 0.9f;
                spotPeriod = dm::saturate(fmodf(animationTime, spotPeriod) / spotPeriod);
                if (spotPeriod < 0.5f)
                    spotLeftI = m_spotLeftIntensity;
                else
                    spotRightI = m_spotRightIntensity;
                
                float blobPeriod = 1.1f;
                blobPeriod = dm::saturate(fmodf(animationTime, blobPeriod) / blobPeriod);
                blobRotAngle = (blobPeriod-0.5f) * PI_f * 2.0f;
                blobLeftI = m_blobLeftIntensity;
                blobRightI = m_blobRightIntensity;
            }
            m_spotLeft->Intensity   = spotLeftI;
            m_spotRight->Intensity  = spotRightI;
            m_blobLeft->Intensity   = blobLeftI;
            m_blobRight->Intensity  = blobRightI;
        }
        else
        {
            m_spotLeft->Intensity = 0.0f;
            m_spotRight->Intensity = 0.0f;
            m_blobLeft->Intensity = 0.0f;
            m_blobRight->Intensity = 0.0f;
        }
        dquat rot = rotationQuat<double>( {(double)blobRotAngle, 0, 0} );
        m_blobLeft->Node->SetRotation( m_blobLeftRot * rot );
        m_blobRight->Node->SetRotation(m_blobRightRot * rot);
    }
};

class PropComponentBasicInteractableUI : public PropComponentBase
{
public:
    PropComponentBasicInteractableUI(class PropBase& prop, const std::string& type) : PropComponentBase(prop, type) { }

protected:

    virtual void Load(const Json::Value& loadData) override
    {
        // loadData["animTimeStart"] >> m_start;
    }

    void Tick(double gameTime, float animationTime, float deltaTime) override
    {
    }

    ScreenGUISel StandaloneGUI(const std::shared_ptr<donut::engine::PlanarView> & view, const float2 & mousePos, const float2 & displaySize) override
    {
        ScreenGUISel sel;

        box3 bbox = m_prop.GetNode()->GetGlobalBoundingBox();
        if (bbox.isempty())
            return sel;

        // Mouse ray through a point unprojected at NDC z = 0.5, which is finite for both regular and reversed (infinite) Z
        const float2 ndc = float2(mousePos.x / displaySize.x * 2.0f - 1.0f, 1.0f - mousePos.y / displaySize.y * 2.0f);
        const float4 rayPt = float4(ndc, 0.5f, 1.0f) * view->GetInverseViewProjectionMatrix();
        const float3 rayOrigin = view->GetViewOrigin();
        const float3 rayDir = normalize(rayPt.xyz() / rayPt.w - rayOrigin);

        // Slab test; tEnter < 0 also rejects a box containing the camera, which would otherwise win over every prop inside or beyond it
        const float3 t0 = (bbox.m_mins - rayOrigin) / rayDir, t1 = (bbox.m_maxs - rayOrigin) / rayDir;
        const float tEnter = maxComponent(min(t0, t1)), tExit = minComponent(max(t0, t1));
        if (tEnter > tExit || tEnter < 0)
            return sel;

        sel.ScreenPos = mousePos;
        sel.Selected = true;
        sel.RangeToCamera = tEnter;
        return sel;
    }
};

std::shared_ptr<PropComponentBase> PropComponentBase::Create(class PropBase & prop, const Json::Value & loadData)
{
    std::string type;
    loadData["type"] >> type;

    std::shared_ptr<PropComponentBase> ret = nullptr;
    if (type == "PoliceLightingOnRX6")
        ret = std::make_shared<PropComponentPoliceLights>(prop, type);
    if (type == "BasicInteractableUI")
        ret = std::make_shared<PropComponentBasicInteractableUI>(prop, type);
    if (ret != nullptr)
        ret->Load(loadData);
    return ret;
}

