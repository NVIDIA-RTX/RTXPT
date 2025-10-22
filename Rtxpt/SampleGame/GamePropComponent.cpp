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
        float3 bcenter = bbox.center();
        float4x4 viewProj = view->GetViewProjectionMatrix();

        static constexpr float2 invalidPos = float2(FLT_MAX, FLT_MAX);
        auto projectToScreen = [&viewProj, &view, &displaySize](const float3 & worldPos)
        {
            float4 projv = float4(worldPos, 1) * viewProj;
            projv /= projv.w;
            if( projv.z < 0 )
                return invalidPos;
            projv.xy() = projv.xy() * float2(0.5,-0.5) + float2(0.5,0.5);
            projv.xy() *= displaySize;
            if (projv.x < 0 || projv.x > displaySize.x || projv.y < 0 || projv.y > displaySize.y)
                return invalidPos;
            return projv.xy();
        };

        float2 sCenter = projectToScreen(bcenter);
        if (sCenter.x == FLT_MAX)
            return sel;

        float sRadius = 0.0;
        for( int i = 0; i < 8; i++ )
        {
            float2 sCorner = projectToScreen(bbox.getCorner(i));
            if (sCorner.x == FLT_MAX) 
                continue;
            sRadius = max( sRadius, length(sCenter - sCorner) );
        }

        if ( sRadius > 0 )
        {
            sRadius += 10.0f;

            if ( length(mousePos - sCenter) < sRadius)
            {
                sel.ScreenPos = sCenter;
                sel.ScreenRadius = sRadius + 10.0f;
                sel.Selected = true;
                sel.RangeToCamera = length(bcenter - view->GetViewOrigin());
            }
        }
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

