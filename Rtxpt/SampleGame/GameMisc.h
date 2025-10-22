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

#include <donut/app/Camera.h>

namespace donut::app
{
    class FirstPersonCamera;
}

class GameScene;

namespace game
{
    class ModelInstance;

    struct Pose
    {
        double3     Translation     = { 0, 0, 0 };
        dquat       Rotation        = { 0, 0, 0, 1 };
        double3     Scaling         = { 1, 1, 1 };
        double      KeyTime         = 0.0;

        bool Read(const Json::Value& node);
        Json::Value Write();

        affine3     ToTransform() const;
        std::tuple<float3, float3, float3> GetPosDirUp() const;

        void        SetTransform(const affine3 & transform);
        void        SetTransformFromCamera(const float3 & pos, const float3 & dir, const float3 & up);
    };

    struct KeyframeAnimation
    {
        std::vector<Pose>     Keys;
        double                KeyTimeMin;
        double                KeyTimeMax;

        bool Read(const Json::Value& node);
        Json::Value Write();

        void FromKeys(const std::vector<game::Pose> & keys);

        bool GetAt(double time, bool wrap, Pose & outPose, float & outAnimTime);

    private:
        int                             LastFound = -1;
    };

    // Currently only tracking analytic lights, but could be extended to emissive; used to set color, intensity and switch on/off; also needed to
    // properly update actual light radius from node scaling, which isn't set from node transforms.
    struct LightController
    {
        donut::engine::SceneGraphNode * Node                = nullptr;

        float3                          Color               = float3(1,1,1);
        float                           Intensity           = 1.0;
        bool                            Enabled             = true;
        bool                            ToggleOnUIClick     = false;
        float                           InnerAngle          = 0.0f;
        float                           OuterAngle          = 0.0f;
        float                           AutoOffTime         = 0.0f; // for blinking lights // TODO: make it so that negative means random()*abs(val)
        float                           AutoOnTime          = 0.0f; // for blinking lights // TODO: make it so that negative means random()*abs(val)
        float                           AutoOnOffTimeOffset = 0.0f; // ability to desync blinking lights if many

        donut::engine::SpotLight *      GetSpotLight()      { if ( Node == nullptr ) return nullptr; return dynamic_cast<donut::engine::SpotLight*>(Node->GetLeaf().get() ); }
        donut::engine::PointLight *     GetPointLight()     { if ( Node == nullptr ) return nullptr; return dynamic_cast<donut::engine::PointLight*>(Node->GetLeaf().get() ); }

        bool                            Read(const Json::Value& node);
        Json::Value                     Write();
    };

    struct ScreenGUISel
    {
        float2  ScreenPos       = {0,0};
        float   ScreenRadius    = 0.0f;
        float   RangeToCamera   = FLT_MAX;
        bool    Selected        = false;
    };
}

inline void operator >>(class Json::Value const& h, game::Pose& p)
{
    p.Read(h);
}
