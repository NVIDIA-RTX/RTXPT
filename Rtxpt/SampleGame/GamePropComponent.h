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
#include "GameMisc.h"

namespace donut::engine
{
    class PlanarView;
    class SceneCamera;
}

namespace game
{
    // placeholder for prop components - which are things that you add to props that makes them do things
    class PropComponentBase
    {
    public:
        PropComponentBase(class PropBase & prop, const std::string & type);
        virtual ~PropComponentBase() { }

        virtual void            Tick(double gameTime, float animationTime, float deltaTime) = 0;
        virtual ScreenGUISel    StandaloneGUI(const std::shared_ptr<donut::engine::PlanarView> & view, const float2 & mousePos, const float2 & displaySize) { return ScreenGUISel{}; }

        static std::shared_ptr<PropComponentBase> Create(class PropBase & prop, const Json::Value & loadData);

    protected:
        virtual void            Load(const Json::Value & loadData) { }

    protected:
        PropBase &              m_prop;
        std::string             m_type;
    };
}