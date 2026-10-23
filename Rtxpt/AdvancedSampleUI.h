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

#include "SampleUI.h"

class AdvancedSampleUI : public SampleUI
{
public:
    using SampleUI::SampleUI;
    AdvancedSampleUI(donut::app::DeviceManager* deviceManager, class SampleBaseApp& baseApp, class Sample& app, SampleUIData& ui, bool NVAPI_SERSupported, const struct CommandLineOptions& cmdLine);
    void DrawSampleUI(float indent, float defItemWidth, const struct ImVec4& warnColor, const struct ImVec4& categoryColor) override;
};
