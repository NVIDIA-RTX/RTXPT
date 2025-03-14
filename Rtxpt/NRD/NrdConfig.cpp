/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NrdConfig.h"

namespace NrdConfig {

    nrd::RelaxSettings getDefaultRELAXSettings() {

        nrd::RelaxSettings settings;
        
         settings.enableAntiFirefly = true;

        settings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF;
        // (pixels) - pre-accumulation spatial reuse pass blur radius (0 = disabled, must be used in case of probabilistic sampling) <- we're using probabilistic sampling
        settings.diffusePrepassBlurRadius = 0.0f;
        settings.specularPrepassBlurRadius = 0.0f;  // <- using prepass blur causes more issues than it solves

        settings.atrousIterationNum = 5; // 5 is default; 4 gives better shadows but more boiling, 6 gives less boiling but loss in contact shadows

        // settings.lobeAngleFraction = 0.5f;
        // settings.roughnessFraction = 0.3;

        settings.specularLobeAngleSlack = 0.2f;         // good to hide noisy secondary bounces

        settings.depthThreshold = 0.004f;

        settings.diffuseMaxAccumulatedFrameNum = 25;
        settings.specularMaxAccumulatedFrameNum = 40;

        // aggressive anti-lag settings
        settings.diffuseMaxFastAccumulatedFrameNum = 3;
        settings.specularMaxFastAccumulatedFrameNum = 5;
        settings.antilagSettings.accelerationAmount = 0.95f;
        settings.antilagSettings.spatialSigmaScale = 0.5f;
        settings.antilagSettings.temporalSigmaScale = 0.25f;
        settings.antilagSettings.resetAmount = 0.95f;

        return settings;
    }

    nrd::ReblurSettings getDefaultREBLURSettings() {

        nrd::ReblurSettings settings;
        settings.enableAntiFirefly = true;
        settings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_5X5;
        settings.maxAccumulatedFrameNum = 50;

        // reducing prepass blurs to reduce loss of sharp shadows
        settings.diffusePrepassBlurRadius = 15.0f;
        settings.specularPrepassBlurRadius = 40.0f;

        return settings;
    }
}

