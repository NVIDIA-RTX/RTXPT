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

#include <algorithm>
#include <cmath>

static inline uint32_t GetNrdHistoryLength(float accumulationTime, float smoothedRenderedFPS, uint32_t maximum)
{
    const float fps = std::isfinite(smoothedRenderedFPS) ? std::max(0.0f, smoothedRenderedFPS) : 60.0f;
    // Bound the conversion before casting to integer; this is NRD's history limit, not a tuning cap.
    const float boundedFPS = std::min(fps, float(maximum) / accumulationTime);
    return std::max(nrd::GetMaxAccumulatedFrameNum(accumulationTime, boundedFPS), 1u);
}

static inline uint32_t GetNrdHistoryFixLength(uint32_t fastHistory, uint32_t defaultFrameNum)
{
    return fastHistory > 0 ? std::min(defaultFrameNum, fastHistory - 1) : 0u;
}

namespace NrdConfig {

    // - 3x3 reconstruction mode because the path tracer uses full-grid probabilistic selection at the primary/PSR surface
    // - anti firefly because the path tracer may produce them

    nrd::RelaxSettings getDefaultRELAXSettings()
    {
        nrd::RelaxSettings settings = {};
        settings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
        settings.enableAntiFirefly = true;

        return settings;
    }

    nrd::ReblurSettings getDefaultREBLURSettings()
    {
        nrd::ReblurSettings settings = {};
        settings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
        settings.enableAntiFirefly = true;

        return settings;
    }

    void UpdateSettings(nrd::ReblurSettings& reblur, nrd::RelaxSettings& relax, float accumulationTime, float smoothedRenderedFPS)
    {
        reblur.maxAccumulatedFrameNum = GetNrdHistoryLength(accumulationTime, smoothedRenderedFPS, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
        reblur.maxFastAccumulatedFrameNum = reblur.maxAccumulatedFrameNum / 6;
        reblur.maxStabilizedFrameNum = reblur.maxAccumulatedFrameNum;
        reblur.historyFixFrameNum = GetNrdHistoryFixLength(reblur.maxFastAccumulatedFrameNum, nrd::ReblurSettings{}.historyFixFrameNum);

        relax.diffuseMaxAccumulatedFrameNum = GetNrdHistoryLength(accumulationTime, smoothedRenderedFPS, nrd::RELAX_MAX_HISTORY_FRAME_NUM);
        relax.specularMaxAccumulatedFrameNum = relax.diffuseMaxAccumulatedFrameNum;
        relax.diffuseMaxFastAccumulatedFrameNum = relax.diffuseMaxAccumulatedFrameNum / 6;
        relax.specularMaxFastAccumulatedFrameNum = relax.specularMaxAccumulatedFrameNum / 6;
        relax.historyFixFrameNum = GetNrdHistoryFixLength(relax.diffuseMaxFastAccumulatedFrameNum, nrd::RelaxSettings{}.historyFixFrameNum);
    }
}
