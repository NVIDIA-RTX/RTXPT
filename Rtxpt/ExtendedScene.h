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

#include <donut/engine/Scene.h>
#include <string.h>

namespace donut::engine
{
    constexpr int LightType_Environment = 1000;

    class EnvironmentLight : public donut::engine::Light
    {
    public:
        dm::float3 radianceScale = 1.f;
        int textureIndex = -1;
        float rotation = 0.f;
        std::string path;

        void Load(const Json::Value& node) override;
        [[nodiscard]] int GetLightType() const override { return LightType_Environment; }
        [[nodiscard]] std::shared_ptr<SceneGraphLeaf> Clone() override;
        void FillLightConstants(LightConstants& lightConstants) const override;
        bool SetProperty(const std::string& name, const dm::float4& value) override { assert( false ); return false; }    // not yet implemented, never needed
    };

    class PerspectiveCameraEx : public PerspectiveCamera
    {
    public:
        std::optional<bool>     enableAutoExposure;
        std::optional<float>    exposureCompensation;
        std::optional<float>    exposureValue;
        std::optional<float>    exposureValueMin;
        std::optional<float>    exposureValueMax;

        [[nodiscard]] std::shared_ptr<SceneGraphLeaf> Clone() override;
        void Load(const Json::Value& node) override;
        bool SetProperty(const std::string& name, const dm::float4& value) override;
    };

    // used to setup initial sample scene settings
    class SampleSettings : public SceneGraphLeaf
    {
    public:
        std::optional<bool>         realtimeMode;
        std::optional<bool>         enableAnimations;
        std::optional<bool>         enableReSTIRDI;
        std::optional<bool>         enableReSTIRGI;
        std::optional<int>          startingCamera;
        std::optional<float>        realtimeFireflyFilter;
        std::optional<int>          maxBounces;
        std::optional<int>          realtimeMaxDiffuseBounces;
        std::optional<int>          referenceMaxDiffuseBounces;
        std::optional<float>        textureMIPBias;

        [[nodiscard]] std::shared_ptr<SceneGraphLeaf> Clone() override;
        void Load(const Json::Value& node) override;
    };

    class ExtendedSceneTypeFactory : public donut::engine::SceneTypeFactory
    {
    public:
        std::shared_ptr<donut::engine::SceneGraphLeaf>  CreateLeaf(const std::string& type) override;
        std::shared_ptr<donut::engine::Material>        CreateMaterial() override;
        std::shared_ptr<donut::engine::MeshInfo>        CreateMesh() override;
        std::shared_ptr<donut::engine::MeshGeometry>    CreateMeshGeometry() override;
    };

    class ExtendedScene : public donut::engine::Scene
    {
    private:
        std::shared_ptr<SampleSettings> m_loadedSettings = nullptr;

    public:
        using Scene::Scene;

        bool LoadWithExecutor(const std::filesystem::path& jsonFileName, tf::Executor* executor) override;
        std::shared_ptr<SampleSettings> GetSampleSettingsNode() const { return m_loadedSettings; }

    private:
        // maybe switch to SceneGraphWalker?
        void ProcessNodesRecursive(donut::engine::SceneGraphNode* node);
    };

    std::shared_ptr<EnvironmentLight> FindEnvironmentLight(std::vector <std::shared_ptr<Light>> lights);
}
