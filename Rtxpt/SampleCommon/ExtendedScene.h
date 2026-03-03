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

constexpr int LightType_Environment = 1000;

struct LightSamplerLink
{
    int         IndexOrBase = -1;       // index of a corresponding PolymorphicLight in the light sampler
    int         LastUpdateTag = -1;     // identifier of when IndexOrBase was last updated (either frame index or similar)
};

struct LightExtension// : virtual public donut::engine::Light
{
    LightSamplerLink            LightLink;
    std::vector<std::string>    Proxies;

    void Load(const Json::Value& node);
    void Store(Json::Value& node) const;
    void Copy(LightExtension & src)         { LightLink = src.LightLink; Proxies = src.Proxies; }
};

class SpotLightEx : public donut::engine::SpotLight, public LightExtension
{
public:
    [[nodiscard]] std::shared_ptr<SceneGraphLeaf> Clone() override;
    void Load(const Json::Value& node) override;
    void Store(Json::Value& node) const override;
};

class PointLightEx : public donut::engine::PointLight, public LightExtension
{
public:
    [[nodiscard]] std::shared_ptr<SceneGraphLeaf> Clone() override;
    void Load(const Json::Value& node) override;
    void Store(Json::Value& node) const override;
};


struct MeshInstanceExtension
{
    std::vector<LightSamplerLink>       PerGeometryLightSamplerLinks;
    std::weak_ptr<LightExtension>       ProxiedAnalyticLight;

    void Copy(MeshInstanceExtension& src) { PerGeometryLightSamplerLinks = src.PerGeometryLightSamplerLinks; }
};

class MeshInstanceEx : public donut::engine::MeshInstance, public MeshInstanceExtension
{
public:
    explicit MeshInstanceEx(const std::shared_ptr<donut::engine::MeshInfo> & mesh);
    [[nodiscard]] std::shared_ptr<SceneGraphLeaf> Clone() override;
};

class SkinnedMeshInstanceEx : public donut::engine::SkinnedMeshInstance, public MeshInstanceExtension
{
public:
    explicit SkinnedMeshInstanceEx(std::shared_ptr<donut::engine::SceneTypeFactory> sceneTypeFactory, std::shared_ptr<donut::engine::MeshInfo> prototypeMesh);
    [[nodiscard]] std::shared_ptr<SceneGraphLeaf> Clone() override;
};

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

class PerspectiveCameraEx : public donut::engine::PerspectiveCamera
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
class SampleSettings : public donut::engine::SceneGraphLeaf
{
public:
    std::optional<bool>         realtimeMode;
    std::optional<bool>         enableAnimations;
    std::optional<int>          startingCamera;
    std::optional<float>        realtimeFireflyFilter;
    std::optional<int>          maxBounces;
    std::optional<int>          maxDiffuseBounces;
    std::optional<float>        textureMIPBias;

    [[nodiscard]] std::shared_ptr<SceneGraphLeaf> Clone() override;
    virtual void Load(const Json::Value& node) override;
};

// used to setup initial sample game settings (if any)
class GameSettings : public donut::engine::SceneGraphLeaf
{
    std::string                 jsonData;

    virtual [[nodiscard]] std::shared_ptr<SceneGraphLeaf> Clone() override;
    virtual void                Load(const Json::Value& node) override;

public:
    const std::string           GetJsonData() const { return jsonData; }
};

class ExtendedSceneTypeFactory : public donut::engine::SceneTypeFactory
{
public:
    std::shared_ptr<donut::engine::SceneGraphLeaf>  CreateLeaf(const std::string& type) override;
    std::shared_ptr<donut::engine::Material>        CreateMaterial() override;
    std::shared_ptr<donut::engine::MeshInfo>        CreateMesh() override;
    std::shared_ptr<donut::engine::MeshGeometry>    CreateMeshGeometry() override;
    std::shared_ptr<donut::engine::MeshInstance>    CreateMeshInstance(const std::shared_ptr<donut::engine::MeshInfo>& mesh);
    std::shared_ptr<donut::engine::SkinnedMeshInstance> CreateSkinnedMeshInstance(const std::shared_ptr<donut::engine::SceneTypeFactory> & sceneTypeFactory, const std::shared_ptr<donut::engine::MeshInfo> & prototypeMesh) override;
};

class ExtendedScene : public donut::engine::Scene
{
private:
    std::shared_ptr<SampleSettings> m_loadedSettings = nullptr;
    std::shared_ptr<GameSettings>   m_loadedGameSettings = nullptr;

public:
    using Scene::Scene;

    bool LoadWithThreadPool(const std::filesystem::path& jsonFileName, donut::engine::ThreadPool* threadPool) override;
    std::shared_ptr<SampleSettings> GetSampleSettingsNode() const   { return m_loadedSettings; }
    std::shared_ptr<GameSettings>   GetGameSettingsNode() const     { return m_loadedGameSettings; }

    const std::vector<donut::engine::SceneImportResult> & GetModels() const               { return m_Models; }

private:
    // maybe switch to SceneGraphWalker?
    void ProcessNodesRecursive(std::shared_ptr<donut::engine::SceneGraphNode> node);
};

std::shared_ptr<EnvironmentLight> FindEnvironmentLight(std::vector <std::shared_ptr<donut::engine::Light>> lights);
