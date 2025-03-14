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

#include <nvrhi/nvrhi.h>
#include <memory>
#include <unordered_map>
#include <donut/render/MipMapGenPass.h>
#include "ToneMapping_cb.h"

namespace donut::engine
{
    class ShaderFactory;
    class CommonRenderPasses;
    class FramebufferFactory;
    class ICompositeView;
}

#ifndef TONEMAPPING_AUTOEXPOSURE_CPU
#error this must be defined
#endif

//namespace donut::render   // <- should we use donut::RTXPT for all our path tracing stuff?
//{
	enum class ExposureMode : uint32_t
	{
		AperturePriority,       // Keep aperture constant when modifying EV
		ShutterPriority,        // Keep shutter constant when modifying EV
	};

	struct ToneMappingParameters
	{
        ExposureMode exposureMode = ExposureMode::AperturePriority;
        ToneMapperOperator toneMapOperator = ToneMapperOperator::Aces;
        bool autoExposure = false;
        float exposureCompensation = 0.0f;
        float exposureValue = 0.0f;
        float filmSpeed = 100.f;
        float fNumber = 1.f;
        float shutter = 1.f;
        bool whiteBalance = false;
        float whitePoint = 6500.0f;
        float whiteMaxLuminance = 1.0f;
        float whiteScale = 11.2f;
        bool clamped = true;
        float exposureValueMin = -16.0f;
        float exposureValueMax = 16.0f;
	};

	static const std::unordered_map<ExposureMode, std::string> ExposureModeToString = {
        {ExposureMode::AperturePriority, "Aperture Priority"},
        {ExposureMode::ShutterPriority, "Shutter Priority"}
	};

	static const std::unordered_map<ToneMapperOperator, std::string> tonemapOperatorToString = {
        {ToneMapperOperator::Linear, "Linear"},
        {ToneMapperOperator::Reinhard, "Reinhard"},
        {ToneMapperOperator::ReinhardModified, "Reinhard Modified"},
        {ToneMapperOperator::HejiHableAlu, "Heji Hable ALU"},
        {ToneMapperOperator::HableUc2, "Hable UC2"},
        {ToneMapperOperator::Aces, "Aces"}
    };

    class ToneMappingPass
    {
    private:

        nvrhi::DeviceHandle m_device;
        nvrhi::ShaderHandle m_LuminanceShader;
        nvrhi::ShaderHandle m_ToneMapShader;

        struct PerViewData
        {
            nvrhi::TextureHandle luminanceTexture;
			nvrhi::FramebufferHandle luminanceFrameBuffer;
            std::unique_ptr<donut::render::MipMapGenPass> mipMapPass;
            nvrhi::BindingSetHandle luminanceBindingSet;
            nvrhi::BindingSetHandle colorBindingSet;
            nvrhi::TextureHandle sourceTexture;

#if TONEMAPPING_AUTOEXPOSURE_CPU
            // used for readback
            static constexpr int cReadbackLag = 3;  // if used once per frame then it should be backbuffer (swapchain) count + 1 to ensure it never blocks
            nvrhi::BufferHandle avgLuminanceBufferGPU;
            nvrhi::BufferHandle avgLuminanceBufferReadback[cReadbackLag];
            int                 avgLuminanceLastWritten     = -1;
            float               avgLuminanceLastCaptured    = 0.0;
#endif
        };
        
        std::vector<PerViewData> m_PerView;

        nvrhi::BufferHandle m_ToneMappingCB;

        nvrhi::SamplerHandle m_linearSampler;
        nvrhi::SamplerHandle m_pointSampler;

        float m_FrameTime = 0.f;

		nvrhi::BindingLayoutHandle m_LuminanceBindingLayout;
		nvrhi::GraphicsPipelineHandle m_LuminancePso;

        nvrhi::BindingLayoutHandle m_ToneMapBindingLayout;
        nvrhi::GraphicsPipelineHandle m_ToneMapPso;
#if TONEMAPPING_AUTOEXPOSURE_CPU
        nvrhi::ShaderHandle m_CaptureLuminanceShader;
        nvrhi::BindingLayoutHandle m_CaptureLumBindingLayout;
        nvrhi::ComputePipelineHandle m_CaptureLumPso;
#endif

        std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
        std::shared_ptr<donut::engine::FramebufferFactory> m_FramebufferFactory;
        
        ExposureMode m_ExposureMode;
        ToneMapperOperator m_ToneMapOperator;
        bool m_AutoExposure;
        float m_ExposureCompensation;
        float m_ExposureValue;
        float m_ExposureValueMin;
        float m_ExposureValueMax;
        float m_FilmSpeed;        
        float m_FNumber;
        float m_Shutter;
        
        bool m_WhiteBalance;
        float m_WhitePoint;
        float m_WhiteMaxLuminance;
        float m_WhiteScale;
        int m_Clamped;
        
        //Pre-computed fields
        float3x3 m_WhiteBalanceTransform; 
        float3 m_SourceWhite;
        float3x3 m_ColorTransform; 
        
        bool m_FrameParamsSet = false;

        void SetParameters(const ToneMappingParameters& params);
        void UpdateExposureValue();
		void UpdateWhiteBalanceTransform();
		void UpdateColorTransform();
        void GenerateMips(nvrhi::ICommandList* commandList, uint32_t numberOfViews);
    public:
        struct CreateParameters
        {
            bool isTextureArray = false;
            uint32_t histogramBins = 256;
            uint32_t numConstantBufferVersions = 16;
            nvrhi::IBuffer* exposureBufferOverride = nullptr;
            nvrhi::ITexture* colorLUT = nullptr;
        };

        ToneMappingPass(
            nvrhi::IDevice* device,
            std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
            std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
            std::shared_ptr<donut::engine::FramebufferFactory> colorFramebufferFactory,
            const donut::engine::ICompositeView& compositeView,
            nvrhi::TextureHandle sourceTexture
            );

        void PreRender(const ToneMappingParameters& params);
        bool Render(
            nvrhi::ICommandList* commandList,
            const donut::engine::ICompositeView& compositeView,
            nvrhi::ITexture* sourceTexture
        );

#if TONEMAPPING_AUTOEXPOSURE_CPU
        float3 GetPreExposedGray( uint viewIndex );
#endif

        void AdvanceFrame(float frameTime);

        nvrhi::TextureHandle GetLuminanceTexture(uint viewIndex)    { return m_PerView[viewIndex].luminanceTexture; }
};
//}