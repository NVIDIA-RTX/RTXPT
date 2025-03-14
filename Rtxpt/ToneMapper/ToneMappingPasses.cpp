/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/View.h>
#include <sstream>
#include <assert.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/core/log.h>

using namespace donut::math;
#include "ToneMappingPasses.h"
#include "ToneMapping_cb.h"
#include "ColorUtils.h"

using namespace donut::engine;
using namespace donut::render;

#ifndef TONEMAPPING_AUTOEXPOSURE_CPU
#error this must be defined
#endif

ToneMappingPass::ToneMappingPass(
    nvrhi::IDevice* device,
    std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
    std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
    std::shared_ptr<donut::engine::FramebufferFactory> colorFramebufferFactory,
    const donut::engine::ICompositeView& compositeView,
	nvrhi::TextureHandle sourceTexture)
    : m_device(device)
    , m_commonPasses(commonPasses)
    , m_FramebufferFactory(colorFramebufferFactory)
{
    const IView* sampleView = compositeView.GetChildView(ViewType::PLANAR, 0);
    nvrhi::IFramebuffer* colorSampleFramebuffer = m_FramebufferFactory->GetFramebuffer(*sampleView);
    {
        m_LuminanceShader = shaderFactory->CreateShader("app/ToneMapper/luminance_ps.hlsl", "main", nullptr, nvrhi::ShaderType::Pixel);
        m_ToneMapShader = shaderFactory->CreateShader("app/ToneMapper/ToneMapping.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
#if TONEMAPPING_AUTOEXPOSURE_CPU
        m_CaptureLuminanceShader = shaderFactory->CreateShader("app/ToneMapper/ToneMapping.hlsl", "capture_cs", nullptr, nvrhi::ShaderType::Compute);
#endif
    }

    nvrhi::BufferDesc constantBufferDesc;
    constantBufferDesc.byteSize = sizeof(ToneMappingConstants);
    constantBufferDesc.debugName = "ToneMappingConstants";
    constantBufferDesc.isConstantBuffer = true;
    constantBufferDesc.isVolatile = true;
    constantBufferDesc.maxVersions = 16;// params.numConstantBufferVersions;
    m_ToneMappingCB = m_device->createBuffer(constantBufferDesc);

    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.setBorderColor(nvrhi::Color(0.f));
    samplerDesc.setAllFilters(true);
    samplerDesc.setMipFilter(false);
    samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
    m_linearSampler = m_device->createSampler(samplerDesc);

    samplerDesc.setAllFilters(false);
	m_pointSampler = m_device->createSampler(samplerDesc);


    m_PerView.resize(compositeView.GetNumChildViews(ViewType::PLANAR));
    {
        for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
        {
            const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);
            nvrhi::IFramebuffer* sampleFrameBuffer = m_FramebufferFactory->GetFramebuffer(*view);
            PerViewData& perViewData = m_PerView[viewIndex];

            nvrhi::Rect viewExtent = view->GetViewExtent();
            uint32_t viewportWidth = viewExtent.maxX - viewExtent.minX;
            uint32_t viewportHeight = viewExtent.maxY - viewExtent.minY;

            nvrhi::Format format = sourceTexture->getDesc().format == nvrhi::Format::RGBA32_FLOAT ? 
                nvrhi::Format::R32_FLOAT : nvrhi::Format::R16_FLOAT;

            nvrhi::TextureDesc luminanceTextureDesc;
            luminanceTextureDesc.format = format;
            luminanceTextureDesc.width = 1U << (int)log2(viewportWidth); //Lower to nearest power of 2
            luminanceTextureDesc.height = 1U << (int)log2(viewportHeight);
            uint32_t dims = viewportWidth | viewportHeight;
            luminanceTextureDesc.mipLevels = (uint32_t)(log2(dims) + 1); //Calculate the number of mip levels required
            luminanceTextureDesc.isRenderTarget = true;
            luminanceTextureDesc.isUAV = true;
            luminanceTextureDesc.debugName = "Luminance Texture";
            luminanceTextureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            luminanceTextureDesc.keepInitialState = true;
            perViewData.luminanceTexture = m_device->createTexture(luminanceTextureDesc);
            perViewData.luminanceFrameBuffer = m_device->createFramebuffer(
                nvrhi::FramebufferDesc().addColorAttachment(perViewData.luminanceTexture));

#if TONEMAPPING_AUTOEXPOSURE_CPU
            // readback for luminance coming out of tonemapper so we can set exposure on the CPU side
            {
                nvrhi::BufferDesc bufferDesc;
                bufferDesc.byteSize = 4;
                bufferDesc.format = nvrhi::Format::R32_FLOAT;
                bufferDesc.canHaveUAVs = true;
                bufferDesc.initialState = nvrhi::ResourceStates::Common;
                bufferDesc.keepInitialState = true;
                bufferDesc.debugName = "AvgLuminanceBuffer";
                bufferDesc.canHaveTypedViews = true;
                perViewData.avgLuminanceBufferGPU = device->createBuffer(bufferDesc);

                bufferDesc.canHaveUAVs = false;
                bufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
                bufferDesc.debugName = "AvgLuminanceReadbackBuffer";
                for (int i = 0; i < PerViewData::cReadbackLag; i++)
                    perViewData.avgLuminanceBufferReadback[i] = device->createBuffer(bufferDesc);
            }
#endif

            perViewData.mipMapPass = std::make_unique<MipMapGenPass>(m_device, shaderFactory, perViewData.luminanceTexture, MipMapGenPass::MODE_COLOR);
        }

		nvrhi::BindingLayoutDesc layoutDesc;
		layoutDesc.visibility = nvrhi::ShaderType::Pixel;
		layoutDesc.bindings = {
			nvrhi::BindingLayoutItem::Texture_SRV(0),
			nvrhi::BindingLayoutItem::Sampler(1)
		};
		m_LuminanceBindingLayout = m_device->createBindingLayout(layoutDesc);

		nvrhi::GraphicsPipelineDesc pipelineDesc;
		pipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
		pipelineDesc.VS = m_commonPasses->m_FullscreenVS;
		pipelineDesc.PS = m_LuminanceShader;
		pipelineDesc.bindingLayouts = { m_LuminanceBindingLayout };

		pipelineDesc.renderState.rasterState.setCullNone();
		pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
		pipelineDesc.renderState.depthStencilState.stencilEnable = false;

		m_LuminancePso = m_device->createGraphicsPipeline(pipelineDesc, m_PerView[0].luminanceFrameBuffer);

#if TONEMAPPING_AUTOEXPOSURE_CPU
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(0),
            nvrhi::BindingLayoutItem::Sampler(0)
        };
        m_CaptureLumBindingLayout = m_device->createBindingLayout(layoutDesc);
        nvrhi::ComputePipelineDesc captureLumPSODesc;
        captureLumPSODesc.bindingLayouts = { m_CaptureLumBindingLayout };
        captureLumPSODesc.CS = m_CaptureLuminanceShader;
        m_CaptureLumPso = m_device->createComputePipeline(captureLumPSODesc);
#endif
    }

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Pixel;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Texture_SRV(1),
            nvrhi::BindingLayoutItem::Sampler(0),
            nvrhi::BindingLayoutItem::Sampler(1)
        };
        m_ToneMapBindingLayout = m_device->createBindingLayout(layoutDesc);

        nvrhi::GraphicsPipelineDesc pipelineDesc;
        pipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
        pipelineDesc.VS = m_commonPasses->m_FullscreenVS;
        pipelineDesc.PS = m_ToneMapShader;
        pipelineDesc.bindingLayouts = { m_ToneMapBindingLayout};

        pipelineDesc.renderState.rasterState.setCullNone();
        pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
        pipelineDesc.renderState.depthStencilState.stencilEnable = false;

        m_ToneMapPso = m_device->createGraphicsPipeline(pipelineDesc, colorSampleFramebuffer);
    }
}

void ToneMappingPass::PreRender(const ToneMappingParameters& params)
{
    SetParameters(params);
    UpdateExposureValue();
    UpdateWhiteBalanceTransform();
    UpdateColorTransform();
    m_FrameParamsSet = true;
}

bool ToneMappingPass::Render(
    nvrhi::ICommandList* commandList, 
    const donut::engine::ICompositeView& compositeView,
    nvrhi::ITexture* sourceTexture)
{
    assert( m_FrameParamsSet ); // forgot to call PreRender before this?
    m_FrameParamsSet = false;

    bool commandListWasClosed = false; // to track the need to re-create volatile constant buffers

    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
        PerViewData& viewData = m_PerView[viewIndex];

        if (viewData.sourceTexture != sourceTexture)
        {
            // Make sure that our cached binding sets represent the right source texture.
            viewData.luminanceBindingSet = nullptr;
            viewData.colorBindingSet = nullptr;
            viewData.sourceTexture = sourceTexture;
        }
    }

    //Not working as expected 
    if(m_AutoExposure) 
    {
		commandList->beginMarker("Luminance");
		for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
		{
            PerViewData & viewData = m_PerView[viewIndex];

            nvrhi::BindingSetHandle& bindingSet = viewData.luminanceBindingSet;
			if (!bindingSet)
			{
				nvrhi::BindingSetDesc bindingSetDesc;
				bindingSetDesc.bindings = {
					nvrhi::BindingSetItem::Texture_SRV(0, sourceTexture),
					nvrhi::BindingSetItem::Sampler(1, m_linearSampler)
				};
				bindingSet = m_device->createBindingSet(bindingSetDesc, m_LuminanceBindingLayout);
			}

			const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);

			nvrhi::GraphicsState state;
			state.pipeline = m_LuminancePso;
            state.framebuffer = viewData.luminanceFrameBuffer;
			state.bindings = { bindingSet };
            nvrhi::ViewportState viewportState;
            viewportState.addViewport(viewData.luminanceFrameBuffer->getFramebufferInfo().getViewport());
            viewportState.addScissorRect(nvrhi::Rect(viewData.luminanceFrameBuffer->getFramebufferInfo().getViewport()));
            state.viewport = viewportState;
           	commandList->setGraphicsState(state);

			nvrhi::DrawArguments args;
			args.instanceCount = 1;
			args.vertexCount = 4;
			commandList->draw(args);    

            GenerateMips(commandList, compositeView.GetNumChildViews(ViewType::PLANAR));

#if TONEMAPPING_AUTOEXPOSURE_CPU
            {
                nvrhi::BindingSetDesc bindingSetDesc; bindingSetDesc.bindings = {
                        nvrhi::BindingSetItem::Texture_SRV(0, viewData.luminanceTexture),
                        nvrhi::BindingSetItem::TypedBuffer_UAV(0, viewData.avgLuminanceBufferGPU),
                        nvrhi::BindingSetItem::Sampler(0, (true) ? m_linearSampler : m_pointSampler) };
                nvrhi::BindingSetHandle cbindingSet = m_device->createBindingSet(bindingSetDesc, m_CaptureLumBindingLayout);
                nvrhi::ComputeState cstate;
                cstate.bindings = { cbindingSet };
                cstate.pipeline = m_CaptureLumPso;
                commandList->setComputeState(cstate);
                commandList->dispatch(1, 1);

                if (viewData.avgLuminanceLastWritten == -1)  // first time init, we want correct data so wait & sync
                {
                    for (int i = 0; i < PerViewData::cReadbackLag; i++)
                        commandList->copyBuffer( viewData.avgLuminanceBufferReadback[i], 0, viewData.avgLuminanceBufferGPU, 0, viewData.avgLuminanceBufferReadback[i]->getDesc().byteSize);
                    commandList->close();
                    m_device->executeCommandList(commandList);
                    m_device->waitForIdle();
                    commandList->open();
                    commandListWasClosed = true;
                }
                viewData.avgLuminanceLastWritten = (viewData.avgLuminanceLastWritten + 1) % PerViewData::cReadbackLag;
                commandList->copyBuffer(viewData.avgLuminanceBufferReadback[viewData.avgLuminanceLastWritten], 0, viewData.avgLuminanceBufferGPU, 0,
                    viewData.avgLuminanceBufferReadback[viewData.avgLuminanceLastWritten]->getDesc().byteSize);
                {   // map/read/unmap CPU readable buffer
                    assert( viewData.avgLuminanceLastWritten >= 0 );
                    int toReadIndex = (viewData.avgLuminanceLastWritten + 1)%PerViewData::cReadbackLag;
                    void* pData = m_device->mapBuffer(viewData.avgLuminanceBufferReadback[toReadIndex], nvrhi::CpuAccessMode::Read);
                    assert(pData);
                    viewData.avgLuminanceLastCaptured = std::exp2f( *static_cast<float*>(pData) );
                    m_device->unmapBuffer(viewData.avgLuminanceBufferReadback[toReadIndex]);
                }
            }
#endif
		}
		commandList->endMarker();


    }

    commandList->beginMarker("ToneMapping");
    for (uint viewIndex = 0; viewIndex < compositeView.GetNumChildViews(ViewType::PLANAR); viewIndex++)
    {
		nvrhi::BindingSetHandle& bindingSet = m_PerView[viewIndex].colorBindingSet;
		if (!bindingSet)
		{
			nvrhi::BindingSetDesc bindingSetDesc;
			bindingSetDesc.bindings = {
				nvrhi::BindingSetItem::ConstantBuffer(0, m_ToneMappingCB),
				nvrhi::BindingSetItem::Texture_SRV(0, sourceTexture),                       //Color texture
				nvrhi::BindingSetItem::Texture_SRV(1, m_PerView[viewIndex].luminanceTexture),                    //Luminance Texture
				nvrhi::BindingSetItem::Sampler(0, m_linearSampler),    //Luminance sampler
				nvrhi::BindingSetItem::Sampler(1, m_pointSampler)      //Color sampler
			};
			bindingSet = m_device->createBindingSet(bindingSetDesc, m_ToneMapBindingLayout);
		}
        const IView* view = compositeView.GetChildView(ViewType::PLANAR, viewIndex);

        nvrhi::GraphicsState state;
        state.pipeline = m_ToneMapPso;
        state.framebuffer = m_FramebufferFactory->GetFramebuffer(*view);
        state.bindings = { bindingSet };
        state.viewport = view->GetViewportState();

        ToneMappingConstants toneMappingConsts = {};
        toneMappingConsts.whiteScale = m_WhiteScale;
        toneMappingConsts.whiteMaxLuminance = m_WhiteMaxLuminance;
        toneMappingConsts.clamped = m_Clamped;
        toneMappingConsts.toneMapOperator = (uint32_t)m_ToneMapOperator;
        toneMappingConsts.autoExposure = m_AutoExposure;
#if TONEMAPPING_AUTOEXPOSURE_CPU
        toneMappingConsts.avgLuminance = m_PerView[viewIndex].avgLuminanceLastCaptured;
#else
        toneMappingConsts.avgLuminance = 0; // unused
#endif
        if(m_AutoExposure)
        {
            toneMappingConsts.autoExposureLumValueMin = std::exp2f( m_ExposureValueMin );
            toneMappingConsts.autoExposureLumValueMax = std::exp2f( m_ExposureValueMax );
        }
        else
        {
            toneMappingConsts.autoExposureLumValueMin = std::exp2f( -16.0 );
            toneMappingConsts.autoExposureLumValueMax = std::exp2f( 16.0 );
        }

        // Copy 3x3 to 3x4 Is there a better way to do this?
       /* affine3 transform(m_ColorTransform, float3(0,0,0));
        affineToColumnMajor(transform, toneMappingConsts.colorTransform);*/ //incorrect output, why?
		toneMappingConsts.colorTransform = float3x4::identity();
		toneMappingConsts.colorTransform[0] = float4(m_ColorTransform.col(0), 0);
		toneMappingConsts.colorTransform[1] = float4(m_ColorTransform.col(1), 0);
		toneMappingConsts.colorTransform[2] = float4(m_ColorTransform.col(2), 0);
        commandList->writeBuffer(m_ToneMappingCB, &toneMappingConsts, sizeof(ToneMappingConstants));

        commandList->setGraphicsState(state);

        nvrhi::DrawArguments args;
        args.instanceCount = 1;
        args.vertexCount = 4;
        commandList->draw(args);
    }
    commandList->endMarker();
    return commandListWasClosed;
}

void ToneMappingPass::AdvanceFrame(float frameTime)
{
    m_FrameTime = frameTime;
    m_FrameParamsSet = false;
}


static const float g_minLogLuminance = -10; // TODO: figure out how to set these properly
static const float g_maxLogLuminamce = 4;

void ToneMappingPass::SetParameters(const ToneMappingParameters& params)
{
    m_ExposureMode = params.exposureMode;
    m_ToneMapOperator = params.toneMapOperator;
    m_AutoExposure = params.autoExposure;
    m_ExposureCompensation = params.exposureCompensation;
	m_ExposureValue = params.exposureValue;
	m_FilmSpeed = params.filmSpeed;
	m_FNumber = params.fNumber;
	m_Shutter = params.shutter;
	m_WhiteBalance = params.whiteBalance;
	m_WhitePoint = params.whitePoint;
	m_WhiteMaxLuminance = params.whiteMaxLuminance;
	m_WhiteScale = params.whiteScale;
    m_Clamped = params.clamped;
    m_ExposureValueMin = params.exposureValueMin;
    m_ExposureValueMax = params.exposureValueMax;
}


void ToneMappingPass::UpdateWhiteBalanceTransform()
{
	//Calculate color transform for the current white point. 
	m_WhiteBalanceTransform = m_WhiteBalance ?
		calculateWhiteBalanceTransformRGB_Rec709(m_WhitePoint) :
		dm::float3x3::identity();

	//Calculate source illuminant, i.e. the color that transform to a pure white (1, 1, 1) output at the current color settings.
	m_SourceWhite = inverse(m_WhiteBalanceTransform) * float3(1, 1, 1);
}
    
void ToneMappingPass::UpdateExposureValue()
{
	const float kShutterMin = 0.001f;        // Min reciprocal shutter time
	const float kShutterMax = 10000.f;       // Max reciprocal shutter time
	const float kFNumberMin = 0.1f;          // Minimum fNumber, > 0 to avoid numerical issues (i.e., non-physical values are allowed)
	const float kFNumberMax = 100.f;       

	// EV is ultimately derived from shutter and fNumber; set its range based on that of its inputs
	const float kExposureValueMin = std::log2(kShutterMin * kFNumberMin * kFNumberMin);
    const float kExposureValueMax = std::log2(kShutterMax * kFNumberMax * kFNumberMax);
    m_ExposureValue = clamp(m_ExposureValue, kExposureValueMin, kExposureValueMax);

    switch (m_ExposureMode)
    {
	case ExposureMode::AperturePriority:
		// Set shutter based on EV and aperture.
		m_Shutter = std::pow(2.f, m_ExposureValue) / (m_FNumber * m_FNumber);
		m_Shutter = clamp(m_Shutter, kShutterMin, kShutterMax);
        break;
    case ExposureMode::ShutterPriority:
		// Set aperture based on EV and shutter.
		m_FNumber = std::sqrt(std::pow(2.f, m_ExposureValue) / m_Shutter);
		m_FNumber = clamp(m_FNumber, kFNumberMin, kFNumberMax);
    }
}

void ToneMappingPass::UpdateColorTransform()
{
	//Exposure scale due to exposure compensation 
	float exposureScale = pow(2.f, m_ExposureCompensation);
	float manualExposureScale = 1.f;

	if (!m_AutoExposure)
	{
       	float normConstant = 1.f / 100.f;
		manualExposureScale = (normConstant * m_FilmSpeed) / (m_Shutter * m_FNumber * m_FNumber);
	}
	m_ColorTransform = m_WhiteBalanceTransform * exposureScale * manualExposureScale;
}

#if TONEMAPPING_AUTOEXPOSURE_CPU
float3 ToneMappingPass::GetPreExposedGray(uint viewIndex) 
{ 
    assert(m_FrameParamsSet); // forgot to call PreRender before this?
    assert(viewIndex < m_PerView.size()); 
    if (viewIndex >= m_PerView.size()) 
        return float3(0,0,0); 

    // should probably sync to be exact https://en.wikipedia.org/wiki/Middle_gray in the future but it does the trick for now
    float3 result = inverse(m_ColorTransform) * float3(0.18f, 0.18f, 0.18f);
    if (m_AutoExposure)
    {
        result /= float3((float)TONEMAPPING_EXPOSURE_KEY / m_PerView[viewIndex].avgLuminanceLastCaptured);
    }
    return result;
}
#endif

void ToneMappingPass::GenerateMips(nvrhi::ICommandList* commandList, uint32_t numberOfViews)
{
    for (uint32_t i = 0; i < numberOfViews; i++)
    {
        m_PerView[i].mipMapPass->Dispatch(commandList);
    }
}
