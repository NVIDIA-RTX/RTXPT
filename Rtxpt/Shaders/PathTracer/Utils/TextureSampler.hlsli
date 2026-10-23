/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __TEXTURE_SAMPLER_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __TEXTURE_SAMPLER_HLSLI__

#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
#include "../../../../external/RtxTf/STFSamplerState.hlsli"
#endif

#if 0 // no interfaces in HLSL
/** Interface for texture sampling techniques.
    Types implementing this interface support sampling using different LOD computation techniques
*/
interface ITextureSampler
{
    /** Sample from a 2D texture using the level of detail computed by this method
    */
    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv);
};
#endif

/** Texture sampling using implicit gradients from finite differences within quads.

    Shader model 6.5 and lower *only* supports gradient operations in pixel shaders.
    Usage in other shader stages will generate a compiler error.

    Shader model 6.6 adds support for gradient operations in other shader stages,
    but the application is responsible for making sure the neighboring threads
    within 2x2 quads have appropriate data for gradient computations to be valid.
*/
struct ImplicitLodTextureSampler //: ITextureSampler
{
    void __init()
    {
    }
    static ImplicitLodTextureSampler make() { ImplicitLodTextureSampler ret; return ret; }

    float4 sampleTexture(Texture2D texture, SamplerState materialSampler, float2 texCoord, uniform bool debugColor = false)
    {
        return texture.Sample(materialSampler, texCoord);
    }
};

/** Texture sampling using an explicit scalar level of detail.
*/
struct ExplicitLodTextureSampler //: ITextureSampler
{
    float lod; ///< The explicit level of detail to use
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    STF_SamplerState stfSamplerState;  

    void __init(float lod, STF_SamplerState stfSamplerState)
    {
        this.lod = lod;
        this.stfSamplerState = stfSamplerState;
    }
    static ExplicitLodTextureSampler make(float lod, STF_SamplerState stfSamplerState) 
    {
        ExplicitLodTextureSampler ret; ret.__init(lod, stfSamplerState); return ret; 
    }
    float4 sampleTexture(Texture2D texture, SamplerState materialSampler, float2 texCoord, uniform bool debugColor = false)
    {
        return stfSamplerState.Texture2DSampleLevel(texture, materialSampler, texCoord, lod);
    }
#else // !RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    void __init(float lod) 
    {
        this.lod = lod;
    }
    static ExplicitLodTextureSampler make(float lod) 
    {
        ExplicitLodTextureSampler ret; ret.__init(lod); return ret; 
    }
    float4 sampleTexture(Texture2D texture, SamplerState materialSampler, float2 texCoord, uniform bool debugColor = false)
    {
        //if( debugColor ) { uint dummy0, dummy1, mipLevels; t.GetDimensions(0,dummy0,dummy1,mipLevels); return float4( GradientHeatMap( 1.0 - lod / float(mipLevels-1) ), t.SampleLevel(s, uv, lod).a ); };
        return texture.SampleLevel(materialSampler, texCoord, lod);
    }
#endif  // RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
};

/** Texture sampling using an explicit scalar level of detail using ray cones (with texture dimensions
    "subtracted" from the LOD value, and added back in before SampleLevel()).
*/
struct ExplicitRayConesLodTextureSampler //: ITextureSampler
{
    float rayconesLODWithoutTexDims;    ///< this is \Delta_t, which is texture independent, plus the rest of the terms, except the texture size, which is added below
#if RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    STF_SamplerState stfSamplerState;  

    void __init(float rayconesLODWithoutTexDims, STF_SamplerState stfSamplerState)
    {
        this.rayconesLODWithoutTexDims = rayconesLODWithoutTexDims;
        this.stfSamplerState = stfSamplerState;
    }
    static ExplicitRayConesLodTextureSampler make(float rayconesLODWithoutTexDims, STF_SamplerState stfSamplerState) 
    { 
        ExplicitRayConesLodTextureSampler ret; ret.__init(rayconesLODWithoutTexDims, stfSamplerState); return ret; 
    }
    float4 sampleTexture(Texture2D texture, SamplerState samplerState, float2 texCoord, uint baseLOD, uint mipLevels)
    {
        // assuming last mip level is 1x1, limit the max MIP to 16x16; this improves both quality and performance via better coherence for MIP sampling and sampling of integer level
        float lambda = 0.5 * baseLOD + rayconesLODWithoutTexDims;
        lambda = min(lambda, max((float) mipLevels - 5.0, 0.0));
        
        // if( debugColor ) { uint dummy0, dummy1, mipLevels; texCoord.GetDimensions(0,dummy0,dummy1,mipLevels); return float4( GradientHeatMap( 1.0 - lambda / float(mipLevels-1) ), t.SampleLevel(s, uv, lambda).a ); };
        return stfSamplerState.Texture2DSampleLevel(texture, samplerState, texCoord, lambda);
    }
#else // !RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE
    void __init(float rayconesLODWithoutTexDims)
    {
        this.rayconesLODWithoutTexDims = rayconesLODWithoutTexDims;
    }
    static ExplicitRayConesLodTextureSampler make(float rayconesLODWithoutTexDims) 
    { 
        ExplicitRayConesLodTextureSampler ret; ret.__init(rayconesLODWithoutTexDims); return ret; 
    }
    float4 sampleTexture(Texture2D texture, SamplerState samplerState, float2 texCoord, uint baseLOD, uint mipLevels)
    {
        // assuming last mip level is 1x1, limit the max MIP to 16x16; this improves both quality and performance via better coherence for MIP sampling and sampling of integer level
        float lambda = 0.5 * baseLOD + rayconesLODWithoutTexDims;
        lambda = min(lambda, max((float) mipLevels - 5.0, 0.0));
        
        // if( debugColor ) { uint dummy0, dummy1, mipLevels; texCoord.GetDimensions(0,dummy0,dummy1,mipLevels); return float4( GradientHeatMap( 1.0 - lambda / float(mipLevels-1) ), t.SampleLevel(s, uv, lambda).a ); };
        return texture.SampleLevel(samplerState, texCoord, lambda);
    }
#endif  // RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE

    float4 sampleTexture(Texture2D texture, SamplerState samplerState, float2 texCoord)
    {
        uint txw, txh, mipLevels;
        texture.GetDimensions(0, txw, txh, mipLevels);
        float baseLOD = log2(txw * txh);
        return sampleTexture(texture, samplerState, texCoord, baseLOD, mipLevels);
    }
};

#if 0 // unsolved compile errors
/** Texture sampling using an explicit scalar, i.e., isotropic, level of detail using ray diffs,
    with the final LOD computations done below, since they are dependent on texture dimensions.
    Use ExplicitGradientTextureSampler if you want anisotropic filtering with ray diffs.
*/
struct ExplicitRayDiffsIsotropicTextureSampler //: ITextureSampler
{
    enum class Mode { IsotropicOpenGLStyle, IsotropicPBRTStyle };
    static const Mode kMode = Mode::IsotropicOpenGLStyle;

    float2 dUVdx;               ///< derivatives in x over uv
    float2 dUVdy;               ///< derivatives in y over uv

    void __init(float2 dUVdx, float2 dUVdy)
    {
        this.dUVdx = dUVdx;
        this.dUVdy = dUVdy;
    }
    static ExplicitRayDiffsIsotropicTextureSampler make(float2 dUVdx, float2 dUVdy) { ExplicitRayDiffsIsotropicTextureSampler ret; ret.__init(dUVdx, dUVdy); return ret; }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        uint2 dim;
        t.GetDimensions(dim.x, dim.y);

        switch (kMode)
        {
        case Mode::IsotropicOpenGLStyle:
            {
                // Sharper, but alias sometimes for sharp edges textures.
                const float2 duvdx = dUVdx * dim.x;
                const float2 duvdy = dUVdy * dim.y;
                const float lambda = 0.5f * log2(max(dot(duvdx, duvdx), dot(duvdy, duvdy)));
                return t.SampleLevel(s, uv, lambda);
            }
        case Mode::IsotropicPBRTStyle:
            {
                // PBRT style (much blurrier, but never (?) aliases).
                const float filterWidth = 2.f * max(dim.x * max(abs(dUVdx.x), abs(dUVdy.x)), dim.y * max(abs(dUVdx.y), abs(dUVdy.y)));
                const float lambda = log2(filterWidth);
                return t.SampleLevel(s, uv, lambda);
            }
        }

        return float4(0,0,0,0);
    }
};
#endif

#if 0
/** Texture sampling using explicit screen-space gradients.
*/
struct ExplicitGradientTextureSampler //: ITextureSampler
{
    float2 gradX; ///< Gradient of texture coordinate in the screen-space X direction
    float2 gradY; ///< Gradient of texture coordiante in teh screen-space Y direction

    void __init(float2 gradX, float2 gradY)
    {
        this.gradX = gradX;
        this.gradY = gradY;
    }
    static ExplicitGradientTextureSampler make(float2 gradX, float2 gradY) { ExplicitGradientTextureSampler ret; ret.__init(gradX, gradY); return ret; }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv, uniform bool debugColor = false)
    {
        return t.SampleGrad(s, uv, gradX, gradY);
    }
};
#endif

#endif // __TEXTURE_SAMPLER_HLSLI__
