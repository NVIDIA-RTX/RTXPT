/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// See https://github.com/fstrugar/XeGTAO/blob/master/Source/Rendering/Shaders/vaHelperTools.hlsl
// Copyright (C) 2016-2021, Intel Corporation, 
// SPDX-License-Identifier: MIT, Author(s):  Filip Strugar (filip.strugar@intel.com)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __ZOOM_TOOL_HLSL__
#define __ZOOM_TOOL_HLSL__

struct ZoomToolShaderConstants
{
    float4                  SourceRectangle;

    float2                  CursorViewportPos;
    int                     ZoomFactor;
    float                   Dummy1;
};

#if !defined(__cplusplus)

ConstantBuffer<ZoomToolShaderConstants> g_zoomToolConstants     : register(b0);
RWTexture2D<float4>                     g_screenTexture         : register(u0);

bool IsInRect( float2 pt, float4 rect )
{
    return ( (pt.x >= rect.x) && (pt.x <= rect.z) && (pt.y >= rect.y) && (pt.y <= rect.w) );
}

void DistToClosestRectEdge( float2 pt, float4 rect, out float dist, out int edge )
{
    edge = 0;
    dist = 1e20;

    float distTmp;
    distTmp = abs( pt.x - rect.x );
    if( distTmp <= dist ) { dist = distTmp; edge = 2; }  // left

    distTmp = abs( pt.y - rect.y );
    if( distTmp <= dist ) { dist = distTmp; edge = 3; }  // top

    distTmp = abs( pt.x - rect.z ); 
    if( distTmp <= dist ) { dist = distTmp; edge = 0; }  // right

    distTmp = abs( pt.y - rect.w );
    if( distTmp <= dist ) { dist = distTmp; edge = 1; }  // bottom
}

float2 RectToRect( float2 pt, float2 srcRCentre, float2 srcRSize, float2 dstRCentre, float2 dstRSize )
{
    pt -= srcRCentre;
    pt /= srcRSize;

    pt *= dstRSize;
    pt += dstRCentre;
    
    return pt;
}

[numthreads(16, 16, 1)]
void main( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    const float2 screenPos = float2( dispatchThreadID ) + float2( 0.5, 0.5 ); // same as SV_Position

    const float zoomFactor = g_zoomToolConstants.ZoomFactor;

    float4 srcRect  = g_zoomToolConstants.SourceRectangle;
    float4 srcColor = g_screenTexture[dispatchThreadID];

    uint2 screenSizeUI;
    g_screenTexture.GetDimensions( screenSizeUI.x, screenSizeUI.y );

    const float2 screenSize     = float2(screenSizeUI);
    const float2 screenCenter   = float2(screenSizeUI) * 0.5;

    float2 srcRectSize = float2( srcRect.z - srcRect.x, srcRect.w - srcRect.y );
    float2 srcRectCenter = srcRect.xy + srcRectSize.xy * 0.5;

    float2 displayRectSize = srcRectSize * zoomFactor.xx;
    float2 displayRectCenter; 
    displayRectCenter.x = (srcRectCenter.x > screenCenter.x)?(srcRectCenter.x - srcRectSize.x * 0.5 - displayRectSize.x * 0.5 - 100):(srcRectCenter.x + srcRectSize.x * 0.5 + displayRectSize.x * 0.5 + 100);
    
    //displayRectCenter.y = (srcRectCenter.y > screenCenter.y)?(srcRectCenter.y - srcRectSize.y * 0.5 - displayRectSize.y * 0.5 - 50):(srcRectCenter.y + srcRectSize.y * 0.5 + displayRectSize.y * 0.5 + 50);
    displayRectCenter.y = lerp( displayRectSize.y/2, screenSize.y - displayRectSize.y/2, srcRectCenter.y / screenSize.y );
    
    float4 displayRect = float4( displayRectCenter.xy - displayRectSize.xy * 0.5, displayRectCenter.xy + displayRectSize.xy * 0.5 );

    bool chessPattern = (((uint)screenPos.x + (uint)screenPos.y) % 2) == 0;

    if( IsInRect(screenPos.xy, displayRect ) )
    {
        float2 texCoord = RectToRect( screenPos.xy, displayRectCenter, displayRectSize, srcRectCenter, srcRectSize );
        float2 cursorPosZoomed = g_zoomToolConstants.CursorViewportPos.xy; //RectToRect( g_globals.CursorViewportPosition.xy+0.5, displayRectCenter, displayRectSize, srcRectCenter, srcRectSize );
        float3 colour = g_screenTexture.Load( int2( texCoord ) ).rgb;

        float crosshairThickness = 0.5f;
        if( abs(texCoord.x - cursorPosZoomed.x)<crosshairThickness || abs(texCoord.y - cursorPosZoomed.y)<crosshairThickness )
        {
            if( length(float2(cursorPosZoomed) - float2(texCoord)) > 1.5 )
                colour.rgb = saturate( float3( 2, 1, 1 ) - normalize(colour.rgb) );
        }

        {
            // draw destination box frame
            float dist; int edge;
            DistToClosestRectEdge( screenPos.xy, displayRect, dist, edge );

            if( dist < 1.1 )
            {
                g_screenTexture[dispatchThreadID] = float4( 1.0, 0.8, 0.8, dist < 1.1 );
                return;
            }
        }
       
        g_screenTexture[dispatchThreadID] = float4( colour, 1 );
        return;
    }

    srcRect.xy -= 1.0;
    srcRect.zw += 1.0;
    if( IsInRect(screenPos.xy, srcRect ) )
    {
        // draw source box frame
        float dist; int edge;
        DistToClosestRectEdge( screenPos.xy, srcRect, dist, edge );

        if( dist < 1.1 )
        {
            g_screenTexture[dispatchThreadID] = float4( 0.8, 1, 0.8, 1 ); // lerp( srcColor, float4( 0.8, 1, 0.8, 1 ) );
            return;
        }
    }

}
#endif

#endif // __ZOOM_TOOL_HLSL__