/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __PATH_TRACER_DEBUG_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __PATH_TRACER_DEBUG_HLSLI__

#include "Config.h"

#include "PathPayload.hlsli"
#include "StablePlanes.hlsli"
#if !defined(__cplusplus) // shader only!
#include "PathState.hlsli"
#endif
#include "Utils/Utils.hlsli"

// when editing don't forget to edit the UI in SampleUI.cpp (...ImGui::Combo( "Debug view"...)
enum class DebugViewType : int
{
    Disabled,
    ImagePlaneRayLength,
    DominantStablePlaneIndex,

    StablePlaneVirtualRayLength,
    StablePlaneMotionVectors,
    StablePlaneNormals,
    StablePlaneRoughness,
    StablePlaneDiffBSDFEstimate,
    StablePlaneDiffRadiance,
    StablePlaneDiffHitDist,
    StablePlaneSpecBSDFEstimate,
    StablePlaneSpecRadiance,
    StablePlaneSpecHitDist,
    StablePlaneRelaxedDisocclusion,
    StablePlaneDiffRadianceDenoised,
    StablePlaneSpecRadianceDenoised,
    StablePlaneCombinedRadianceDenoised,
    StablePlaneViewZ,
    StablePlaneThroughput,
    StablePlaneDenoiserValidation,
    
    StableRadiance,

    FirstHitBarycentrics,
    FirstHitFaceNormal,
    FirstHitGeometryNormal,
    FirstHitShadingNormal,
    FirstHitShadingTangent,
    FirstHitShadingBitangent,
    FirstHitFrontFacing,
    FirstHitThinSurface,
    FirstHitShaderPermutation,
    FirstHitDiffuse,
    FirstHitSpecular,
    FirstHitRoughness,
    FirstHitMetallic,
    VBufferMotionVectors,
    VBufferDepth,

    FirstHitOpacityMicroMapInWorld,
    FirstHitOpacityMicroMapOverlay,
    
    SecondarySurfacePosition,
    SecondarySurfaceRadiance,
    ReSTIRGIOutput,

    ReSTIRDIInitialOutput, 
    ReSTIRDITemporalOutput,
    ReSTIRDISpatialOutput,
    ReSTIRDIFinalOutput,  
    ReSTIRDIFinalContribution,
    ReGIRIndirectOutput,

    MaxCount
};


struct DebugConstants
{
	int     pickX;
	int     pickY;
	int     pick;
	float   debugLineScale;
    bool    showWireframe;
    int     debugViewType;
    int     debugViewStablePlaneIndex;
    int     exploreDeltaTree;
    int     imageWidth;
    int     imageHeight;
    int     mouseX;
    int     mouseY;
    float3  cameraPosW;
    float   _padding0;
};

#define MAX_DEBUG_PRINT_SLOTS   16

#define MAX_DEBUG_LINES         2048

struct DeltaTreeVizPathVertex
{
    DeltaLobe       deltaLobes[cMaxDeltaLobes];     // this should be stable with regards to material, invariant to surface Wi (viewing angle) and local surface properties (i.e. texture) and only currently used will have probability > 0
    uint            deltaLobeCount;                 // this should be stable with regards to material, invariant to surface Wi (viewing angle) and local surface properties (i.e. texture) and only currently used will have probability > 0
    float3          throughput;                     // throughput so far, not including next bounces
    float           nonDeltaPart;                   // estimate of light going through non-delta lobes (currently computed as "1 - sum(deltaLobes[].probability)")
    uint            vertexIndex;                    // 0 is camera, 1 is first bounce, etc
    uint            materialID;                     // 0xFFFFFFFF for sky
    uint            stableBranchID;                 // see PathState::stableBranchID
    float           volumeAbsorption;               // "1 - luminance(transmittance)" - this is the througput that gets lost between previous and this vertex due to volume absorption
    uint            padding0; // for use as stableBranchID
    uint            padding1; // for use as PrimarySurfaceChoice
    uint            padding2; // for use as SecondarySurfaceChoice
    float3          worldPos;
    bool            isDominant;

#if !defined(__cplusplus) // shader only!
    static DeltaTreeVizPathVertex make()   
    { DeltaTreeVizPathVertex ret; for (int i = 0; i < cMaxDeltaLobes; i++) ret.deltaLobes[i] = DeltaLobe::make(); ret.deltaLobeCount = 0; ret.nonDeltaPart = 1.0; ret.vertexIndex = 0; ret.stableBranchID = 0; ret.materialID = 0xFFFFFFFF; ret.throughput = float3(0,0,0); ret.volumeAbsorption = 0; ret.padding0 = 0; ret.padding1 = 0; ret.padding2 = 0; ret.worldPos = float3(0,0,0); return ret; }
    static DeltaTreeVizPathVertex make( uint vertexIndex, uint stableBranchID, uint materialID, float3 throughput, float volumeAbsorption, float3 worldPos, bool isDominant ) 
    { DeltaTreeVizPathVertex ret = DeltaTreeVizPathVertex::make(); ret.vertexIndex = vertexIndex; ret.stableBranchID = stableBranchID; ret.materialID = materialID; ret.throughput = throughput; ret.volumeAbsorption = volumeAbsorption; ret.worldPos = worldPos; ret.isDominant = isDominant; return ret; }
#endif

    uint            getParentLobe()
#if defined(__cplusplus) // shader only!
        const               
#endif
    {
        return StablePlanesGetParentLobeID(stableBranchID);
    }
};

static const float      cDeltaTreeVizThpIgnoreThreshold = 0.001f;       // ignore delta paths with less than 0.1% potential contribution
static const uint       cDeltaTreeVizMaxStackSize = 256;
static const uint       cDeltaTreeVizMaxVertices  = 256;

struct DeltaTreeVizHeader
{
    uint2           pixelPos;
     int            nodeCount;
    uint            sampleIndex;

     int            searchStackCount;
    uint            dominantStablePlaneIndex;
    uint            padding1;
    uint            padding2;

    uint4           stableBranchIDs;

//#if !defined(__cplusplus) // shader only!
    static DeltaTreeVizHeader make()            { DeltaTreeVizHeader ret; ret.pixelPos = uint2(0,0); ret.nodeCount = 0; ret.sampleIndex = 0; ret.searchStackCount = 0; ret.dominantStablePlaneIndex = 0xFFFFFFFF; ret.padding1 = 0; ret.padding2 = 0; ret.stableBranchIDs = uint4(cStablePlaneInvalidBranchID,cStablePlaneInvalidBranchID,cStablePlaneInvalidBranchID,cStablePlaneInvalidBranchID); return ret; }
//#endif
};

//padding to multiple pf float4
struct DebugFeedbackStruct
{
    float4 debugPrint[MAX_DEBUG_PRINT_SLOTS];

	int lineVertexCount;
    int pickedMaterialID;
    int padding1;
    int padding2;

    DeltaTreeVizHeader deltaPathTree;
};

struct DebugLineStruct
{
	float4 pos;
	float4 col;
};

#if !defined(__cplusplus) || defined(__INTELLISENSE__)

struct DebugContext
{
#if ENABLE_DEBUG_VIZUALISATION
    RWStructuredBuffer<DebugFeedbackStruct> feedbackBufferUAV;
#if ENABLE_DEBUG_LINES_VIZUALISATION
    RWStructuredBuffer<DebugLineStruct>     debugLinesBufferUAV;
#endif
    RWStructuredBuffer<DeltaTreeVizPathVertex> deltaPathTreeUAV;
    RWStructuredBuffer<PathPayload>         deltaPathSearchStackUAV;
    RWTexture2D<float4>                     debugVizOutput;
#endif

    DebugConstants                          constants;
    uint2                                   pixelPos;
    bool                                    isDebugPixel;

    void Init(const uint2 _pixelPos, const DebugConstants _constants, uniform RWStructuredBuffer<DebugFeedbackStruct> _feedbackBufferUAV, uniform RWStructuredBuffer<DebugLineStruct> _debugLinesBufferUAV, RWStructuredBuffer<DeltaTreeVizPathVertex> _deltaPathTreeUAV, RWStructuredBuffer<PathPayload> _deltaPathSearchStackUAV, RWTexture2D<float4> _debugVizOutput)
    {
        constants           = _constants;
        pixelPos            = _pixelPos;
        isDebugPixel        = all( pixelPos == uint2(constants.pickX, constants.pickY) );
#if ENABLE_DEBUG_VIZUALISATION
        feedbackBufferUAV   = _feedbackBufferUAV;
#if ENABLE_DEBUG_LINES_VIZUALISATION
        debugLinesBufferUAV = _debugLinesBufferUAV;
#endif
        deltaPathTreeUAV    = _deltaPathTreeUAV;
        debugVizOutput      = _debugVizOutput;
        deltaPathSearchStackUAV = _deltaPathSearchStackUAV;
#endif
    }

    void DrawDebugViz( uint2 pixelPos, float4 colorWithAlpha )   
    { 
#if ENABLE_DEBUG_VIZUALISATION
        debugVizOutput[pixelPos] = colorWithAlpha; 
#endif
    }
    void DrawDebugViz( float4 colorWithAlpha )   { DrawDebugViz(pixelPos, colorWithAlpha);  }

    bool IsDebugPixel() { return isDebugPixel; }

    // Returns 0 when debug lines are disabled
    float LineScale()   { return constants.debugLineScale; }

    // should call this before fist used in frame
    void Reset(uint sampleIndex)
    {
#if ENABLE_DEBUG_VIZUALISATION
        if (isDebugPixel)
        {
            feedbackBufferUAV[0].lineVertexCount = 0;
            feedbackBufferUAV[0].deltaPathTree = DeltaTreeVizHeader::make(); 
            feedbackBufferUAV[0].deltaPathTree.pixelPos = pixelPos;
            feedbackBufferUAV[0].deltaPathTree.sampleIndex = sampleIndex;
            for (int i = 0; i < MAX_DEBUG_PRINT_SLOTS; i++)
                Print(i, float4( -1, -1, -1, -1 ));
        }
#endif
    }

    void DrawLine( float3 start, float3 stop, float4 col1, float4 col2 )
    {
        if ( LineScale()!=0 )   // early out when only debug lines (but not other debugging viz) disabled at runtime
        {
#if ENABLE_DEBUG_VIZUALISATION && ENABLE_DEBUG_LINES_VIZUALISATION
            uint lineVertexCount;
            InterlockedAdd(feedbackBufferUAV[0].lineVertexCount, 2, lineVertexCount);
            if ((lineVertexCount + 1) < MAX_DEBUG_LINES)
            {
                debugLinesBufferUAV[lineVertexCount].pos = float4(start, 1);
                debugLinesBufferUAV[lineVertexCount].col = col1;
                debugLinesBufferUAV[lineVertexCount + 1].pos = float4(stop, 1);
                debugLinesBufferUAV[lineVertexCount + 1].col = col2;
            }
            else
                InterlockedAdd(feedbackBufferUAV[0].lineVertexCount, -2);
#endif
        }
    }

    void DrawLine( float3 start, float3 stop, float3 col1, float3 col2 )
    {
        DrawLine( start, stop, float4(col1, 1), float4(col2, 1) );
    }

    void Print( uint slot, float4 val )
    {
#if ENABLE_DEBUG_VIZUALISATION
        if ( slot < MAX_DEBUG_PRINT_SLOTS )
            feedbackBufferUAV[0].debugPrint[slot] = val;
#endif
    }
    void Print( uint slot, float3 val )     { Print( slot, float4( val, 0 ) ); }
    void Print( uint slot, float2 val )     { Print( slot, float4( val, 0, 0 ) ); }
    void Print( uint slot, float val )      { Print( slot, float4( val, 0, 0, 0 ) ); }
    void Print( uint slot, float val0, float val1 )                         { Print( slot, float4( val0, val1, 0, 0 ) ); }
    void Print( uint slot, float val0, float val1, float val2 )             { Print( slot, float4( val0, val1, val2, 0 ) ); }
    void Print( uint slot, float val0, float val1, float val2, float val3 ) { Print( slot, float4( val0, val1, val2, val3 ) ); }

    void SetPickedMaterial(uint materialID)
    {
#if ENABLE_DEBUG_VIZUALISATION
        feedbackBufferUAV[0].pickedMaterialID = materialID;
#endif
    }

#if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION
    // order of addition is important to be able to reconstruct & visualize the tree!
    bool DeltaTreeVertexAdd(DeltaTreeVizPathVertex vert)
    {
#if ENABLE_DEBUG_VIZUALISATION
        int index;
        InterlockedAdd( feedbackBufferUAV[0].deltaPathTree.nodeCount, 1, index );
        if (index >= cDeltaTreeVizMaxVertices)
        {
            InterlockedAdd( feedbackBufferUAV[0].deltaPathTree.nodeCount, -1 );
            return false;
        }
        deltaPathTreeUAV[index] = vert;
        return true;
#else
        return false;
#endif
    }
    bool DeltaSearchStackPush(PathPayload node)
    {
#if ENABLE_DEBUG_VIZUALISATION
        int index;
        InterlockedAdd( feedbackBufferUAV[0].deltaPathTree.searchStackCount, 1, index );
        if (index >= cDeltaTreeVizMaxStackSize)
        {
            InterlockedAdd( feedbackBufferUAV[0].deltaPathTree.searchStackCount, -1 );
            return false;
        }
        deltaPathSearchStackUAV[index] = node;
        return true;
#else
        return false;
#endif
    }
    bool DeltaSearchStackPop(inout PathPayload element) // inout because we're leaving contents unmodified in case of empty stack
    {
#if ENABLE_DEBUG_VIZUALISATION
        PathPayload node;
        int count;
        InterlockedAdd( feedbackBufferUAV[0].deltaPathTree.searchStackCount, -1, count );
        if (count <= 0)
            return false;
        element = deltaPathSearchStackUAV[count-1];
        return true;
#else
        return false;
#endif
    }
    void DeltaTreeStoreStablePlaneID(uint stablePlaneIndex, uint stableBranchID)
    {
#if ENABLE_DEBUG_VIZUALISATION
        feedbackBufferUAV[0].deltaPathTree.stableBranchIDs[stablePlaneIndex] = stableBranchID;
#endif
    }
    void DeltaTreeStoreDominantStablePlaneIndex(uint dominantStablePlaneIndex)
    {
#if ENABLE_DEBUG_VIZUALISATION
        feedbackBufferUAV[0].deltaPathTree.dominantStablePlaneIndex = dominantStablePlaneIndex;
#endif
    }
#endif // #if ENABLE_DEBUG_DELTA_TREE_VIZUALISATION

#if ENABLE_DEBUG_VIZUALISATION
    void StablePlanesDebugViz(StablePlanesContext stablePlanes)
    {
        if( (constants.debugViewType < (int)DebugViewType::ImagePlaneRayLength || constants.debugViewType > (int)DebugViewType::StablePlaneSpecHitDist) && constants.debugViewType != (int)DebugViewType::StableRadiance )
            return;

        float4 outColor = 0;
        
        // debug viz section
        {
            uint stableBranchID = cStablePlaneInvalidBranchID; 
        
            uint3 displayPos = uint3(pixelPos, 0);

            // split only valid for per-plane data
            if ( constants.debugViewType >= (int)DebugViewType::StablePlaneVirtualRayLength && constants.debugViewType <= (int)DebugViewType::StablePlaneSpecHitDist )
                displayPos = StablePlaneDebugVizFourWaySplitCoord(constants.debugViewStablePlaneIndex, pixelPos, uint2(constants.imageWidth, constants.imageHeight));
            if ( displayPos.z < cStablePlaneCount ) 
                stableBranchID = stablePlanes.GetBranchID(displayPos.xy, displayPos.z);

            if (stableBranchID == cStablePlaneInvalidBranchID)
            {
                outColor = float4( ((pixelPos.x+pixelPos.y)%2).xxx, 1 );   // just dump checkerboard pattern
            }
            else
            {
                StablePlane sp = stablePlanes.LoadStablePlane(displayPos.xy, displayPos.z);

                PackedHitInfo packedHitInfo; float3 rayDir; uint vertexIndex; uint SERSortKey; float sceneLength; float3 thp; float3 motionVectors;
                StablePlanesContext::UnpackStablePlane( sp, vertexIndex, packedHitInfo, SERSortKey, rayDir, sceneLength, thp, motionVectors );

                float rangeVizScale = 1.0 / 100.0;   // repeat pattern every 10 units

                float3 diffBSDFEstimate; float3 specBSDFEstimate;
                UnpackTwoFp32ToFp16( sp.DenoiserPackedBSDFEstimate, diffBSDFEstimate, specBSDFEstimate );

                float firstHitRayLength = stablePlanes.LoadFirstHitRayLength(pixelPos);

                float3 firstHitRayLengthCol = (firstHitRayLength>=kMaxRayTravel*0.99)?float3(0,0.5,1):(frac(rangeVizScale*firstHitRayLength).xxx);
                float3 sceneLengthCol = (sceneLength>=kMaxRayTravel*0.99)?float3(0,0.5,1):(frac(rangeVizScale*sceneLength).xxx);

                float4 diffht, specht; UnpackTwoFp32ToFp16(sp.DenoiserPackedRadianceHitDist, diffht, specht);

                uint dominantSP = stablePlanes.LoadDominantIndexCenter();

                float3 stableRadiance = stablePlanes.LoadStableRadiance(pixelPos);

                switch (constants.debugViewType)
                {
                case ((int)DebugViewType::ImagePlaneRayLength):                 outColor = float4( firstHitRayLengthCol, 1 ); break;
                case ((int)DebugViewType::DominantStablePlaneIndex):            outColor = float4( StablePlaneDebugVizColor(dominantSP), 1); break;
                case ((int)DebugViewType::StablePlaneVirtualRayLength):         outColor = float4( sceneLengthCol, 1 ); break;
                case ((int)DebugViewType::StablePlaneMotionVectors):            outColor = float4( 0.5 + motionVectors.xyz * float3(0.2, 0.2, 10), 1 ); break;
                case ((int)DebugViewType::StablePlaneNormals):                  outColor = float4( DbgShowNormalSRGB(sp.DenoiserNormalRoughness.xyz), 1 ); break;
                case ((int)DebugViewType::StablePlaneRoughness):                outColor = float4( sp.DenoiserNormalRoughness.www, 1 ); break;
                case ((int)DebugViewType::StablePlaneDiffBSDFEstimate):         outColor = float4( diffBSDFEstimate, 1 ); break;
                case ((int)DebugViewType::StablePlaneDiffRadiance):             outColor = float4( diffht.rgb / diffBSDFEstimate, 1 ); break;
                case ((int)DebugViewType::StablePlaneDiffHitDist):              outColor = float4( rangeVizScale*diffht.www, 1 ); break;
                case ((int)DebugViewType::StablePlaneSpecBSDFEstimate):         outColor = float4( specBSDFEstimate, 1 ); break;
                case ((int)DebugViewType::StablePlaneSpecRadiance):             outColor = float4( specht.rgb / specBSDFEstimate, 1 ); break;
                case ((int)DebugViewType::StablePlaneSpecHitDist):              outColor = float4( rangeVizScale*specht.www, 1 ); break;
                case ((int)DebugViewType::StableRadiance):                      outColor = float4( sqrt(stablePlanes.LoadStableRadiance(pixelPos)), 1 ); break; // note: sqrt there is a cheap debug tonemapper :D
                }
            }
        }

        // mouse cursor info
        {
            uint2 mousePos = float2(constants.mouseX, constants.mouseY);
            uint3 mouseSrcPos = uint3(pixelPos, 0);
            if ( constants.debugViewType >= (int)DebugViewType::StablePlaneVirtualRayLength && constants.debugViewType <= (int)DebugViewType::StablePlaneSpecHitDist )
                mouseSrcPos = StablePlaneDebugVizFourWaySplitCoord(constants.debugViewStablePlaneIndex, mousePos, uint2(constants.imageWidth, constants.imageHeight));

            uint stableBranchID = cStablePlaneInvalidBranchID; 
            if ( mouseSrcPos.z < cStablePlaneCount ) 
                stableBranchID = stablePlanes.GetBranchID(mouseSrcPos.xy, mouseSrcPos.z);

            if (stableBranchID != cStablePlaneInvalidBranchID)
            {
                int detailInfoDigits = 0;
                float4 detailInfoValue = 0;
                StablePlane sp = stablePlanes.LoadStablePlane(mouseSrcPos.xy, mouseSrcPos.z);
                float4 diffht, specht; UnpackTwoFp32ToFp16(sp.DenoiserPackedRadianceHitDist, diffht, specht);

                if ((constants.debugViewType == (int)DebugViewType::StablePlaneDiffHitDist) || (constants.debugViewType == (int)DebugViewType::StablePlaneSpecHitDist))
                {
                    detailInfoDigits = 4;
                    detailInfoValue = (constants.debugViewType == (int)DebugViewType::StablePlaneDiffHitDist)?(diffht):(specht);
                }
                const float3 cols[4] = {float3(1,0.2,0.2), float3(0.2,1.0,0.2), float3(0.2,0.2,1.0), float3(1,1,0.2) };
                for( int d = 0; d < detailInfoDigits; d++)
                {
                    int digits = 10;
                    float2 fontSize = float2(8.0, 15.0); float lineSize = fontSize.x*(digits+2) ;
                    float fontValue = ShaderDrawFloat(pixelPos, mousePos + float2(lineSize*(d-detailInfoDigits*0.5-0.5), -fontSize.y * 2), fontSize, detailInfoValue[d], digits, 4);

                    outColor = lerp( outColor, float4(cols[d], 1), fontValue);
                }
                outColor = lerp( outColor, float4(1, 0, 1, 1), ShaderDrawCrosshair(pixelPos, mousePos, 10, 0.5) );
            }
        }

        if (outColor.w>0)
            DrawDebugViz( outColor );

    }
#endif // #if ENABLE_DEBUG_VIZUALISATION

};

#endif


#endif // __PATH_TRACER_DEBUG_HLSLI__
