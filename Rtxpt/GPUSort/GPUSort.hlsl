/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __GPU_SORT_HLSL__
#define __GPU_SORT_HLSL__

//#pragma pack_matrix(row_major)

// This uses older version of AMD's https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/docs/techniques/parallel-sort.md
// that was ported from https://github.com/GameTechDev/XeGTAO/blob/master/Source/Rendering/Shaders/FFX_ParallelSort.h 

#define FFX_HLSL
#include "FFX_ParallelSort.h"

#include <donut/shaders/binding_helpers.hlsli>
#include "../ShaderDebug.hlsli"

//cbuffer FFX_ParallelSortCBConstantsBuffer                   : register( b0 )
//{
//	FFX_ParallelSortCB   g_consts;
//}

struct TinyRootConsts { uint4 params; };
VK_PUSH_CONSTANT ConstantBuffer<TinyRootConsts> g_RootConst : register(b1);

Buffer<uint>      SrcKeys                                   : register(t0);			// The unsorted keys or scan data
Buffer<uint>      SrcIndices                                : register(t1);			// The indices into keys that are to be sorted

RWBuffer<uint>    ScratchBuffer                             : register(u0);			// a.k.a SumTable - the sum table we will write sums to
RWBuffer<uint>    ReducedScratchBuffer                      : register(u1);			// a.k.a. ReduceTable - the reduced sum table we will write sums to
RWBuffer<uint>    DstIndices                                : register(u2);			// The indices into keys that are to be sorted

RWStructuredBuffer<FFX_ParallelSortCB> ConstsUAV            : register(u3);
RWStructuredBuffer<FFX_DispatchIndirectBuffer> DispIndUAV   : register(u4);

[numthreads(1, 1, 1)]
void SetupIndirect( uint index : SV_DispatchThreadID )
{
    const uint MaxThreadGroups = 800;

    //CBuffer[0].NumKeys = NumKeys; // the only thing we init on the cpp side
    const uint NumKeys = ConstsUAV[0].NumKeys;

    uint BlockSize = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;
    uint NumBlocks = (NumKeys + BlockSize - 1) / BlockSize;

    // Figure out data distribution
    uint NumThreadGroupsToRun = MaxThreadGroups;
    uint BlocksPerThreadGroup = (NumBlocks / NumThreadGroupsToRun);
    ConstsUAV[0].NumThreadGroupsWithAdditionalBlocks = NumBlocks % NumThreadGroupsToRun;

    if (NumBlocks < NumThreadGroupsToRun)
    {
        BlocksPerThreadGroup = 1;
        NumThreadGroupsToRun = NumBlocks;
        ConstsUAV[0].NumThreadGroupsWithAdditionalBlocks = 0;
    }

    ConstsUAV[0].NumThreadGroups = NumThreadGroupsToRun;
    ConstsUAV[0].NumBlocksPerThreadGroup = BlocksPerThreadGroup;

    // Calculate the number of thread groups to run for reduction (each thread group can process BlockSize number of entries)
    uint NumReducedThreadGroupsToRun = FFX_PARALLELSORT_SORT_BIN_COUNT * ((BlockSize > NumThreadGroupsToRun) ? 1 : (NumThreadGroupsToRun + BlockSize - 1) / BlockSize);
    ConstsUAV[0].NumReduceThreadgroupPerBin = NumReducedThreadGroupsToRun / FFX_PARALLELSORT_SORT_BIN_COUNT;
    ConstsUAV[0].NumScanValues = NumReducedThreadGroupsToRun;	// The number of reduce thread groups becomes our scan count (as each thread group writes out 1 value that needs scan prefix)

    ConstsUAV[0].NumThreadGroupsToRun = NumThreadGroupsToRun;
    ConstsUAV[0].NumReducedThreadGroupsToRun = NumReducedThreadGroupsToRun;

    // Setup dispatch arguments
    DispIndUAV[0].CountScatterArgs[0] = NumThreadGroupsToRun;
    DispIndUAV[0].CountScatterArgs[1] = 1;
    DispIndUAV[0].CountScatterArgs[2] = 1;
    DispIndUAV[0].CountScatterArgs[3] = 0;

    DispIndUAV[0].ReduceScanArgs[0] = NumReducedThreadGroupsToRun;
    DispIndUAV[0].ReduceScanArgs[1] = 1;
    DispIndUAV[0].ReduceScanArgs[2] = 1;
    DispIndUAV[0].ReduceScanArgs[3] = 0;
}

// FPS Count
[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void Count(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	// Call the uint version of the count part of the algorithm
	FFX_ParallelSort_Count_uint( localID, groupID, /*g_consts*/ConstsUAV[0], g_RootConst.params.x, SrcKeys, ScratchBuffer, SrcIndices );
}

// FPS FirstPassCount (see RTXPT_GPUSORT_FIRST_PASS_INIT_INDICES)
[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void CountIIFP(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	// Call the uint version of the count part of the algorithm
	FFX_ParallelSort_Count_uint( localID, groupID, /*g_consts*/ConstsUAV[0], g_RootConst.params.x, SrcKeys, ScratchBuffer, SrcIndices );
}

// FPS Reduce
[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void CountReduce(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	// Call the reduce part of the algorithm
	FFX_ParallelSort_ReduceCount( localID, groupID, /*g_consts*/ConstsUAV[0], ScratchBuffer, ReducedScratchBuffer );
}

// FPS Scan
[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void ScanPrefix(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	uint BaseIndex = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE * groupID;
	FFX_ParallelSort_ScanPrefix( /*g_consts*/ConstsUAV[0].NumScanValues, localID, groupID, 0, BaseIndex, false, /*g_consts*/ConstsUAV[0], ReducedScratchBuffer, ReducedScratchBuffer, ReducedScratchBuffer );
}
// FPS ScanAdd
[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void ScanAdd(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    FFX_ParallelSortCB consts = /*g_consts*/ConstsUAV[0];

	// When doing adds, we need to access data differently because reduce 
	// has a more specialized access pattern to match optimized count
	// Access needs to be done similarly to reduce
	// Figure out what bin data we are reducing
	uint BinID = groupID / consts.NumReduceThreadgroupPerBin;
	uint BinOffset = BinID * consts.NumThreadGroups;

	// Get the base index for this thread group
	uint BaseIndex = (groupID % consts.NumReduceThreadgroupPerBin) * FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;

	FFX_ParallelSort_ScanPrefix(consts.NumThreadGroups, localID, groupID, BinOffset, BaseIndex, true, consts, ScratchBuffer, ScratchBuffer, ReducedScratchBuffer);
}

// FPS Scatter & FirstPassScatter (see RTXPT_GPUSORT_FIRST_PASS_INIT_INDICES)
[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void Scatter(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	FFX_ParallelSort_Scatter_uint(localID, groupID, /*g_consts*/ConstsUAV[0], g_RootConst.params.x, SrcKeys, /*DstBuffer,*/ ScratchBuffer, SrcIndices, DstIndices );
}

// FPS Scatter & FirstPassScatter (see RTXPT_GPUSORT_FIRST_PASS_INIT_INDICES)
[numthreads(FFX_PARALLELSORT_THREADGROUP_SIZE, 1, 1)]
void ScatterIIFP(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
	FFX_ParallelSort_Scatter_uint(localID, groupID, /*g_consts*/ConstsUAV[0], g_RootConst.params.x, SrcKeys, /*DstBuffer,*/ ScratchBuffer, SrcIndices, DstIndices );
}

[numthreads(1, 1, 1)]
void Validate(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID)
{
    FFX_ParallelSortCB consts = /*g_consts*/ConstsUAV[0];
	
    //SrcKeys   
    //SrcIndices
    uint prevKey = SrcKeys[SrcIndices[0]];
    for( int i = 1; i < consts.NumKeys; i++ )
    {
        uint index = SrcIndices[i];
        uint key = SrcKeys[index];
#ifdef DebugPrint
        if( prevKey > key )
            DebugPrint( "Error at {0} : Index {1}, Key {2}, prev key {3} ", i, index, key, prevKey );
#endif
    }
}

#endif // #ifndef __GPU_SORT_HLSL__