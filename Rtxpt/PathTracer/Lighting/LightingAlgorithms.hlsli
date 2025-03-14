/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __LIGHTING_ALGORITHMS_HLSLI__
#define __LIGHTING_ALGORITHMS_HLSLI__

#include "../Utils/NoiseAndSequences.hlsli"

// This is a slow, reference only 
template<uint _MaxSize/*, typename _Key = uint, typename _Payload = uint*/>
class SortedLightList   
{
    // static_assert( _MaxSize <= 255 ); // note, _MaxSize must always be <= 255
    struct Element
    {
        uint /*_Key*/        Key;       // in our case it's light index in global list
        uint /*_Payload*/    Payload;   // in our case it's the number of lights in local list
    };

    int         Count;
    Element     Items[_MaxSize];

    // int         RootIndex;

    static SortedLightList<_MaxSize> empty()    { SortedLightList<_MaxSize> ret; ret.Count = 0; return ret; }

    void        InsertOrIncCounter(uint key)
    {
        bool notFound = true;
        for( int i = 0; i < Count; i++ )
        {
            if( Items[i].Key == key)
            {
                Items[i].Payload++;
                notFound = false;
                break;
            }
        }
        if( notFound )
        {
            Element n = { key, 1 /*payload*/ };
            Items[Count++] = n;
        }
    }

    void        Swap( int a, int b )
    {
        Element temp = Items[a];
        Items[a] = Items[b];
        Items[b] = temp;
    }

    void        Sort(bool debugTile)
    {
        int n = Count;
        bool swapped;
        do 
        {
            swapped = false;
            for( int i = 1; i < n; i++ )
            {
                if( Items[i - 1].Key > Items[i].Key )
                {
                    Swap( i, i-1 );
                    swapped = true;
                }
            }
            n--;
        } while (swapped);

        #if 0
        for( int i = 1; i < Count; i++ )
            if( Items[i - 1].Key > Items[i].Key )
                {
                    DebugPrint("Sort failed {0}, {1}, {2}", i, Items[i - 1].Key, Items[i].Key );
                    return;
                }
        #endif
    }

    uint        Store(RWTexture3D<uint> storageBuffer, uint2 tileAddress)
    {
        Sort(false);
        uint totalSampleCount = 0;
        for( int i = 0; i < Count; i++ )
        {
            uint lightIndex = Items[i].Key;
            uint thisCount  = Items[i].Payload;

            for( int j = 0; j < thisCount; j++ )
                storageBuffer[uint3(tileAddress.xy, totalSampleCount++)] = PackMiniListLightAndCount( lightIndex, thisCount );
        }
        return totalSampleCount;
    }

    bool            Validate(uint dataToValidate[_MaxSize], bool debugPrint)
    {
        bool sallGoodMan = true;
        Sort(false);
        uint totalSampleCount = 0;
        for( int i = 0; i < Count; i++ )
        {
            uint lightIndex = Items[i].Key;
            uint thisCount  = Items[i].Payload;

            for( int j = 0; j < thisCount; j++ )
                if( dataToValidate[totalSampleCount++] != PackMiniListLightAndCount( lightIndex, thisCount ) )
                {
                    if ( debugPrint )
                        DebugPrint("SortedLightList::Validate failed, index {0}, count {1}", lightIndex, thisCount);
                    sallGoodMan = false;
                }
        }
        if( _MaxSize != totalSampleCount )
        {
            if ( debugPrint )
                DebugPrint("SortedLightList::Validate failed, count different");
            return false;
        }
        return sallGoodMan;
    }
};

template<uint _MaxSize>
struct Bucket
{
    uint Elements[_MaxSize];
    uint Count;

    // note - tuple has encoding that is different from light storage - it's (lightIndex << 24) | counter
    bool HasSmallerUpdateSmaller(inout uint smallestTuple)
    {
        if( Count > 0 )
        {
            uint back = Elements[Count-1];
            if( back < smallestTuple )
            {
                smallestTuple = Elements[Count-1];
                return true;
            }
        }
        return false;
    }
    void Pop()
    {
        Count--;
    }

    bool SortInsertOrIncCounter(uint key)
    {
        const uint bucketCount = Count;
        
        uint insertLocation = 0;

#if 0 // linear search
        for( int i = bucketCount-1; i >= 0; i-- )
        {
            uint iLight = Elements[i] & 0xFFFFFF00;
            if( iLight == key)
            {
                Elements[i] ++;   // no overflow checking
                return true;
            }
            if( iLight > key )
            {
                insertLocation = i+1;
                break;
            }
        }
#else // binary search
        int indexLeft = 0; 
        int indexRight = bucketCount-1;
        while( indexLeft <= indexRight )
        {
            int indexMiddle = (indexLeft+indexRight)/2;
            const uint eMiddle = Elements[indexMiddle];
            uint iLightMiddle = eMiddle & 0xFFFFFF00;

            if( iLightMiddle > key )
                indexLeft = indexMiddle+1;
            else if( iLightMiddle < key )
                indexRight = indexMiddle-1;
            else
            {
                Elements[indexMiddle] = eMiddle + 1;   // no overflow checking, implicit packing
                return true;
            }
        }
        insertLocation = indexLeft;
#endif

        // do we have more space in the list? if not - we have to indicate we couldn't store this one
        if( bucketCount == _MaxSize )
            return false; // overflow!

        // make space (shift everything by 1 to the right)
        for( uint i = bucketCount; i > insertLocation; i-- )
            Elements[i] = Elements[i-1];
            
        // and finally, insert 
        Elements[insertLocation] = key/* | 1*/; // start from 0 to allow for range up to 256
        Count = bucketCount+1;
        return true;
    }
};

// Fixed size hash buckets plus one fallback; they're twice the min size to minimize overflow chances due to hash collisions (can never rule it out!)
template<uint _MaxSize, uint _BucketCount = 16, uint _BucketSize = (_MaxSize*5/3+_BucketCount-1)/_BucketCount+1 >
class HashBucketSortTable
{
    Bucket<_BucketSize>      Buckets[_BucketCount+1];

    static HashBucketSortTable<_MaxSize, _BucketCount, _BucketSize> empty() 
    { 
        HashBucketSortTable<_MaxSize, _BucketCount, _BucketSize> ret; 
        for (uint i = 0; i < _BucketCount+1; i++) 
            ret.Buckets[i].Count = 0; 
        return ret; 
    }

    void        InsertOrIncCounter(uint key, bool debugTile)
    {
        key = key << 8;    // we do our own light/counter encoding
        uint index = Hash32(key+0x9e3779b9) % _BucketCount;
        if (!Buckets[index].SortInsertOrIncCounter(key))
        {
            // if( debugTile )
            //     DebugPrint("Fallback key {0} bucket {1}", key, index);

            if( !Buckets[_BucketCount].SortInsertOrIncCounter(key) ) // primary bucket filled, use fallback bucket
            {
                DebugPrint("Fallback bucket overflow when adding {0}", key);
                // 
                // DebugPrint("--list of counters--");
                // for (uint i = 0; i < _BucketCount+1; i++) 
                //     DebugPrint(" {0} -> {1}", i, Buckets[i].Count);
                // DebugPrint("--------------------");
            }
        }
    }

    uint        Store(RWTexture3D<uint> storageBuffer, uint2 tileAddress, bool debugTile)
    {
        // if( Buckets[_BucketCount].Count > 10)
        //     DebugPrint("Fallback bucket size {0}, max {1}", Buckets[_BucketCount].Count, _BucketSize );

        uint totalSampleCount = 0;
        do 
        {
            int bucketWithSmallest  = -1;
            uint smallestTuple      = 0xFFFFFFFF;
            [unroll] for( uint bucketIndex = 0; bucketIndex < _BucketCount+1; bucketIndex++ )
                if( Buckets[bucketIndex].HasSmallerUpdateSmaller(smallestTuple) )
                    bucketWithSmallest = bucketIndex;
            if( bucketWithSmallest == -1 )
                break;

            Buckets[bucketWithSmallest].Pop();
            uint lightCounter = (smallestTuple & 0xFF)+1;   // we start counting from 0 instead of 1 to better use up the range, so +1 here accounts for that
            uint packedValue = PackMiniListLightAndCount(smallestTuple>>8, lightCounter);
            /*[unroll]*/ for( int j = 0; j < lightCounter; j++ )
                storageBuffer[uint3(tileAddress.xy, totalSampleCount++)] = packedValue;

        } while (true);
        return totalSampleCount;
    }

};

#if 0 // no longer used & maintained
// This is based on a fast, "Left-leaning Red-Black Tree" (self-balancing binary tree) by Robert Sedgewick 2008
// See https://sedgewick.io/wp-content/themes/sedgewick/papers/2008LLRB.pdf
// With additional ideas from https://www.geeksforgeeks.org/red-black-tree-in-cpp/
// Unfinished - see comments & code
template<uint _MaxSize/*, typename _Key = uint, typename _Payload = uint*/>
class SortedLightLLRBTree
{
    #define RED         true
    #define BLACK       false
    #define NULLINDEX   _MaxSize

    // static_assert( _MaxSize <= 255 ); // note, _MaxSize must always be <= 255
    struct Node
    {
        // uint /*_Key*/       Key;            // in our case it's light index in global list
        // uint /*_Payload*/   Payload;        // in our case it's the number of lights in local list

        uint                KeyAndPayload;

        uint                PackedInfo;

        #define LRBT_COLOUR_BIT         0x01000000

        #define LRBT_PARENT_MASK        0x00FF0000
        #define LRBT_PARENT_SHIFT       16
        #define LRBT_RIGHT_MASK         0x0000FF00
        #define LRBT_RIGHT_SHIFT        8
        #define LRBT_LEFT_MASK          0x000000FF
        #define LRBT_LEFT_SHIFT         0

        // index in SortedLightLLRBTree::Nodes
        uint                GetParent()             { return (PackedInfo & LRBT_PARENT_MASK) >> LRBT_PARENT_SHIFT; }
        void                SetParent(uint value)   { PackedInfo = (PackedInfo & ~LRBT_PARENT_MASK) | (value << LRBT_PARENT_SHIFT); }
        uint                GetLeft()               { return (PackedInfo & LRBT_LEFT_MASK) >> LRBT_LEFT_SHIFT; }
        void                SetLeft(uint value)     { PackedInfo = (PackedInfo & ~LRBT_LEFT_MASK) | (value << LRBT_LEFT_SHIFT); }
        uint                GetRight()              { return (PackedInfo & LRBT_RIGHT_MASK) >> LRBT_RIGHT_SHIFT; }
        void                SetRight(uint value)    { PackedInfo = (PackedInfo & ~LRBT_RIGHT_MASK) | (value << LRBT_RIGHT_SHIFT); }
        bool                GetColour()             { return (PackedInfo & LRBT_COLOUR_BIT) != 0; }
        void                SetColour(bool value)   { PackedInfo = (PackedInfo & ~LRBT_COLOUR_BIT) | (value?LRBT_COLOUR_BIT:0); }

        void                SetColourRed()          { PackedInfo = (PackedInfo & ~LRBT_COLOUR_BIT) | LRBT_COLOUR_BIT; }
        void                SetColourBlack()        { PackedInfo = (PackedInfo & ~LRBT_COLOUR_BIT); }

        uint                GetKey()                { return KeyAndPayload & 0x00FFFFFF; }
        uint                GetPayload()            { return (KeyAndPayload >> 24) & 0xFF; }
        void                IncrementPayload()      { KeyAndPayload += 1<<24; }

        static Node make(uint key, uint payload)
        {
            Node ret;
            // ret.Key = key;
            // ret.Payload = payload;
            ret.KeyAndPayload = PackMiniListLightAndCount(key, payload);
            ret.PackedInfo = 0;
            ret.SetParent(NULLINDEX);
            ret.SetLeft(NULLINDEX);
            ret.SetRight(NULLINDEX);
            ret.SetColourRed();
            return ret;
        }


    };

    int             Count;
    Node            Nodes[_MaxSize];

    int             RootIndex;

    static SortedLightLLRBTree<_MaxSize> empty()    
    { 
        SortedLightLLRBTree<_MaxSize> ret; 
        ret.Count = 0; 
        ret.RootIndex = NULLINDEX;
        return ret; 
    }

    // Utility function: Left Rotation
    void rotateLeft( const uint nodeIndex )
    {
        uint childIndex = Nodes[nodeIndex].GetRight();
        uint newRight = Nodes[childIndex].GetLeft();
        Nodes[nodeIndex].SetRight( newRight );
        if (newRight != NULLINDEX)
            Nodes[newRight].SetParent( nodeIndex );
        uint parentIndex = Nodes[nodeIndex].GetParent();
        Nodes[childIndex].SetParent( parentIndex );
        if (parentIndex == NULLINDEX)
            RootIndex = childIndex;
        else if (nodeIndex == Nodes[parentIndex].GetLeft())
            Nodes[parentIndex].SetLeft( childIndex );
        else
            Nodes[parentIndex].SetRight( childIndex );
        Nodes[childIndex].SetLeft( nodeIndex );
        Nodes[nodeIndex].SetParent( childIndex );
    }

    // Utility function: Right Rotation
    void rotateRight( const uint nodeIndex )
    {
        uint childIndex = Nodes[nodeIndex].GetLeft();
        uint newLeft = Nodes[childIndex].GetRight();
        Nodes[nodeIndex].SetLeft( newLeft );
        if (newLeft != NULLINDEX)
            Nodes[newLeft].SetParent( nodeIndex );
        uint parentIndex = Nodes[nodeIndex].GetParent();
        Nodes[childIndex].SetParent( parentIndex );
        if (parentIndex == NULLINDEX)
            RootIndex = childIndex;
        else if (nodeIndex == Nodes[parentIndex].GetLeft())
            Nodes[parentIndex].SetLeft( childIndex );
        else
            Nodes[parentIndex].SetRight( childIndex );
        Nodes[childIndex].SetRight( nodeIndex );
        Nodes[nodeIndex].SetParent( childIndex );
    }

    void SwapColours( uint nodeA, uint nodeB )
    {
    #if 0
        bool colorTmp = Nodes[nodeA].GetColour();
        Nodes[nodeA].SetColour( Nodes[nodeB].GetColour() );
        Nodes[nodeB].SetColour( colorTmp );
    #else
        uint APackedInfo = Nodes[nodeA].PackedInfo;
        uint BPackedInfo = Nodes[nodeB].PackedInfo;
        Nodes[nodeA].PackedInfo = (APackedInfo & ~LRBT_COLOUR_BIT) | (BPackedInfo & LRBT_COLOUR_BIT);
        Nodes[nodeB].PackedInfo = (BPackedInfo & ~LRBT_COLOUR_BIT) | (APackedInfo & LRBT_COLOUR_BIT);
    #endif
    }

    // Utility function: Fixing Insertion Violation
    void fixInsert(inout uint node)
    {
        uint parent = NULLINDEX;
        uint grandparent = NULLINDEX;
        Node tmpN = Nodes[node];
        uint tmpNParent = tmpN.GetParent();
        while (node != RootIndex && tmpN.GetColour() == RED && Nodes[tmpNParent].GetColour() == RED) 
        {
            parent = tmpNParent;
            grandparent = Nodes[parent].GetParent();
            Node tmpGrandparentN = Nodes[grandparent];
            if (parent == tmpGrandparentN.GetLeft()) 
            {
                uint uncle = tmpGrandparentN.GetRight();
                if (uncle != NULLINDEX && Nodes[uncle].GetColour() == RED) 
                {
                    Nodes[grandparent].SetColourRed( );
                    Nodes[parent].SetColourBlack( );
                    Nodes[uncle].SetColourBlack( );
                    node = grandparent;
                }
                else 
                {
                    if (node == Nodes[parent].GetRight()) 
                    {
                        rotateLeft(parent);
                        node = parent;
                        parent = Nodes[node].GetParent();
                    }
                    rotateRight(grandparent);
                    SwapColours(parent, grandparent);
                    node = parent;
                }
            }
            else 
            {
                uint uncle = tmpGrandparentN.GetLeft();
                if (uncle != NULLINDEX && Nodes[uncle].GetColour() == RED) 
                {
                    Nodes[grandparent].SetColourRed( );
                    Nodes[parent].SetColourBlack( );
                    Nodes[uncle].SetColourBlack( );
                    node = grandparent;
                }
                else 
                {
                    if (node == Nodes[parent].GetLeft()) 
                    {
                        rotateRight(parent);
                        node = parent;
                        parent = Nodes[node].GetParent();
                    }
                    rotateLeft(grandparent);
                    SwapColours(parent,grandparent);
                    node = parent;
                }
            }

            tmpN = Nodes[node];
            tmpNParent = tmpN.GetParent();
        }
        Nodes[RootIndex].SetColourBlack( );
    }

    #if 0
    void WriteToList( inout uint2 listOfKeyValuePairs[_MaxSize], inout uint currentListCount, uint rootIndex )
    {
        if (rootIndex == NULLINDEX)
            return;

        uint stackNode[_MaxSize];
        uint stackState[_MaxSize];    // 0 - add left at state0, 1 - print node, 2 - add right at state0 and terminate
        uint stackCount = 1;
        stackNode[0] = rootIndex;
        stackState[0] = 0;

        while( stackCount>0 )
        {
            stackCount--;
            uint currentNodeIndex = stackNode[stackCount];
            uint currentState = stackState[stackCount];
            Node currentNode = Nodes[currentNodeIndex];

            if( currentState == 0 )
            {
                stackState[stackCount]++;
                stackCount++;   // add back our node since it's not finished

                // explore left
                if( currentNode.GetLeft() != NULLINDEX )
                {
                    stackNode[stackCount]   = currentNode.GetLeft();
                    stackState[stackCount]  = 0;
                    stackCount++;
                }
            }
            else
            {
                // print out
                listOfKeyValuePairs[currentListCount++] = uint2( currentNode.GetKey(), currentNode.GetPayload() );

                // explore right
                if( currentNode.GetRight() != NULLINDEX )
                {
                    stackNode[stackCount]   = currentNode.GetRight();
                    stackState[stackCount]  = 0;
                    stackCount++;
                }
            }
        }
    }

    void Print()
    {
        DebugPrint("DEBUG PRINTING LIST {0}", Count);

        uint2 listOfKeyValuePairs[_MaxSize];
        uint count = 0;
        WriteToList(listOfKeyValuePairs, count, RootIndex);

        if( count != Count )
            DebugPrint("WRONG COUNT");

        for( int i = 0; i < count; i++ )
            DebugPrint("  key {0}, count {1}", listOfKeyValuePairs[i].x, listOfKeyValuePairs[i].y);
    }
    #endif

    uint Store(RWTexture3D<uint> storageBuffer, uint2 tileAddress)
    {
        if (RootIndex == NULLINDEX)
            return 0;

        uint totalSampleCount = 0;

        // state is simply: 0 - add left at state0, non-0 - print node add right at state0 and terminate
        #define LRBT_INDEX_MASK         0x000000FF
        #define LRBT_STATE_MASK         0x0000FF00
        #define LRBT_STATE_SHIFT        8

        uint stackNode[_MaxSize];
        uint stackCount = 1;
        stackNode[0] = RootIndex;   // state == 0

        while( stackCount>0 )
        {
            stackCount--;
            uint currentNodeIndex = stackNode[stackCount] & LRBT_INDEX_MASK;
            uint currentState = (stackNode[stackCount] & LRBT_STATE_MASK);// >> LRBT_STATE_SHIFT;
            Node currentNode = Nodes[currentNodeIndex];

            if( currentState == 0 )
            {
                stackNode[stackCount] |= LRBT_STATE_MASK;
                stackCount++;   // add back our node since it's not finished

                // explore left
                if( currentNode.GetLeft() != NULLINDEX )
                {
                    stackNode[stackCount]   = currentNode.GetLeft(); // state == 0
                    stackCount++;
                }
            }
            else
            {
                // print out - need to expand in-place
                uint counter = currentNode.GetPayload();
                for( int j = 0; j < counter; j++ )
                    storageBuffer[uint3(tileAddress.xy, totalSampleCount++)] = currentNode.KeyAndPayload; // same as PackMiniListLightAndCount( currentNode.GetKey(), counter );

                // explore right
                if( currentNode.GetRight() != NULLINDEX )
                {
                    stackNode[stackCount]   = currentNode.GetRight(); // state == 0
                    stackCount++;
                }
            }
        }
        return totalSampleCount;
    }

    // Public function: Insert a value into Red-Black Tree
    void InsertOrIncCounter(uint key)
    //void insert(T key)
    {
        uint parent = NULLINDEX;
        uint current = RootIndex;
        Node parentN;
        while (current != NULLINDEX)
        {
            parent = current;
            parentN = Nodes[parent];
            uint currentKey = parentN.GetKey();
            if (key < currentKey)
                current = parentN.GetLeft();
            else if (key > currentKey)
                current = parentN.GetRight();
            else // they're equal, just increment counter and bail!
            {
                Nodes[current].IncrementPayload();
                return;
            }
        }

        uint node = Count++;
        Nodes[node] = Node::make(key, 1);

        Nodes[node].SetParent( parent );
        if (parent == NULLINDEX)
            RootIndex = node;
        else if (key < parentN.GetKey())
            Nodes[parent].SetLeft( node );
        else
            Nodes[parent].SetRight( node );

// **********************************************************************************************************************************************************************
// **********************************************************************************************************************************************************************
// NOTE: disabling the whole rebalancing for now as the perf is worse; it could be that we simply don't benefit from balancing but it's more likely rebalancing is broken        
#if 0
        fixInsert(node);
#endif
// **********************************************************************************************************************************************************************
// **********************************************************************************************************************************************************************
    }

    #undef RED  
    #undef BLACK
    #undef NULLINDEX
};

#endif

// Expects elements of storageBuffer[tileAddress, x] to be sorted and 'narrowLightCount' is the depth of storageBuffer.
// Note: returning value that consists both of key and counter (which needs unpacking to get actual global index); if not found, RTXPT_INVALID_LIGHT_INDEX returned
inline uint NarrowLightBinarySearch(Texture3D<uint> storageBuffer, const uint2 tileAddress, const uint narrowLightCount, const uint globalLightIndexToFind)
{
    int indexLeft = 0; 
    int indexRight = narrowLightCount-1;

    //uint remainingLoops = some number;
    while( indexLeft <= indexRight ) // && remainingLoops != 0 )
    {
        int indexMiddle = (indexLeft+indexRight)/2;

        uint value = storageBuffer[uint3(tileAddress, indexMiddle)];

        uint keyMiddle = UnpackMiniListLight(value);

        if( keyMiddle < globalLightIndexToFind )
            indexLeft = indexMiddle+1;
        else if( keyMiddle > globalLightIndexToFind )
            indexRight = indexMiddle-1;
        else 
            return value;
        //remainingLoops--;
    }
    return RTXPT_INVALID_LIGHT_INDEX;
}


#endif // #ifndef __LIGHTING_ALGORITHMS_HLSLI__