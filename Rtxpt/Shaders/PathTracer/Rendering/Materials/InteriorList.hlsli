/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __INTERIOR_LIST_HLSLI__ // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define __INTERIOR_LIST_HLSLI__

#include "../../Config.h"    

/** Slots on the interior list have the following bit layout.
    0-27  materialID
    28-31 nestedPriority

    We put the nestedPriority in the highest value bits in order to simplify sorting the list.
    Internally the value 0 is reserved for empty slots.
*/

#ifndef INTERIOR_LIST_SLOT_COUNT
#define INTERIOR_LIST_SLOT_COUNT 2  // changing this requires changes in PathPayload::pack/unpack :)
#endif

struct InteriorList
{
    static const uint kNoMaterial           = 0xffffffff;

    static const uint kMaterialBits         = 28;
    static const uint kNestedPriorityBits   = 4;

    static const uint kMaterialOffset       = 0;
    static const uint kNestedPriorityOffset = kMaterialOffset + kMaterialBits;

    static const uint kMaterialMask         = ((1u << kMaterialBits) - 1u) << kMaterialOffset;
    static const uint kNestedPriorityMask   = ((1u << kNestedPriorityBits) - 1u) << kNestedPriorityOffset;

    static const uint kMaxNestedPriority    = ((1u << kNestedPriorityBits) - 1u);

    #if INTERIOR_LIST_SLOT_COUNT == 2
    uint2 slots;
    #elif INTERIOR_LIST_SLOT_COUNT == 4
    uint4 slots;
    #else
    #error not (yet) implemented
    #endif

    /** Make an active material slot given the material and priority.
        \param[in] materialID Material ID.
        \param[in] nestedPriority Nested priority, 0 is reserved for empty slots.
        \return Returns the encoded slot.
    */
    uint makeSlot(uint materialID, uint nestedPriority)
    {
        return (nestedPriority << kNestedPriorityOffset) | (materialID & kMaterialMask);
    }

    /** Check if a slot is active.
        \param[in] slot Slot value.
        \return Returns true if slot is active.
    */
    bool isSlotActive(uint slot)
    {
        return slot != 0;
    }

    /** Check if the interior list is empty.
        \return Returns true if interior list is empty, false otherwise.
    */
    bool isEmpty()
    {
        return !isSlotActive(slots[0]);
    }

    /** Get the nested priority from a slot.
        \param[in] slot Slot value.
        \param Returns the nested priority or 0 for empty slots.
    */
    uint getSlotNestedPriority(uint slot)
    {
        return slot >> kNestedPriorityOffset;
    }

    /** Get the material ID from a slot.
        \param[in] slot Slot value.
        \return Returns the material ID or 0 for empty slots.
    */
    uint getSlotMaterialID(uint slot)
    {
        return slot & kMaterialMask;
    }

    /** Return current highest nested priority on interior list.
        Because the interior list is sorted by nested priority and empty slots have priority 0,
        we can simple fetch the first slot.
        \return Returns the highest nested priority on the interior list.
    */
    uint getTopNestedPriority()
    {
        return getSlotNestedPriority(slots[0]);
    }

    /** Return the material ID with the highest priority on the interior list.
        \return Returns the material ID or kNoMaterial if interior list is empty.
    */
    uint getTopMaterialID()
    {
        return isSlotActive(slots[0]) ? getSlotMaterialID(slots[0]) : kNoMaterial;
    }

    /** Return the material ID with the 2nd highest priority on the interior list.
        \return Returns the material ID or kNoMaterial if interior list is empty.
    */
    uint getNextMaterialID()
    {
        return isSlotActive(slots[1]) ? getSlotMaterialID(slots[1]) : kNoMaterial;
    }

    /** Check if an intersection with a given surface is a true intersection.
        True intersection occurs if nested priority of intersected mesh is equal or higher
        priority than the highest nested priority on the interior list.
        \param[in] nestedPriority Nested priority of intersected surface, with 0 reserved for the highest possible priority.
        \return Returns true if intersection is a true intersection, false otherwise.
    */
    bool isTrueIntersection(uint nestedPriority)
    {
        // Compare nested priority to current top of stack.
        return nestedPriority == 0 || nestedPriority >= getTopNestedPriority();
    }

    /** Handle an intersection with a given material:
        If material is already on interior list -> remove it.
        If material is not on interior list -> add it.
        \param[in] materialID Material ID of intersected material.
        \param[in] nestedPriority Nested priority of intersected surface, with 0 reserved for the highest possible priority.
        \param[in] entering True if material is entered, false if material is left.
    */
    void handleIntersection(uint materialID, uint nestedPriority, bool entering)
    {
        // Remap priority 0 to the highest priority to allow sorting by high->low priority,
        // and as internally 0 is reserved for empty slots.
        if (nestedPriority == 0) nestedPriority = kMaxNestedPriority;

        // The following loop is manually unrolled below to avoid stack allocation.
#if 0
        for (uint slotIndex = 0; slotIndex < INTERIOR_LIST_SLOT_COUNT; ++slotIndex)
        {
            uint slot = slots[slotIndex];

            // Add new slot to interior list
            if (entering && slot == 0)
            {
                slots[slotIndex] = makeSlot(materialID, nestedPriority);
                break;
            }

            // Remove existing slot from interior list
            if (!entering && isSlotActive(slot) && getSlotMaterialID(slot) == materialID)
            {
                slots[slotIndex] = 0;
                break;
            }
        }
#endif

#if INTERIOR_LIST_SLOT_COUNT >= 1
        if (entering && slots[0] == 0)
        {
            slots[0] = makeSlot(materialID, nestedPriority);
        }
        else if (!entering && isSlotActive(slots[0]) && getSlotMaterialID(slots[0]) == materialID)
        {
            slots[0] = 0;
        }
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 2
        else if (entering && slots[1] == 0)
        {
            slots[1] = makeSlot(materialID, nestedPriority);
        }
        else if (!entering && isSlotActive(slots[1]) && getSlotMaterialID(slots[1]) == materialID)
        {
            slots[1] = 0;
        }
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 3
        else if (entering && slots[2] == 0)
        {
            slots[2] = makeSlot(materialID, nestedPriority);
        }
        else if (!entering && isSlotActive(slots[2]) && getSlotMaterialID(slots[2]) == materialID)
        {
            slots[2] = 0;
        }
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 4
        else if (entering && slots[3] == 0)
        {
            slots[3] = makeSlot(materialID, nestedPriority);
        }
        else if (!entering && isSlotActive(slots[3]) && getSlotMaterialID(slots[3]) == materialID)
        {
            slots[3] = 0;
        }
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 5
#error "intersection not handled for given number of slots"
#endif

        sortSlots();
    }

    /** Sort the interior list by priority.
    */
    void sortSlots()
    {
#define CSWAP(_a, _b)               \
        if (slots[_a] < slots[_b])  \
        {                           \
            uint tmp = slots[_a];   \
            slots[_a] = slots[_b];  \
            slots[_b] = tmp;        \
        }

        // sorting networks: http://pages.ripco.net/~jgamble/nw.html
#if INTERIOR_LIST_SLOT_COUNT == 2
        CSWAP(0, 1)
#elif INTERIOR_LIST_SLOT_COUNT == 3
        CSWAP(0, 1)
        CSWAP(1, 2)
        CSWAP(0, 1)
#elif INTERIOR_LIST_SLOT_COUNT == 4
        CSWAP(0, 1)
        CSWAP(2, 3)
        CSWAP(0, 2)
        CSWAP(1, 3)
        CSWAP(1, 2)
#else
#error "sorting not handled for given number of slots"
#endif

#undef CSWAP
    }
};

#endif // __INTERIOR_LIST_HLSLI__