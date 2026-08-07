/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward dS epilogue.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SUB_MUL_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SUB_MUL_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

#include "fag_block.h"

namespace Catlass::Epilogue::Block {

/**
 * V2 dS implementation hook.
 *
 * Call chain:
 *   FlashAttentionV3Bwd950
 *     -> FlashAttentionScoreGrad950
 *     -> ProcessV2Stage
 *     -> this BlockEpilogue::operator()
 *
 * Intended data path:
 *   C2 dP UB + V1 P L1 + delta GM -> dS = P * (dP - delta) -> dS L1.
 *
 * Only the interface and type wiring are provided for now. ProcessV2Stage
 * owns the cross-core ready/return handshake around this implementation.
 */
template <
    FAGTiling950::Layout INPUT_LAYOUT_,
    class ElementDS_,
    class ElementP_,
    class ElementDP_,
    class TilingData_>
class BlockEpilogue<
    EpilogueAscend950FAGSubMul<INPUT_LAYOUT_>,
    ElementDS_,
    ElementP_,
    ElementDP_,
    TilingData_> {
public:
    using DispatchPolicy =
        EpilogueAscend950FAGSubMul<INPUT_LAYOUT_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementDS = ElementDS_;
    using ElementP = ElementP_;
    using ElementDP = ElementDP_;
    using TilingData = TilingData_;

    static constexpr FAGTiling950::Layout INPUT_LAYOUT = INPUT_LAYOUT_;

    CATLASS_DEVICE
    BlockEpilogue() = default;

    CATLASS_DEVICE
    void Init(
        Arch::Resource<ArchTag> &resource,
        GM_ADDR workspace,
        GM_ADDR tiling)
    {
        // TODO: Bind deltaWorkspaceGm from workspace + tiling->deltaOffset,
        // decode the BSND/TND addressing parameters, and allocate V2 UB views.
        (void)resource;
        (void)workspace;
        (void)tiling;
    }

    template <class TensorDP, class TensorP, class TensorDS>
    CATLASS_DEVICE
    void operator()(
        FAGBlockInfo const &block,
        TensorDP const &ubDPTensor,
        TensorP const &l1PTensor,
        TensorDS &l1DSTensor,
        uint32_t subBlockIdx)
    {
        // TODO: Implement the V2 data path in this file:
        //   1. Read this AIV's dP rows from C2 result UB.
        //   2. Copy the matching P rows from L1 to a temporary UB view.
        //   3. Read/broadcast delta for each Q row.
        //   4. Compute dS = P * (dP - delta), then cast if required.
        //   5. Copy this AIV's dS rows to its disjoint slice in dS L1.
        (void)block;
        (void)ubDPTensor;
        (void)l1PTensor;
        (void)l1DSTensor;
        (void)subBlockIdx;
    }
};

}  // namespace Catlass::Epilogue::Block

#endif  // FLASH_ATTN_NPU_ASCEND950_V3_FAG_EPILOGUE_SUB_MUL_HPP
