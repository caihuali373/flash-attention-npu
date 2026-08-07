/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend950 FlashAttention v3 backward block dispatch policies.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_BLOCK_H
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_BLOCK_H

#include "catlass/arch/arch.hpp"

#include "fag_common.h"

namespace Catlass::Epilogue {

// V1 epilogue policy. The policy carries compile-time layout and mask choices
// from the kernel entrypoint to the matching BlockEpilogue specialization.
template <FAGTiling950::Layout INPUT_LAYOUT_, bool IS_ATTEN_MASK_>
struct EpilogueAscend950FAGScaledMaskSoftmax {
    using ArchTag = Arch::Ascend950;

    static constexpr FAGTiling950::Layout INPUT_LAYOUT = INPUT_LAYOUT_;
    static constexpr bool IS_ATTEN_MASK = IS_ATTEN_MASK_;
};

// V2 epilogue policy. Layout remains a compile-time property so the future
// implementation can specialize delta addressing for BSND and TND.
template <FAGTiling950::Layout INPUT_LAYOUT_>
struct EpilogueAscend950FAGSubMul {
    using ArchTag = Arch::Ascend950;

    static constexpr FAGTiling950::Layout INPUT_LAYOUT = INPUT_LAYOUT_;
};

}  // namespace Catlass::Epilogue

#endif  // FLASH_ATTN_NPU_ASCEND950_V3_FAG_BLOCK_H
