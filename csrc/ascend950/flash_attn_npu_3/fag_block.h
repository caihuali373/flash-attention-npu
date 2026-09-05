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
template <FAGTiling950::Layout INPUT_LAYOUT_, bool IS_ATTEN_MASK_, bool IS_SOFTCAP_>
struct EpilogueAscend950FAGScaledMaskSoftmax {
    using ArchTag = Arch::Ascend950;

    static constexpr FAGTiling950::Layout INPUT_LAYOUT = INPUT_LAYOUT_;
    static constexpr bool IS_ATTEN_MASK = IS_ATTEN_MASK_;
    static constexpr bool IS_SOFTCAP = IS_SOFTCAP_;
};

// V2 epilogue policy. Layout remains a compile-time property so the future
// implementation can specialize delta addressing for BSND and TND.
template <FAGTiling950::Layout INPUT_LAYOUT_, bool IS_SOFTCAP_>
struct EpilogueAscend950FAGSubMul {
    using ArchTag = Arch::Ascend950;

    static constexpr FAGTiling950::Layout INPUT_LAYOUT = INPUT_LAYOUT_;
    static constexpr bool IS_SOFTCAP = IS_SOFTCAP_;
};

}  // namespace Catlass::Epilogue

namespace Catlass::Gemm {
struct Ascend950FagL0CLayout {
    static constexpr uint32_t L0C_BUF_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SLOT_NUM = 4;
    static constexpr uint32_t SLOT_SDP_0 = 0;
    static constexpr uint32_t SLOT_SDP_1 = 1;
    static constexpr uint32_t SLOT_DK = 2;
    static constexpr uint32_t SLOT_DV = 3;
    static constexpr uint32_t SLOT_DQ = SLOT_SDP_0;
    static constexpr uint32_t SLOT_DQ_PING = SLOT_SDP_1;
    static constexpr uint32_t L0C_SDP_SLOT_NUM = 2;
};

// Shared cube L1 map after P/dS ping-pong (see Mm12L1Offset in fag_kernel.cpp).
// Resident KT/VT/K survive across contiguous s1 of the same (batch,n2,s2).
// WORK_Q / WORK_DY are overwritten every task by C1/C2/C5/C34.
// Layout (tile = 128 x 256 x sizeof(elem), typically 64KB):
//   [0] RES_KT  K^T for S = Q K^T
//   [1] RES_VT  V^T for dP = dY V^T
//   [2] RES_K   K  (RowMajor) for dQ = dS K
//   [3] WORK_Q  Q for C1 / C34
//   [4] WORK_DY dY for C2 / C5
struct Ascend950FagL1Layout {
    static constexpr uint32_t BASE = 128;
    static constexpr uint32_t SLOT_RES_KT = 0;
    static constexpr uint32_t SLOT_RES_VT = 1;
    static constexpr uint32_t SLOT_RES_K = 2;
    static constexpr uint32_t SLOT_WORK_Q = 3;
    static constexpr uint32_t SLOT_WORK_DY = 4;
    static constexpr uint32_t SLOT_COUNT = 5;
};


// Ascend950 / Arch3501 FAG dQKV
// Computes dq=dS*K, dk=dS^T*Q, dv=P^T*dY in one block.
template <uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, bool ENABLE_UNIT_FLAG_ = true>
struct MmadAscend950FagdQKV {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = Ascend950FagL0CLayout::L0C_SLOT_NUM;
    static constexpr uint32_t L0C_BUF_SIZE = Ascend950FagL0CLayout::L0C_BUF_SIZE;
    static constexpr uint32_t BASE = 128;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

// Ascend950 FAG S / dP: S = Q * K^T, dP = dY * V^T.
template <uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, bool ENABLE_UNIT_FLAG_ = true>
struct MmadAscend950FagSdP {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = Ascend950FagL0CLayout::L0C_SDP_SLOT_NUM;
    static constexpr uint32_t L0C_BUF_SIZE = Ascend950FagL0CLayout::L0C_BUF_SIZE;
    static constexpr uint32_t BASE = 128;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

}
// namespace Catlass::Gemm

#endif  // FLASH_ATTN_NPU_ASCEND950_V3_FAG_BLOCK_H
