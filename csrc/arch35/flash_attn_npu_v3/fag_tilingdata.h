/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Host/device tiling ABI for the Ascend950 FlashAttention v3 backward path.
 *
 * Keep this structure POD-only.  It is copied byte-for-byte from a CPU tensor
 * to device global memory and will be consumed by the future arch35 FAG
 * kernel.
 */

#ifndef FLASH_ATTN_NPU_ARCH35_V3_FAG_TILINGDATA_H
#define FLASH_ATTN_NPU_ARCH35_V3_FAG_TILINGDATA_H

#include <cstdint>
#include <type_traits>

namespace FAGTiling950 {

constexpr uint32_t FAG_TILING_ABI_VERSION = 1;
constexpr uint64_t GM_ALIGNMENT = 512;
constexpr uint64_t MULTI_CORE_SYNC_BYTES = 64 * 1024;
constexpr uint32_t SOFTMAX_REDUCE_FLOATS = 8;

enum class Layout : uint32_t {
    BSND = 0,
    TND = 1,
};

enum class MaskType : uint32_t {
    NO_MASK = 0,
    CAUSAL = 1,
};

struct FAGInfo {
    Layout layout = Layout::BSND;
    MaskType maskType = MaskType::NO_MASK;
    uint32_t deterministic = 0;

    uint64_t batch = 0;
    uint64_t qSeqlen = 0;
    uint64_t kvSeqlen = 0;
    uint64_t totalQ = 0;
    uint64_t totalKv = 0;
    uint64_t qHeadNum = 0;
    uint64_t kvHeadNum = 0;
    uint64_t qkHeadDim = 0;
    uint64_t vHeadDim = 0;

    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    float scaleValue = 1.0f;
};

struct FAGTilingData {
    uint32_t layout = 0;
    uint32_t maskType = 0;
    uint32_t deterministic = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t usedCoreNum = 0;

    uint64_t ubSize = 0;
    uint64_t batch = 0;
    uint64_t qSeqlen = 0;
    uint64_t kvSeqlen = 0;
    uint64_t totalQ = 0;
    uint64_t totalKv = 0;
    uint64_t qHeadNum = 0;
    uint64_t kvHeadNum = 0;
    uint64_t groupSize = 0;
    uint64_t qkHeadDim = 0;
    uint64_t vHeadDim = 0;
    float scaleValue = 1.0f;
    uint32_t reserved = 0;

    uint32_t qTile = 0;
    uint32_t kvTile = 0;

    uint64_t dqOffset = 0;
    uint64_t dkOffset = 0;
    uint64_t dvOffset = 0;
    uint64_t deltaOffset = 0;
    uint64_t workspaceSize = 0;
};

static_assert(std::is_standard_layout_v<FAGTilingData>,
              "FAGTilingData must have a stable host/device layout");
static_assert(std::is_trivially_copyable_v<FAGTilingData>,
              "FAGTilingData must be byte-copyable to device");

int64_t GetFAGTilingParam(const FAGInfo &info, FAGTilingData &tiling);

}  // namespace FAGTiling950

#endif
