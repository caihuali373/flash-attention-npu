/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_DQKV_HPP
#define FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_DQKV_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

template <
    uint32_t L1A_STAGES_,
    uint32_t L1B_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadTla <
    MmadAscend950FagdQKV<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_
> {
public:
    using DispatchPolicy = MmadAscend950FagdQKV<L1A_STAGES_, L1B_STAGES_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using TileMmad = TileMmad_;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A; // zN for RowMajor A
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B; // zN for RowMajor B
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A; // zN on Ascend950
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B; // nZ

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy::L1BAlignHelper;

    // ColumnMajor-A TileCopy for dk/dv (dS^T / P^T)
    using TileCopyCol = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, layout::ColumnMajor, ElementB, layout::RowMajor, ElementC, layout::RowMajor>;
    using LayoutTagL1ACol = typename TileCopyCol::LayoutTagL1A; // nZ
    using LayoutTagL0ACol = typename TileCopyCol::LayoutTagL0A; // zN
    using CopyL1ToL0ACol = typename TileCopyCol::CopyL1ToL0A;
    using TileMmadCol = Gemm::Tile::TileMmadTla<ArchTag, ElementA, LayoutTagL1ACol>;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!");
    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "Requires Ascend950");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t L0AB_STAGES = DispatchPolicy::L0AB_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L0C_BUF_SIZE = DispatchPolicy::L0C_BUF_SIZE;
    static constexpr uint32_t BASE = DispatchPolicy::BASE;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0AB_STAGES;

    static constexpr uint32_t SLOT_DQ = Ascend950FagL0CLayout::SLOT_DQ;
    static constexpr uint32_t SLOT_DQ_PING = Ascend950FagL0CLayout::SLOT_DQ_PING;
    static constexpr uint32_t SLOT_DK = Ascend950FagL0CLayout::SLOT_DK;
    static constexpr uint32_t SLOT_DV = Ascend950FagL0CLayout::SLOT_DV;

    // L1 partitions (byte): ds / p / k / q / dy, each up to 128x256
    static constexpr uint32_t L1_TILE_MAX = BASE * 256 * sizeof(ElementA);
    static constexpr uint32_t L1_DS_OFFSET = 0;
    static constexpr uint32_t L1_P_OFFSET = L1_TILE_MAX;
    static constexpr uint32_t L1_K_OFFSET = L1_TILE_MAX * 2;
    static constexpr uint32_t L1_Q_OFFSET = L1_TILE_MAX * 3;
    static constexpr uint32_t L1_DY_OFFSET = L1_TILE_MAX * 4;
    static_assert(L1_DY_OFFSET + L1_TILE_MAX <= ArchTag::L1_SIZE, "L1 overflow");
    static_assert(L0C_STAGES * L0C_BUF_SIZE <= ArchTag::L0C_SIZE, "L0C overflow");

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0, uint32_t eventIdStart = 0)
    {
        l1Ds = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1_DS_OFFSET);
        l1P  = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1_P_OFFSET);
        l1K  = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_K_OFFSET);
        l1Q  = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_Q_OFFSET);
        l1Dy = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1_DY_OFFSET);

        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0AEventList[i] = static_cast<int32_t>(i + eventIdStart);
            l0BEventList[i] = static_cast<int32_t>(i + L0AB_STAGES + eventIdStart);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_BUF_SIZE * i);
            l0CEventList[i] = static_cast<int32_t>((i + eventIdStart) % 8);
        }
        for (uint32_t i = 0; i < 5; i++) {
            l1EventList[i] = static_cast<int32_t>((i + eventIdStart) % 8);
        }
        l0AListId = 0;
        l0BListId = 0;
    }

    CATLASS_DEVICE
    ~BlockMmadTla() {}

    CATLASS_DEVICE
    void SetFlag(bool armDkDv = true)
    {
        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[SLOT_DQ]);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[SLOT_DQ_PING]);
        // slots [2,3]: only arm when taking ownership (initDkDv); while holding
        if (armDkDv) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[SLOT_DK]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[SLOT_DV]);
        }
        for (uint32_t i = 0; i < 5; i++) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[i]);
        }
        l0AListId = 0;
        l0BListId = 0;
    }

    CATLASS_DEVICE
    void WaitFlag(bool releaseDkDv = true)
    {
        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[SLOT_DQ]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[SLOT_DQ_PING]);
        if (releaseDkDv) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[SLOT_DK]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[SLOT_DV]);
        }
        for (uint32_t i = 0; i < 5; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[i]);
        }
    }


    template <
        class TensorDs, class TensorK, class TensorQ, class TensorP, class TensorDy,
        class TensorDq, class TensorDk, class TensorDv>
    CATLASS_DEVICE
    void operator()(
        TensorDs& ds, TensorK& k, TensorQ& q, TensorP& p, TensorDy& dy,
        TensorDq& dq, TensorDk& dk, TensorDv& dv,
        GemmCoord const& actualShape,
        bool initDkDv, bool writeDkDv,
        bool enAtomicDq, bool enAtomicDkDv,
        uint32_t valueD = 0)
    {
        uint32_t sqActual = actualShape.m();
        uint32_t dActual = actualShape.n();
        uint32_t skvActual = actualShape.k();
        uint32_t dvActual = (valueD == 0) ? dActual : valueD;
        bool smallHead = (dActual <= BASE && dvActual <= BASE);

        SetFlag(/*armDkDv=*/initDkDv || !smallHead);

        if (smallHead) {
            ComputeSmallHeadDim(ds, k, q, p, dy, dq, dk, dv,
                sqActual, skvActual, dActual, dvActual,
                initDkDv, writeDkDv, enAtomicDq, enAtomicDkDv);
        } else {
            ComputeLargeHeadDim(ds, k, q, p, dy, dq, dk, dv,
                sqActual, skvActual, dActual, dvActual,
                enAtomicDq, enAtomicDkDv);
        }

        WaitFlag(/*releaseDkDv=*/writeDkDv || !smallHead);
    }

protected:
    // -----------------------------------------------------------------
    // D <= 128
    // -----------------------------------------------------------------
    template <
        class TensorDs, class TensorK, class TensorQ, class TensorP, class TensorDy,
        class TensorDq, class TensorDk, class TensorDv>
    CATLASS_DEVICE
    void ComputeSmallHeadDim(
        TensorDs& ds, TensorK& k, TensorQ& q, TensorP& p, TensorDy& dy,
        TensorDq& dq, TensorDk& dk, TensorDv& dv,
        uint32_t sqActual, uint32_t skvActual, uint32_t dActual, uint32_t dvActual,
        bool initDkDv, bool writeDkDv, bool enAtomicDq, bool enAtomicDkDv)
    {
        uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);
        uint32_t dRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dActual);
        uint32_t dvRound = RoundUp<L1BAlignHelper::N_ALIGNED>(dvActual);

        // GM -> L1: dS/K/Q/dY as RowMajor.
        CopyGmToL1Row(l1Ds, ds, sqActual, skvActual, sqRound, skvRound, 0);
        CopyGmToL1B(l1K, k, skvActual, dActual, skvRound, dRound, 2);
        CopyGmToL1B(l1Q, q, sqActual, dActual, sqRound, dRound, 3);
        CopyGmToL1B(l1Dy, dy, sqActual, dvActual, sqRound, dvRound, 4);

        // dq = dS * K
        GemmRow(l1Ds, l1K, l0CTensorList[SLOT_DQ],
            sqRound, dRound, skvRound, sqActual, dActual, skvActual,
            true, 0b11, SLOT_DQ, 0, 2);
        FixpipeTla(dq, l0CTensorList[SLOT_DQ], sqActual, dActual, sqRound, enAtomicDq, SLOT_DQ);

        // dk += dS^T * Q
        CopyGmToL1Col(l1Ds, ds, sqActual, skvActual, sqRound, skvRound, 0);
        GemmColA(l1Ds, l1Q, l0CTensorList[SLOT_DK],
            skvRound, dRound, sqRound, skvActual, dActual, sqActual,
            initDkDv, writeDkDv ? 0b11 : 0b10, SLOT_DK, 0, 3);
        if (writeDkDv) {
            FixpipeTla(dk, l0CTensorList[SLOT_DK], skvActual, dActual, skvRound, enAtomicDkDv, SLOT_DK);
        }

        // dv += P^T * dY
        CopyGmToL1Col(l1P, p, sqActual, skvActual, sqRound, skvRound, 1);
        GemmColA(l1P, l1Dy, l0CTensorList[SLOT_DV],
            skvRound, dvRound, sqRound, skvActual, dvActual, sqActual,
            initDkDv, writeDkDv ? 0b11 : 0b10, SLOT_DV, 1, 4);
        if (writeDkDv) {
            FixpipeTla(dv, l0CTensorList[SLOT_DV], skvActual, dvActual, skvRound, enAtomicDkDv, SLOT_DV);
        }
    }

    // -----------------------------------------------------------------
    // D > 128
    // -----------------------------------------------------------------
    template <
        class TensorDs, class TensorK, class TensorQ, class TensorP, class TensorDy,
        class TensorDq, class TensorDk, class TensorDv>
    CATLASS_DEVICE
    void ComputeLargeHeadDim(
        TensorDs& ds, TensorK& k, TensorQ& q, TensorP& p, TensorDy& dy,
        TensorDq& dq, TensorDk& dk, TensorDv& dv,
        uint32_t sqActual, uint32_t skvActual, uint32_t dActual, uint32_t dvActual,
        bool enAtomicDq, bool enAtomicDkDv)
    {
        uint32_t sqRound = RoundUp<L1AAlignHelper::M_ALIGNED>(sqActual);
        uint32_t skvRound = RoundUp<L1AAlignHelper::K_ALIGNED>(skvActual);

        CopyGmToL1Row(l1Ds, ds, sqActual, skvActual, sqRound, skvRound, 0);

        uint32_t dLoops = CeilDiv(dActual, BASE);
        uint32_t dvLoops = CeilDiv(dvActual, BASE);
        uint32_t maxLoops = dLoops > dvLoops ? dLoops : dvLoops;

        for (uint32_t dIdx = 0; dIdx < maxLoops; dIdx++) {
            uint32_t nAct = (dIdx < dLoops)
                ? ((dIdx + 1 < dLoops) ? BASE : (dActual - dIdx * BASE)) : 0;
            uint32_t nvAct = (dIdx < dvLoops)
                ? ((dIdx + 1 < dvLoops) ? BASE : (dvActual - dIdx * BASE)) : 0;
            uint32_t nRound = nAct ? RoundUp<L1BAlignHelper::N_ALIGNED>(nAct) : 0;
            uint32_t nvRound = nvAct ? RoundUp<L1BAlignHelper::N_ALIGNED>(nvAct) : 0;
            uint32_t slotDq = (dIdx & 1) ? SLOT_DQ_PING : SLOT_DQ;

            if (nAct > 0) {
                uint32_t nOff = dIdx * BASE;
                auto kTile = GetTile(k, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));
                auto qTile = GetTile(q, tla::MakeCoord(0u, nOff), tla::MakeShape(sqActual, nAct));
                auto dqTile = GetTile(dq, tla::MakeCoord(0u, nOff), tla::MakeShape(sqActual, nAct));
                auto dkTile = GetTile(dk, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nAct));

                CopyGmToL1B(l1K, kTile, skvActual, nAct, skvRound, nRound, 2);
                CopyGmToL1B(l1Q, qTile, sqActual, nAct, sqRound, nRound, 3);

                GemmRow(l1Ds, l1K, l0CTensorList[slotDq],
                    sqRound, nRound, skvRound, sqActual, nAct, skvActual,
                    true, 0b11, slotDq, 0, 2);
                FixpipeTla(dqTile, l0CTensorList[slotDq], sqActual, nAct, sqRound, enAtomicDq, slotDq);

                CopyGmToL1Col(l1Ds, ds, sqActual, skvActual, sqRound, skvRound, 0);
                GemmColA(l1Ds, l1Q, l0CTensorList[SLOT_DK],
                    skvRound, nRound, sqRound, skvActual, nAct, sqActual,
                    true, 0b11, SLOT_DK, 0, 3);
                FixpipeTla(dkTile, l0CTensorList[SLOT_DK], skvActual, nAct, skvRound, enAtomicDkDv, SLOT_DK);

                if (dIdx + 1 < dLoops) {
                    CopyGmToL1Row(l1Ds, ds, sqActual, skvActual, sqRound, skvRound, 0);
                }
            }

            if (nvAct > 0) {
                uint32_t nOff = dIdx * BASE;
                auto dyTile = GetTile(dy, tla::MakeCoord(0u, nOff), tla::MakeShape(sqActual, nvAct));
                auto dvTile = GetTile(dv, tla::MakeCoord(0u, nOff), tla::MakeShape(skvActual, nvAct));

                CopyGmToL1B(l1Dy, dyTile, sqActual, nvAct, sqRound, nvRound, 4);
                CopyGmToL1Col(l1P, p, sqActual, skvActual, sqRound, skvRound, 1);
                GemmColA(l1P, l1Dy, l0CTensorList[SLOT_DV],
                    skvRound, nvRound, sqRound, skvActual, nvAct, sqActual,
                    true, 0b11, SLOT_DV, 1, 4);
                FixpipeTla(dvTile, l0CTensorList[SLOT_DV], skvActual, nvAct, skvRound, enAtomicDkDv, SLOT_DV);
            }
        }
    }

    // -------------------- TLA helpers --------------------
    template <class TensorGm>
    CATLASS_DEVICE
    void CopyGmToL1Row(
        AscendC::LocalTensor<ElementA> l1Buf, TensorGm& gm,
        uint32_t mActual, uint32_t kActual, uint32_t mRound, uint32_t kRound, uint32_t eventIdx)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorGm>;
        CopyGmToL1A copyGmToL1A;

        auto layoutL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kRound);
        auto tensorL1 = tla::MakeTensor(l1Buf, layoutL1, Arch::PositionL1{});
        auto tensorTileGm = GetTile(gm, tla::MakeCoord(0u, 0u), tla::MakeShape(mActual, kActual));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[eventIdx]);
        copyGmToL1A(tensorL1, tensorTileGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[eventIdx]);
    }

    /// Reload RowMajor GM [sq, skv] as ColumnMajor [skv, sq] into L1 (nZ)
    template <class TensorGm>
    CATLASS_DEVICE
    void CopyGmToL1Col(
        AscendC::LocalTensor<ElementA> l1Buf, TensorGm& gm,
        uint32_t sqActual, uint32_t skvActual, uint32_t sqRound, uint32_t skvRound, uint32_t eventIdx)
    {
        auto layoutCol = tla::MakeLayout(
            tla::MakeShape(skvActual, sqActual),
            tla::MakeStride(tla::Int<1>{}, static_cast<int64_t>(tla::get<0>(gm.stride()))),
            tla::MakeShape(skvActual, sqActual));
        auto gmCol = tla::MakeTensor(gm.data(), layoutCol, Arch::PositionGM{});

        using CopyGmToL1ACol = typename TileCopyCol::template CopyGmToL1A<decltype(gmCol)>;
        CopyGmToL1ACol copyGmToL1ACol;

        auto layoutL1 = tla::MakeLayout<ElementA, LayoutTagL1ACol>(skvRound, sqRound);
        auto tensorL1 = tla::MakeTensor(l1Buf, layoutL1, Arch::PositionL1{});
        auto tensorTileGm = GetTile(gmCol, tla::MakeCoord(0u, 0u), tla::MakeShape(skvActual, sqActual));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[eventIdx]);
        copyGmToL1ACol(tensorL1, tensorTileGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[eventIdx]);
    }

    template <class TensorGm>
    CATLASS_DEVICE
    void CopyGmToL1B(
        AscendC::LocalTensor<ElementB> l1Buf, TensorGm& gm,
        uint32_t kActual, uint32_t nActual, uint32_t kRound, uint32_t nRound, uint32_t eventIdx)
    {
        using CopyGmToL1BOp = typename TileCopy::template CopyGmToL1B<TensorGm>;
        CopyGmToL1BOp copyGmToL1B;

        auto layoutL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nRound);
        auto tensorL1 = tla::MakeTensor(l1Buf, layoutL1, Arch::PositionL1{});
        auto tensorTileGm = GetTile(gm, tla::MakeCoord(0u, 0u), tla::MakeShape(kActual, nActual));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[eventIdx]);
        copyGmToL1B(tensorL1, tensorTileGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[eventIdx]);
    }

    CATLASS_DEVICE
    void GemmRow(
        AscendC::LocalTensor<ElementA> l1A,
        AscendC::LocalTensor<ElementB> l1B,
        AscendC::LocalTensor<ElementAccumulator> l0C,
        uint32_t mRound, uint32_t nRound, uint32_t kRound,
        uint32_t mActual, uint32_t nActual, uint32_t kActual,
        bool initC, uint8_t unitFlag, uint32_t l0cSlot,
        uint32_t l1AEvent, uint32_t l1BEvent)
    {
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1A>(mRound, kRound);
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nRound);
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kRound, nRound);
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nRound);

        auto tensorL1A = tla::MakeTensor(l1A, layoutAInL1, Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutBInL1, Arch::PositionL1{});
        auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0u, 0u), tla::MakeShape(mRound, kRound));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0u, 0u), tla::MakeShape(kRound, nRound));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1AEvent]);
        copyL1ToL0A(tensorL0A, tileL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1AEvent]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1BEvent]);
        copyL1ToL0B(tensorL0B, tileL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1BEvent]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
        }
        uint32_t mMad = (mActual == 1) ? 2 : mRound;
        uint8_t uf = ENABLE_UNIT_FLAG ? unitFlag : 0;
        tileMmad(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, uf);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
    }

    CATLASS_DEVICE
    void GemmColA(
        AscendC::LocalTensor<ElementA> l1A,
        AscendC::LocalTensor<ElementB> l1B,
        AscendC::LocalTensor<ElementAccumulator> l0C,
        uint32_t mRound, uint32_t nRound, uint32_t kRound,
        uint32_t mActual, uint32_t nActual, uint32_t kActual,
        bool initC, uint8_t unitFlag, uint32_t l0cSlot,
        uint32_t l1AEvent, uint32_t l1BEvent)
    {
        auto layoutAInL1 = tla::MakeLayout<ElementA, LayoutTagL1ACol>(mRound, kRound);
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0ACol>(mRound, kRound);
        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kRound, nRound);
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kRound, nRound);
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nRound);

        auto tensorL1A = tla::MakeTensor(l1A, layoutAInL1, Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutBInL1, Arch::PositionL1{});
        auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0u, 0u), tla::MakeShape(mRound, kRound));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0u, 0u), tla::MakeShape(kRound, nRound));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1AEvent]);
        copyL1ToL0ACol(tensorL0A, tileL1A);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1AEvent]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1EventList[l1BEvent]);
        copyL1ToL0B(tensorL0B, tileL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1EventList[l1BEvent]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
        if (initC) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
        }
        uint32_t mMad = (mActual == 1) ? 2 : mRound;
        uint8_t uf = ENABLE_UNIT_FLAG ? unitFlag : 0;
        tileMmadCol(tensorL0C, tensorL0A, tensorL0B, mMad, nActual, kActual, initC, uf);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        l0AListId = (l0AListId + 1 < L0AB_STAGES) ? (l0AListId + 1) : 0;
        l0BListId = (l0BListId + 1 < L0AB_STAGES) ? (l0BListId + 1) : 0;
    }

    template <class TensorGm>
    CATLASS_DEVICE
    void FixpipeTla(
        TensorGm& gmC,
        AscendC::LocalTensor<ElementAccumulator> l0C,
        uint32_t mActual, uint32_t nActual, uint32_t mRound,
        bool enAtomic, uint32_t l0cSlot)
    {
        auto layoutCInL0 = tla::MakeLayoutL0C(mRound, nActual);
        auto tensorL0C = tla::MakeTensor(l0C, layoutCInL0, Arch::PositionL0C{});
        auto tensorGmTile = GetTile(gmC, tla::MakeCoord(0u, 0u), tla::MakeShape(mActual, nActual));

        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(tensorGmTile)>;
        CopyL0CToDst copyL0CToDst;

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0cSlot]);

        if (enAtomic) {
            AscendC::SetAtomicType<float>();
            copyL0CToDst(tensorGmTile, tensorL0C, 0b11);
            AscendC::SetAtomicNone();
        } else {
            copyL0CToDst(tensorGmTile, tensorL0C, 0b11);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0cSlot]);
    }

    AscendC::LocalTensor<ElementA> l1Ds;
    AscendC::LocalTensor<ElementA> l1P;
    AscendC::LocalTensor<ElementB> l1K;
    AscendC::LocalTensor<ElementB> l1Q;
    AscendC::LocalTensor<ElementB> l1Dy;

    AscendC::LocalTensor<ElementA> l0ATensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];

    int32_t l0AEventList[L0AB_STAGES];
    int32_t l0BEventList[L0AB_STAGES];
    int32_t l0CEventList[L0C_STAGES];
    int32_t l1EventList[5];

    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    TileMmadCol tileMmadCol;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0ACol copyL1ToL0ACol;
    CopyL1ToL0B copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block

#endif // FLASH_ATTN_NPU_ASCEND950_V3_FAG_MMAD_DQKV_HPP
