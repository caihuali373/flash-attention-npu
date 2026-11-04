/*!
 * D64 Cube23 (dq∥dk) — port of ops-transformer basic_modules cube23_d64_op.h
 * for FAGGeneral, isolated to Aligned64 && !IS_DTM.
 *
 * Differences vs basic_modules:
 *   - dS is continuous ElementA in mulWorkSpace (stride = s2CvExtendAlign * 2),
 *     not float-workspace 128×128 packed blocks
 *   - Q load per outer s1-group (not full-km every outer) so s1CvExtend≤512 fits
 *   - L1/L0 from Catlass resource.*; AscendC SetFlag/WaitFlag (no WAIT_FLAG macros)
 *   - dv stays on Catlass BlockMmad3 after this path drains
 *
 * Included inside FlashAttentionScoreGrad.
 */
#ifndef FAG_COMMON_CUBE23_D64_COMPUTE_HPP
#define FAG_COMMON_CUBE23_D64_COMPUTE_HPP

    // Fused dq = dS @ K and dk = dS^T @ Q for one CV tile (D=64).
    __aicore__ inline void Cube23ComputeD64(DBParams &dbParam, int64_t dsBase, int64_t dqOffset, int64_t dkvOffset,
                                            int64_t dqKc, int64_t dkvKc, int64_t s2Size, int64_t qSeqStride,
                                            int64_t kSeqStride, int64_t qGmBase, int64_t kGmBase)
    {
        constexpr int32_t BASE = static_cast<int32_t>(MMAD_BASE_SIZE);
        constexpr int32_t HEAD_DIM = 64;
        constexpr uint32_t SIZE_32 = 32;
        constexpr uint32_t SIZE_64 = 64;
        constexpr uint32_t SIZE_96 = 96;
        constexpr uint32_t SIZE_128 = 128;
        constexpr uint32_t SIZE_256 = 256;
        constexpr uint32_t SIZE_320 = 320;
        constexpr uint32_t SIZE_384 = 384;
        constexpr uint32_t SIZE_448 = 448;
        constexpr uint32_t SIZE_ONE_K = 1024;

        // L1 layout mirrors basic CubeOp D64 Init (byte offsets into 512KB L1).
        AscendC::LocalTensor<ElementA> l1_a_ping =
            resource.l1Buf.template GetBufferByByte<ElementA>(0);
        AscendC::LocalTensor<ElementA> l1_a_pong =
            resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_128 * SIZE_ONE_K);
        AscendC::LocalTensor<ElementA> l1_b1_ping =
            resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_256 * SIZE_ONE_K);
        AscendC::LocalTensor<ElementA> l1_b1_pong =
            resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_320 * SIZE_ONE_K);
        AscendC::LocalTensor<ElementA> l1_b2_ping =
            resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_384 * SIZE_ONE_K);
        AscendC::LocalTensor<ElementA> l1_b2_pong =
            resource.l1Buf.template GetBufferByByte<ElementA>(SIZE_448 * SIZE_ONE_K);

        AscendC::LocalTensor<ElementA> l0_a_ping =
            resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        AscendC::LocalTensor<ElementA> l0_a_pong =
            resource.l0ABuf.template GetBufferByByte<ElementA>(SIZE_32 * SIZE_ONE_K);
        AscendC::LocalTensor<ElementA> l0_b_ping =
            resource.l0BBuf.template GetBufferByByte<ElementA>(0);
        AscendC::LocalTensor<ElementA> l0_b_pong =
            resource.l0BBuf.template GetBufferByByte<ElementA>(SIZE_32 * SIZE_ONE_K);

        // 4×32KB L0C: C1=dq ping/pong, C2=dk ping/pong (basic Cube23 layout).
        AscendC::LocalTensor<float> l0_c1_ping =
            resource.l0CBuf.template GetBufferByByte<float>(0);
        AscendC::LocalTensor<float> l0_c1_pong =
            resource.l0CBuf.template GetBufferByByte<float>(SIZE_32 * SIZE_ONE_K);
        AscendC::LocalTensor<float> l0_c2_ping =
            resource.l0CBuf.template GetBufferByByte<float>(SIZE_64 * SIZE_ONE_K);
        AscendC::LocalTensor<float> l0_c2_pong =
            resource.l0CBuf.template GetBufferByByte<float>(SIZE_96 * SIZE_ONE_K);

        AscendC::Nd2NzParams nd2nz{1, BASE, BASE, 0, BASE, BASE, 1, 0};
        AscendC::LoadData2dParams loadNoT{0, BASE, BASE, 0, 0, false, 0};
        AscendC::LoadData2dParams loadT{0, BASE, BASE, 0, 0, true, 0};
        AscendC::MmadParams madParams{BASE, BASE, BASE, 3, false, true};
        AscendC::FixpipeParamsV220 fixParams{BASE, BASE, BASE, BASE, false};
        fixParams.quantPre = QuantMode_t::NoQuant;
        fixParams.unitFlag = 3;

        const int32_t km = static_cast<int32_t>(dbParam.s1CvExtend);
        const int32_t kn = static_cast<int32_t>(dbParam.s2CvExtend);
        const int32_t km_tail = (km % BASE) == 0 ? BASE : (km % BASE);
        const int32_t kn_tail = (kn % BASE) == 0 ? BASE : (kn % BASE);

        const int32_t s1_loop = (km + BASE - 1) / BASE;
        const int32_t s2_loop = (kn + BASE - 1) / BASE;
        const int32_t outer_loop = (s1_loop + 1) / 2;
        const int32_t outer_tail = s1_loop % 2;

        int32_t L1A_flag = 0;
        int32_t L1B1_flag = 0;
        int32_t L1B2_flag = 0;
        constexpr int32_t L1A_SHIFT = 0;
        constexpr int32_t L1B1_SHIFT = 2;
        constexpr int32_t L1B2_SHIFT = 4;
        constexpr int32_t L0A_PING = 3;
        constexpr int32_t L0A_PONG = 4;
        constexpr int32_t L0B_PING = 5;
        constexpr int32_t L0B_PONG = 6;

        AscendC::SetLoadDataPaddingValue(static_cast<ElementA>(0));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_SHIFT + 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_SHIFT + 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1B1_SHIFT + 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1B1_SHIFT + 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1B2_SHIFT + 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1B2_SHIFT + 1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_PING);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_PONG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0B_PING);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0B_PONG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);

        for (int32_t outer_index = 0; outer_index < outer_loop; outer_index++) {
            const int32_t start_s1Idx = outer_index * 2;
            const int32_t inner_s1Loop =
                (outer_index == outer_loop - 1) && (outer_tail != 0) ? outer_tail : 2;

            // Prefetch this outer's Q rows into L1B2 (dk B operand).
            const int32_t b2_row0 = start_s1Idx * BASE;
            const int32_t b2_rows =
                (start_s1Idx + inner_s1Loop == s1_loop) ? (km - b2_row0) : (inner_s1Loop * BASE);
            const int32_t b2_rows_align = ((b2_rows + static_cast<int32_t>(C0_SIZE) - 1) /
                                          static_cast<int32_t>(C0_SIZE)) *
                                         static_cast<int32_t>(C0_SIZE);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1B2_flag + L1B2_SHIFT);
            AscendC::LocalTensor<ElementA> *l1_b2 =
                L1B2_flag ? &l1_b2_pong : &l1_b2_ping;
            nd2nz.nValue = b2_rows;
            nd2nz.dValue = HEAD_DIM;
            nd2nz.dstNzC0Stride = b2_rows_align;
            nd2nz.srcDValue = qSeqStride;
            AscendC::DataCopy(*l1_b2, queryGm[qGmBase + static_cast<int64_t>(b2_row0) * qSeqStride], nd2nz);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1B2_flag + L1B2_SHIFT);

            for (int32_t s2_idx = 0; s2_idx < s2_loop; s2_idx++) {
                const int32_t L0C2_pp = s2_idx % 2;
                for (int32_t s1_idx = 0; s1_idx < inner_s1Loop; s1_idx++) {
                    const int32_t L0C1_pp = s1_idx % 2;
                    const int32_t row_start = (start_s1Idx + s1_idx) * BASE;
                    const int32_t col_start = s2_idx * BASE;
                    const int32_t s1_blen = (s1_idx == inner_s1Loop - 1)
                                               ? ((start_s1Idx + s1_idx == s1_loop - 1) ? km_tail : BASE)
                                               : BASE;
                    const int32_t s2_blen = (s2_idx == s2_loop - 1) ? kn_tail : BASE;
                    const int32_t s1_balign = ((s1_blen + static_cast<int32_t>(C0_SIZE) - 1) /
                                              static_cast<int32_t>(C0_SIZE)) *
                                             static_cast<int32_t>(C0_SIZE);
                    const int32_t s2_balign = ((s2_blen + static_cast<int32_t>(C0_SIZE) - 1) /
                                              static_cast<int32_t>(C0_SIZE)) *
                                             static_cast<int32_t>(C0_SIZE);

                    ///////////////////////////////// dQ /////////////////////////
                    // dS tile GM→L1A (continuous half workspace)
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_flag + L1A_SHIFT);
                    AscendC::LocalTensor<ElementA> *l1_a = L1A_flag ? &l1_a_pong : &l1_a_ping;
                    nd2nz.nValue = s1_blen;
                    nd2nz.dValue = s2_blen;
                    nd2nz.dstNzC0Stride = s1_balign;
                    nd2nz.srcDValue = s2Size;
                    AscendC::DataCopy(
                        *l1_a,
                        mulWorkSpaceGm[dsBase + static_cast<int64_t>(row_start) * s2Size + col_start],
                        nd2nz);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_flag + L1A_SHIFT);

                    // K tile GM→L1B1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1B1_flag + L1B1_SHIFT);
                    AscendC::LocalTensor<ElementA> *l1_b1 = L1B1_flag ? &l1_b1_pong : &l1_b1_ping;
                    nd2nz.nValue = s2_blen;
                    nd2nz.dValue = HEAD_DIM;
                    nd2nz.srcDValue = kSeqStride;
                    nd2nz.dstNzC0Stride = s2_balign;
                    AscendC::DataCopy(
                        *l1_b1,
                        keyGm[kGmBase + static_cast<int64_t>(col_start) * kSeqStride], nd2nz);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1B1_flag + L1B1_SHIFT);

                    // L1A → L0A ping (no transpose) for dq
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_flag + L1A_SHIFT);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_PING);
                    loadNoT.repeatTimes = static_cast<uint16_t>(s2_balign / static_cast<int32_t>(C0_SIZE));
                    loadNoT.srcStride = static_cast<uint16_t>(s1_balign / static_cast<int32_t>(C0_SIZE));
                    for (int32_t i = 0; i < s1_balign / static_cast<int32_t>(C0_SIZE); i++) {
                        AscendC::LoadData(
                            l0_a_ping[i * s2_balign * static_cast<int32_t>(C0_SIZE)],
                            (*l1_a)[i * static_cast<int32_t>(C0_SIZE) * static_cast<int32_t>(C0_SIZE)], loadNoT);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0A_PING);

                    // L1B1 → L0B ping (transpose) for dq
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1B1_flag + L1B1_SHIFT);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0B_PING);
                    loadT.repeatTimes = static_cast<uint16_t>(HEAD_DIM / static_cast<int32_t>(C0_SIZE));
                    loadT.srcStride = static_cast<uint16_t>(s2_balign / static_cast<int32_t>(C0_SIZE));
                    for (int32_t i = 0; i < s2_balign / static_cast<int32_t>(C0_SIZE); i++) {
                        AscendC::LoadData(
                            l0_b_ping[i * HEAD_DIM * static_cast<int32_t>(C0_SIZE)],
                            (*l1_b1)[i * static_cast<int32_t>(C0_SIZE) * static_cast<int32_t>(C0_SIZE)], loadT);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1B1_flag + L1B1_SHIFT);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0B_PING);

                    // mad dq: (s1,s2) x (s2,D)
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0A_PING);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0B_PING);
                    AscendC::LocalTensor<float> *l0_c1 = L0C1_pp ? &l0_c1_pong : &l0_c1_ping;
                    const uint16_t s1_modify = (s1_blen == 1) ? 2 : static_cast<uint16_t>(s1_blen);
                    const bool l0_c1_init = (s2_idx == 0);
                    const bool l0_c1_last = (s2_idx == (s2_loop - 1));
                    madParams.m = s1_modify;
                    madParams.n = HEAD_DIM;
                    madParams.k = s2_blen;
                    madParams.unitFlag = l0_c1_last ? 3 : 2;
                    madParams.cmatrixInitVal = l0_c1_init;
                    AscendC::Mmad(*l0_c1, l0_a_ping, l0_b_ping, madParams);

                    if (l0_c1_last) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C1_pp);
                        fixParams.mSize = s1_blen;
                        fixParams.nSize = HEAD_DIM;
                        fixParams.srcStride = s1_balign;
                        fixParams.dstStride = static_cast<uint32_t>(dqKc);
                        AscendC::SetAtomicType<float>();
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                            dqWorkSpaceGm[dqOffset + static_cast<int64_t>(row_start) * dqKc], *l0_c1, fixParams);
                        AscendC::SetAtomicNone();
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C1_pp);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_PING);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0B_PING);

                    ///////////////////////////////// dK /////////////////////////
                    // L1A → L0A pong (transpose = dS^T)
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_PONG);
                    const uint32_t s1_c0_loop =
                        static_cast<uint32_t>((s1_blen + static_cast<int32_t>(C0_SIZE) - 1) /
                                              static_cast<int32_t>(C0_SIZE));
                    const uint32_t s2_c0_loop =
                        static_cast<uint32_t>((s2_blen + static_cast<int32_t>(C0_SIZE) - 1) /
                                              static_cast<int32_t>(C0_SIZE));
                    loadT.repeatTimes = static_cast<uint16_t>(s1_c0_loop * s2_c0_loop);
                    loadT.srcStride = 1;
                    AscendC::LoadData(l0_a_pong, (*l1_a), loadT);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_flag + L1A_SHIFT);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0A_PONG);

                    // L1B2 → L0B pong (transpose) for dk
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0B_PONG);
                    if (s1_idx == 0 && s2_idx == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1B2_flag + L1B2_SHIFT);
                    }
                    const int32_t l1_b2_off = s1_idx * BASE * static_cast<int32_t>(C0_SIZE);
                    loadT.repeatTimes = static_cast<uint16_t>(HEAD_DIM / static_cast<int32_t>(C0_SIZE));
                    loadT.srcStride = static_cast<uint16_t>(b2_rows_align / static_cast<int32_t>(C0_SIZE));
                    for (int32_t i = 0; i < s1_balign / static_cast<int32_t>(C0_SIZE); i++) {
                        AscendC::LoadData(
                            l0_b_pong[i * static_cast<int32_t>(C0_SIZE) * HEAD_DIM],
                            (*l1_b2)[l1_b2_off + i * static_cast<int32_t>(C0_SIZE) * static_cast<int32_t>(C0_SIZE)],
                            loadT);
                    }
                    if (s1_idx == inner_s1Loop - 1 && s2_idx == s2_loop - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1B2_flag + L1B2_SHIFT);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0B_PONG);

                    // mad dk: (s2,s1) x (s1,D)
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0A_PONG);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0B_PONG);
                    AscendC::LocalTensor<float> *l0_c2 = L0C2_pp ? &l0_c2_pong : &l0_c2_ping;
                    const uint16_t s2_modify = (s2_blen == 1) ? 2 : static_cast<uint16_t>(s2_blen);
                    const bool l0_c2_init = (s1_idx == 0);
                    const bool l0_c2_last = (s1_idx == inner_s1Loop - 1);
                    AscendC::Mmad(*l0_c2, l0_a_pong, l0_b_pong,
                                  AscendC::MmadParams(s2_modify, HEAD_DIM, s1_blen, l0_c2_last ? 3 : 2, false,
                                                     l0_c2_init));
                    AscendC::PipeBarrier<PIPE_M>();

                    if (l0_c2_last) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C2_pp);
                        fixParams.mSize = s2_blen;
                        fixParams.nSize = HEAD_DIM;
                        fixParams.srcStride = s2_balign;
                        fixParams.dstStride = static_cast<uint32_t>(dkvKc);
                        AscendC::SetAtomicType<float>();
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                            dkWorkSpaceGm[dkvOffset + static_cast<int64_t>(col_start) * dkvKc], *l0_c2, fixParams);
                        AscendC::SetAtomicNone();
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C2_pp);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_PONG);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0B_PONG);

                    L1B1_flag = 1 - L1B1_flag;
                    L1A_flag = 1 - L1A_flag;
                }
            }
            L1B2_flag = 1 - L1B2_flag;
        }

        // Drain Cube23 events before Catlass dv reuses L1/L0/FIX.
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_PING);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0B_PING);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0B_PONG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_SHIFT + 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_SHIFT + 1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1B1_SHIFT + 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1B1_SHIFT + 1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1B2_SHIFT + 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1B2_SHIFT + 1);
    }

    // Aligned64-only: Cube23 dq+dk, then Catlass dv. Does not touch other D paths.
    __aicore__ inline void ComputeMMDqkvCube23D64(DBParams &dbParam, int64_t nextBlockId)
    {
        pingpongIdx = dbParam.taskId % 2;
        const int64_t dqOffset = dbParam.aTensorOffsetCv;
        const int64_t dkvOffset = dbParam.bTensorOffsetCv;
        const int64_t dvOffset = dbParam.bTensorOffsetCv / d * value_d;
        const int64_t dqKc = dbParam.s1Stride;
        const int64_t dkvKc = dbParam.s2Stride;
        const int64_t dvKc = dbParam.s2Stride / d * value_d;
        const int64_t s2_size = dbParam.s2CvExtendAlign * DTYPE_FACTOR;
        const int64_t dsBase = pingpongIdx * cubeBaseMN * 2;
        const int64_t qGmBase = InputOffset(inputStrides.q, dbParam, true);
        const int64_t kGmBase = InputOffset(inputStrides.k, dbParam, false);

        Cube23ComputeD64(dbParam, dsBase, dqOffset, dkvOffset, dqKc, dkvKc, s2_size, inputStrides.q.sequence,
                         inputStrides.k.sequence, qGmBase, kGmBase);

        // dv remains Catlass (dropWorkSpace × dout). eventIdStart=4, disjoint from
        // Cube23's drained 0..6 set — SetFlag re-arms on first use / after Drain.
        (void)nextBlockId;
        if (!blockMmad3Inited) {
            blockMmad3Persist.Init(resource, 0, 4);
            blockMmad3Inited = true;
        } else {
            // Cube23 drained its own events; reset Catlass arm state if any.
            blockMmad3Persist.Drain();
        }

        GemmCoord actualBlockShape3(dbParam.s2CvExtend, value_d, dbParam.s1CvExtend);
        LayoutA3 layoutA3{actualBlockShape3.m(), actualBlockShape3.k(), s2_size};
        LayoutB3 layoutB3{actualBlockShape3.k(), actualBlockShape3.n(), inputStrides.dout.sequence};
        LayoutC3 layoutC3{actualBlockShape3.m(), actualBlockShape3.n(), dvKc};
        blockMmad3Persist(dropWorkSpaceGm[pingpongIdx * cubeBaseMN * 2],
                          dxGm[InputOffset(inputStrides.dout, dbParam, true)], dvWorkSpaceGm[dvOffset], layoutA3,
                          layoutB3, layoutC3, actualBlockShape3, true, true);
    }

#endif // FAG_COMMON_CUBE23_D64_COMPUTE_HPP
