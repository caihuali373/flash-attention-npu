"""
单独调用 flash_attn_npu 中 flash_attn_varlen_func 的反向接口。

flash_attn_varlen_func 对外是 autograd Function；底层反向算子为
flash_attn_npu.flash_attn_interface._flash_attn_varlen_backward。

本脚本提供两种调用方式：
  1. direct  — 先 _flash_attn_varlen_forward，再单独调 _flash_attn_varlen_backward（不走 autograd）
  2. autograd — flash_attn_varlen_func 前向 + torch.autograd.grad / backward（标准训练路径）

用法示例:
  conda activate lfz_tri
  cd /data/lfz/test/test_/flash-attention-npu_final/tests
  python call_flash_attn_varlen_bwd.py --device 5
  # 脚本会自动 source ，无需手动配置
"""

from __future__ import annotations

import os
import shlex
import sys

CANN_SET_ENV = ""
_ENV_READY_FLAG = "_FLASH_ATTN_CANN_ENV_READY"


def ensure_ascend_env() -> None:
    """通过 set_env.sh 重新拉起进程，确保加载 CANN 9.0+ 的 libascendcl。

    flash_attn_npu_2 依赖 aclrtLaunchKernelWithHostArgs；若 shell 中
    LD_LIBRARY_PATH 混入了旧版 CANN（如 8.3），会在 import 时报 undefined symbol。
    Python 进程启动后修改 LD_LIBRARY_PATH 无效，因此需 re-exec。
    """
    if os.environ.get(_ENV_READY_FLAG) == "1":
        return
    if not os.path.isfile(CANN_SET_ENV):
        return
    cmd = (
        f"source {shlex.quote(CANN_SET_ENV)} && "
        f"export {_ENV_READY_FLAG}=1 && "
        f"exec {shlex.quote(sys.executable)} "
        + " ".join(shlex.quote(arg) for arg in sys.argv)
    )
    os.execvp("bash", ["bash", "-lc", cmd])


ensure_ascend_env()

import argparse

import numpy as np
import torch
import torch_npu

from flash_attn_npu import flash_attn_varlen_func
from flash_attn_npu.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)


def build_cu_seqlens(seqlens_list, device):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int32, device=device)
    for i in range(1, len(seqlens_list) + 1):
        cu[i] = int(sum(seqlens_list[:i]))
    return cu


def maybe_pad_head_dim(tensor, head_size_og):
    if head_size_og % 8 == 0:
        return tensor
    pad = 8 - head_size_og % 8
    return torch.nn.functional.pad(tensor, [0, pad])


def run_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    block_table,
):
    head_size_og = q.size(-1)
    q_pad = maybe_pad_head_dim(q, head_size_og)
    k_pad = maybe_pad_head_dim(k, head_size_og)
    v_pad = maybe_pad_head_dim(v, head_size_og)

    out_padded, softmax_lse, _, rng_state = _flash_attn_varlen_forward(
        q_pad,
        k_pad,
        v_pad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax=False,
        block_table=block_table,
    )
    out = out_padded[..., :head_size_og]
    return out, out_padded, softmax_lse, rng_state, q_pad, k_pad, v_pad


def run_varlen_backward_direct(
    dout,
    q_pad,
    k_pad,
    v_pad,
    out_padded,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    rng_state,
    *,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
):
    """直接调用 _flash_attn_varlen_backward，返回 dq/dk/dv。"""
    head_size_og = dout.size(-1)
    dout_pad = maybe_pad_head_dim(dout, head_size_og)

    dq = torch.zeros_like(q_pad)
    dk = torch.zeros_like(k_pad)
    dv = torch.zeros_like(v_pad)

    _flash_attn_varlen_backward(
        dout_pad,
        q_pad,
        k_pad,
        v_pad,
        out_padded,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        alibi_slopes,
        deterministic,
        rng_state=rng_state,
    )
    torch.npu.synchronize()

    dq = dq[..., :head_size_og]
    dk = dk[..., :head_size_og]
    dv = dv[..., :head_size_og]
    return dq, dk, dv


def run_varlen_backward_autograd(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dout,
    *,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
    block_table,
):
    """通过 flash_attn_varlen_func 的 autograd 路径求反向梯度。"""
    q_g = q.detach().clone().requires_grad_(True)
    k_g = k.detach().clone().requires_grad_(True)
    v_g = v.detach().clone().requires_grad_(True)

    out = flash_attn_varlen_func(
        q_g,
        k_g,
        v_g,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=False,
        block_table=block_table,
    )
    dq, dk, dv = torch.autograd.grad(out, (q_g, k_g, v_g), dout)
    torch.npu.synchronize()
    return dq, dk, dv


def make_inputs(
    device,
    nheads,
    nheads_k,
    headdim,
    seqlens_list,
    dtype,
    seed,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    seqlens_q = np.array(seqlens_list, dtype=np.int64)
    seqlens_k = seqlens_q.copy()
    max_seqlen_q = int(seqlens_q.max())
    max_seqlen_k = int(seqlens_k.max())
    total_q = int(seqlens_q.sum())
    total_k = int(seqlens_k.sum())

    cu_seqlens_q = build_cu_seqlens(seqlens_list, device)
    cu_seqlens_k = build_cu_seqlens(seqlens_list, device)

    limit = 2.0
    q = (limit * (torch.rand(total_q, nheads, headdim) - 0.5)).to(dtype).to(device)
    k = (limit * (torch.rand(total_k, nheads_k, headdim) - 0.5)).to(dtype).to(device)
    v = (limit * (torch.rand(total_k, nheads_k, headdim) - 0.5)).to(dtype).to(device)
    dout = (limit * (torch.rand(total_q, nheads, headdim) - 0.5)).to(dtype).to(device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "dout": dout,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="单独调用 flash_attn_varlen_func 反向接口示例"
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device id")
    parser.add_argument(
        "--mode",
        choices=("direct", "autograd", "both"),
        default="both",
        help="direct=底层反向; autograd=标准 autograd; both=两种都跑并比对",
    )
    parser.add_argument("--nheads", type=int, default=16)
    parser.add_argument("--nheads_k", type=int, default=1)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument(
        "--seq_lens",
        type=str,
        default="512,33,111",
        help="变长 batch 各样本真实长度，逗号分隔",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--dropout_p", type=float, default=0.0)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no_causal", action="store_false", dest="causal")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.npu.set_device(args.device)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    seqlens_list = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    if args.nheads % args.nheads_k != 0:
        raise ValueError("nheads 必须是 nheads_k 的整数倍")

    softmax_scale = args.headdim ** -0.5
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    block_table = None

    inputs = make_inputs(
        device=f"npu:{args.device}",
        nheads=args.nheads,
        nheads_k=args.nheads_k,
        headdim=args.headdim,
        seqlens_list=seqlens_list,
        dtype=dtype,
        seed=args.seed,
    )

    forward_kwargs = dict(
        dropout_p=args.dropout_p,
        softmax_scale=softmax_scale,
        causal=args.causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        block_table=block_table,
    )
    backward_kwargs = dict(
        dropout_p=args.dropout_p,
        softmax_scale=softmax_scale,
        causal=args.causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=args.deterministic,
    )
    autograd_kwargs = dict(
        dropout_p=args.dropout_p,
        softmax_scale=softmax_scale,
        causal=args.causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=args.deterministic,
        block_table=block_table,
    )

    print(f"device={args.device}, mode={args.mode}")
    print(f"q={tuple(inputs['q'].shape)}, k={tuple(inputs['k'].shape)}, dout={tuple(inputs['dout'].shape)}")
    print(f"cu_seqlens_q={inputs['cu_seqlens_q'].tolist()}")
    print(f"max_seqlen_q={inputs['max_seqlen_q']}, max_seqlen_k={inputs['max_seqlen_k']}")

    dq_direct = dk_direct = dv_direct = None
    if args.mode in ("direct", "both"):
        out, out_padded, softmax_lse, rng_state, q_pad, k_pad, v_pad = run_varlen_forward(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["cu_seqlens_q"],
            inputs["cu_seqlens_k"],
            inputs["max_seqlen_q"],
            inputs["max_seqlen_k"],
            **forward_kwargs,
        )
        dq_direct, dk_direct, dv_direct = run_varlen_backward_direct(
            inputs["dout"],
            q_pad,
            k_pad,
            v_pad,
            out_padded,
            softmax_lse,
            inputs["cu_seqlens_q"],
            inputs["cu_seqlens_k"],
            inputs["max_seqlen_q"],
            inputs["max_seqlen_k"],
            rng_state,
            **backward_kwargs,
        )
        print("[direct] backward done")
        print(f"  dq: {tuple(dq_direct.shape)}, dk: {tuple(dk_direct.shape)}, dv: {tuple(dv_direct.shape)}")
        print(f"  dq[0,0,:4]={dq_direct[0, 0, :4].detach().cpu().float().tolist()}")

    dq_auto = dk_auto = dv_auto = None
    if args.mode in ("autograd", "both"):
        dq_auto, dk_auto, dv_auto = run_varlen_backward_autograd(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["cu_seqlens_q"],
            inputs["cu_seqlens_k"],
            inputs["max_seqlen_q"],
            inputs["max_seqlen_k"],
            inputs["dout"],
            **autograd_kwargs,
        )
        print("[autograd] backward done")
        print(f"  dq: {tuple(dq_auto.shape)}, dk: {tuple(dk_auto.shape)}, dv: {tuple(dv_auto.shape)}")
        print(f"  dq[0,0,:4]={dq_auto[0, 0, :4].detach().cpu().float().tolist()}")

    if args.mode == "both":
        dq_close = torch.allclose(dq_direct.float(), dq_auto.float(), rtol=1e-2, atol=1e-2)
        dk_close = torch.allclose(dk_direct.float(), dk_auto.float(), rtol=1e-2, atol=1e-2)
        dv_close = torch.allclose(dv_direct.float(), dv_auto.float(), rtol=1e-2, atol=1e-2)
        print(f"[compare] dq={dq_close}, dk={dk_close}, dv={dv_close}")
        if not (dq_close and dk_close and dv_close):
            raise SystemExit(1)
        print("direct 与 autograd 反向结果一致。")


if __name__ == "__main__":
    main()
