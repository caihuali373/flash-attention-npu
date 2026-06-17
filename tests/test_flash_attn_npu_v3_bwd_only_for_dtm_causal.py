"""
Golden：npu_fusion_attention 前向 + autograd 反传（参考梯度）
FlashAttention：仅 _flash_attn_backward（不走 autograd、不调 flash_attn_varlen_func）

用法:
  python test_flash_attn_npu_v3_bwd_only.py --device 1 --test_layout BSND --case_file fag_cases_BSND_drop.xlsx
  python test_flash_attn_npu_v3_bwd_only.py --device 1 --test_layout TND --enable_perf --prof_path ./prof_bwd
"""

import argparse
import ast
import gc
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch_npu

from flash_attn_npu_v3.flash_attn_interface import _flash_attn_backward

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
from precision_compare import data_compare

DEFAULT_BSND_XLSX = os.path.join(TESTS_DIR, "fag_cases_BSND_drop.xlsx")
DEFAULT_TND_XLSX = os.path.join(TESTS_DIR, "fag_cases_TND_drop.xlsx")
DEFAULT_PROF_BASE = os.path.join(TESTS_DIR, "prof_bwd")
DETERMINISTIC_REPEAT = 10
PROF_ITERATIONS = 10
RESULT_COLUMNS = [
    "nan_result", "torch_bwd_mem", "tridao_bwd_mem", "compare_result",
    "torch_bwd_kernel_ms", "tridao_bwd_kernel_ms", "torch_prof_path", "tridao_prof_path",
]

np.random.seed(3)
torch.manual_seed(3)


def check_nan():
    torch.npu.synchronize()
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    import acl
    device1 = torch.npu.current_device()
    free_byte, total_byte, ret = acl.rt.get_mem_info(device1)
    x = torch.empty(max(free_byte // 4 - 134217728, 0), device=f"npu:{device1}")
    x.fill_(float("nan"))
    del x
    torch.npu.synchronize()


def setup_npu_device(device_id):
    torch.npu.set_device(device_id)
    print(f"Using NPU device: {device_id}")


def npu_between_runs_cleanup():
    torch.npu.synchronize()
    gc.collect()
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu


def lse_from_golden_max_sum(x_max, x_sum, layout):
    """TND: NT8 -> TN; BSND: BNS8 -> BSN."""
    lse = x_max.to(torch.float32) + torch.log(x_sum.to(torch.float32).clamp_min(1e-20))
    if layout == "tnd":
        if lse.dim() != 3:
            raise ValueError(
                f"TND 期望 x_max/x_sum 为 NT8 (N,T,8)，实际 dim={lse.dim()} shape={tuple(lse.shape)}"
            )
        lse = lse[..., 0].transpose(0, 1)
    else:
        if lse.dim() == 4:
            lse = lse[..., 0].transpose(1, 2)
        elif lse.dim() == 3:
            lse = lse.transpose(1, 2)
        else:
            raise ValueError(
                f"BSND 期望 x_max/x_sum 为 BNS8 或 BNS，实际 dim={lse.dim()} shape={tuple(lse.shape)}"
            )
    return lse.contiguous()


def _dtype_from_str(dtype_str):
    if str(dtype_str).lower() in ("half", "fp16", "float16"):
        return torch.float16
    if str(dtype_str).lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_str}")


def _parse_flag(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes")
    return bool(int(val))


def _parse_list(val):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return list(val)


def _mb_used(max_mem, start_mem):
    return f"{((max_mem - start_mem) / 1024 ** 2):.2f}"


def _reset_mem_stats():
    torch.npu.reset_peak_memory_stats()
    return torch.npu.memory_allocated()


def _sanitize_prof_name(name):
    return re.sub(r"[^\w\-.]+", "_", str(name))


def get_kernel_time_total(file_path):
    if not os.path.exists(file_path):
        print(f"profiler error: {file_path} not exists")
        return 0.0
    df = pd.read_csv(file_path)
    total_kernel_time = df["Duration(us)"].sum()
    return float(total_kernel_time) / 1000.0


def get_kernel_time_forward(file_path):
    if not os.path.exists(file_path):
        print(f"profiler error: {file_path} not exists")
        return 0.0
    df = pd.read_csv(file_path)
    forward_kernel_time = 0.0
    for i in range(len(df.index)):
        op_name = str(df["Name"][i])
        if "aclnnFlashAttentionScore_" in op_name or op_name == "FlashAttentionScore":
            forward_kernel_time += df.iloc[i]["Duration(us)"]
    return forward_kernel_time / 1000.0


def get_kernel_time_backward(file_path):
    if not os.path.exists(file_path):
        print(f"profiler error: {file_path} not exists")
        return 0.0
    df = pd.read_csv(file_path)
    backward_kernel_time = 0.0
    for i in range(len(df.index)):
        op_name = str(df["Name"][i])
        if (
            "aclnnFlashAttentionScoreGrad" in op_name
            or op_name == "FlashAttentionScoreGrad"
            or "FlashAttentionScoreGrad" in op_name
        ):
            backward_kernel_time += df.iloc[i]["Duration(us)"]
    return backward_kernel_time / 1000.0


def get_kernel_time_tridao_bwd(file_path):
    if not os.path.exists(file_path):
        print(f"profiler error: {file_path} not exists")
        return 0.0
    df = pd.read_csv(file_path)
    backward_kernel_time = 0.0
    for i in range(len(df.index)):
        op_name = str(df["Name"][i])
        if (
            "flash_attn" in op_name.lower()
            or "faggeneral" in op_name.lower()
            or "flashattentionscoregrad" in op_name.lower()
            or "_flash_attn_backward" in op_name.lower()
        ):
            backward_kernel_time += df.iloc[i]["Duration(us)"]
    if backward_kernel_time > 0:
        return backward_kernel_time / 1000.0
    return get_kernel_time_total(file_path)


def _kernel_details_csv(prof_path):
    if not os.path.exists(prof_path):
        return None
    sub_folders = [
        name for name in os.listdir(prof_path)
        if os.path.isdir(os.path.join(prof_path, name))
    ]
    if not sub_folders:
        return None
    return os.path.join(
        prof_path, sub_folders[0], "ASCEND_PROFILER_OUTPUT", "kernel_details.csv"
    )


def _parse_profiler_kernel_time(prof_path, parser):
    csv_path = _kernel_details_csv(prof_path)
    if csv_path is None:
        print(f"profiler error: no kernel_details.csv under {prof_path}")
        return 0.0
    kernel_ms = parser(csv_path)
    print(f"profiler {csv_path}: {kernel_ms:.6f} ms")
    return kernel_ms


@contextmanager
def _npu_profiler(prof_path):
    if os.path.exists(prof_path):
        subprocess.run(["rm", "-rf", prof_path], check=False)
    os.makedirs(prof_path, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=1, active=1, repeat=1, skip_first=5
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=experimental_config,
        with_modules=True,
    ) as prof:
        yield prof


def _profile_with_loop(prof_path, step_fn, parser):
    print(f"start profiler: {prof_path}")
    with _npu_profiler(prof_path) as prof:
        for _ in range(PROF_ITERATIONS):
            torch.npu.manual_seed(2)
            step_fn()
            prof.step()
    return _parse_profiler_kernel_time(prof_path, parser)


def _run_nan_check():
    try:
        check_nan()
        return True
    except Exception as exc:
        print(f"nan check failed: {exc}")
        return False


def _outputs_have_nan(*tensors):
    return any(torch.isnan(t).any().item() for t in tensors if t is not None)


def _compare_grad(name, fa_grad, golden_grad):
    fa_np = fa_grad.detach().float().cpu().numpy()
    gold_np = golden_grad.detach().float().cpu().numpy()
    print(f"\n{'=' * 72}")
    print(f"[精度] {name}  (FA bwd vs golden autograd)")
    result, pct, max_err = data_compare(fa_np, gold_np)
    print(f"  => {result}, pass_rate={pct:.4f}%, max_rel_err={max_err:.6g}")
    print(f"  |max| FA={np.abs(fa_np).max():.6g} golden={np.abs(gold_np).max():.6g}")
    return result


def _run_flash_attn_backward_bsnd(
    dout, q, k, v, out, softmax_lse, dq, dk, dv,
    scale, causal_switch, window_left, window_right, deterministic,
):
    _flash_attn_backward(
        dout, q, k, v, out, softmax_lse,
        None, None, None, None, None, None,
        dq, dk, dv,
        scale, causal_switch, window_left, window_right,
        0.0, deterministic, 0,
    )
    torch.npu.synchronize()


def _run_flash_attn_backward_tnd(
    dout, q, k, v, out, softmax_lse,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dq, dk, dv, scale, causal_switch, window_left, window_right, deterministic,
):
    _flash_attn_backward(
        dout, q, k, v, out, softmax_lse,
        cu_seqlens_q, cu_seqlens_k,
        None, None, max_seqlen_q, max_seqlen_k,
        dq, dk, dv,
        scale, causal_switch, window_left, window_right,
        0.0, deterministic, 0,
    )
    torch.npu.synchronize()


def _check_deterministic_repeat(run_once_fn, repeat=DETERMINISTIC_REPEAT):
    dq_ref, dk_ref, dv_ref = run_once_fn()
    dq_ref = dq_ref.detach().cpu()
    dk_ref = dk_ref.detach().cpu()
    dv_ref = dv_ref.detach().cpu()

    for step in range(repeat):
        torch.npu.manual_seed(2)
        dq1, dk1, dv1 = run_once_fn()
        if not (
            torch.equal(dq_ref, dq1.cpu())
            and torch.equal(dk_ref, dk1.cpu())
            and torch.equal(dv_ref, dv1.cpu())
        ):
            print(
                f"deterministic check failed at step {step}: "
                f"dq={torch.equal(dq_ref, dq1.cpu())}, "
                f"dk={torch.equal(dk_ref, dk1.cpu())}, "
                f"dv={torch.equal(dv_ref, dv1.cpu())}"
            )
            return False
        print(f"deterministic check step {step} success")
    return True


def _build_test_result(
    *,
    nan_ok,
    torch_bwd_mem,
    tridao_bwd_mem,
    cmp_results,
    deterministic_ok=True,
    exception=None,
    torch_bwd_kernel_ms="",
    tridao_bwd_kernel_ms="",
    torch_prof_path="",
    tridao_prof_path="",
):
    compare_ok = (
        exception is None
        and all(
            result == "success"
            for key, result in cmp_results.items()
            if key != "output_nan"
        )
        and deterministic_ok
    )
    output_nan = cmp_results.get("output_nan", False)
    nan_result_ok = nan_ok and not output_nan
    overall_success = nan_result_ok and compare_ok
    return {
        "overall_success": overall_success,
        "nan_ok": nan_result_ok,
        "torch_bwd_mem": torch_bwd_mem,
        "tridao_bwd_mem": tridao_bwd_mem,
        "compare_ok": compare_ok,
        "cmp_results": cmp_results,
        "deterministic_ok": deterministic_ok,
        "exception": exception,
        "torch_bwd_kernel_ms": torch_bwd_kernel_ms,
        "tridao_bwd_kernel_ms": tridao_bwd_kernel_ms,
        "torch_prof_path": torch_prof_path,
        "tridao_prof_path": tridao_prof_path,
    }


def _run_perf_profiling_bsnd(
    case_label,
    prof_base_path,
    q,
    k,
    v,
    dout,
    nheads,
    scale,
    keep_prob,
    atten_mask_npu,
    sparse_mode,
    causal_switch,
    window_size_left,
    window_size_right,
    is_deterministic,
    out_npu,
    softmax_lse,
):
    safe_name = _sanitize_prof_name(case_label)
    torch_prof_path = os.path.join(prof_base_path, f"{safe_name}_torch_bwd")
    tridao_prof_path = os.path.join(prof_base_path, f"{safe_name}_tridao_bwd")

    def _torch_bwd_step():
        q_g = q.detach().clone().requires_grad_(True)
        k_g = k.detach().clone().requires_grad_(True)
        v_g = v.detach().clone().requires_grad_(True)
        out = torch_npu.npu_fusion_attention(
            q_g, k_g, v_g, nheads,
            pse=None,
            padding_mask=None,
            atten_mask=atten_mask_npu,
            scale=scale,
            keep_prob=float(keep_prob),
            input_layout="BSND",
            pre_tockens=65536,
            next_tockens=0,
            inner_precise=0,
            sparse_mode=sparse_mode,
            prefix=None,
        )[0]
        out.backward(dout)
        torch.npu.synchronize()

    def _tridao_bwd_step():
        dq1 = torch.empty_like(q)
        dk1 = torch.empty_like(k)
        dv1 = torch.empty_like(v)
        _run_flash_attn_backward_bsnd(
            dout, q, k, v, out_npu.detach(), softmax_lse, dq1, dk1, dv1,
            scale, causal_switch, window_size_left, window_size_right,
            bool(is_deterministic),
        )

    torch_bwd_kernel_ms = _profile_with_loop(
        torch_prof_path, _torch_bwd_step, get_kernel_time_backward
    )
    tridao_bwd_kernel_ms = _profile_with_loop(
        tridao_prof_path, _tridao_bwd_step, get_kernel_time_tridao_bwd
    )
    return {
        "torch_bwd_kernel_ms": f"{torch_bwd_kernel_ms:.6f}",
        "tridao_bwd_kernel_ms": f"{tridao_bwd_kernel_ms:.6f}",
        "torch_prof_path": torch_prof_path,
        "tridao_prof_path": tridao_prof_path,
    }


def _run_perf_profiling_tnd(
    case_label,
    prof_base_path,
    q,
    k,
    v,
    dout,
    nheads,
    scale,
    keep_prob,
    atten_mask_npu,
    sparse_mode,
    causal_switch,
    window_size_left,
    window_size_right,
    is_deterministic,
    out_npu,
    softmax_lse,
    cu_seqlens_q_npu,
    cu_seqlens_k_npu,
    max_seqlen_q,
    max_seqlen_k,
    cu_seq_len_list,
    cu_seq_kvlen_list,
):
    safe_name = _sanitize_prof_name(case_label)
    torch_prof_path = os.path.join(prof_base_path, f"{safe_name}_torch_bwd")
    tridao_prof_path = os.path.join(prof_base_path, f"{safe_name}_tridao_bwd")

    def _torch_bwd_step():
        q_g = q.detach().clone().requires_grad_(True)
        k_g = k.detach().clone().requires_grad_(True)
        v_g = v.detach().clone().requires_grad_(True)
        out = torch_npu.npu_fusion_attention(
            q_g, k_g, v_g, nheads,
            pse=None,
            padding_mask=None,
            atten_mask=atten_mask_npu,
            scale=scale,
            keep_prob=float(keep_prob),
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seq_len_list),
            actual_seq_kvlen=tuple(cu_seq_kvlen_list),
            pre_tockens=65536,
            next_tockens=0,
            inner_precise=0,
            sparse_mode=sparse_mode,
            prefix=None,
        )[0]
        out.backward(dout)
        torch.npu.synchronize()

    def _tridao_bwd_step():
        dq1 = torch.empty_like(q)
        dk1 = torch.empty_like(k)
        dv1 = torch.empty_like(v)
        _run_flash_attn_backward_tnd(
            dout, q, k, v, out_npu.detach(), softmax_lse,
            cu_seqlens_q_npu, cu_seqlens_k_npu, max_seqlen_q, max_seqlen_k,
            dq1, dk1, dv1, scale, causal_switch, window_size_left, window_size_right,
            bool(is_deterministic),
        )

    torch_bwd_kernel_ms = _profile_with_loop(
        torch_prof_path, _torch_bwd_step, get_kernel_time_backward
    )
    tridao_bwd_kernel_ms = _profile_with_loop(
        tridao_prof_path, _tridao_bwd_step, get_kernel_time_tridao_bwd
    )
    return {
        "torch_bwd_kernel_ms": f"{torch_bwd_kernel_ms:.6f}",
        "tridao_bwd_kernel_ms": f"{tridao_bwd_kernel_ms:.6f}",
        "torch_prof_path": torch_prof_path,
        "tridao_prof_path": tridao_prof_path,
    }


def test_tnd_bwd_only_npu(
    nheads,
    nheads_k,
    headdim,
    list_seq_q,
    list_seq_kv,
    *,
    dtype="bf16",
    keep_prob=1.0,
    is_causal=False,
    is_deterministic=False,
    window_size_left=-1,
    window_size_right=-1,
    case_name=None,
    seed=3,
    enable_perf=False,
    prof_base_path=None,
):
    case_label = case_name or f"TND(batch={len(list_seq_q)})"
    nan_ok = _run_nan_check()
    scale = 1 / (headdim ** 0.5)
    seqlens_list_q = np.array(list_seq_q, dtype=np.int64)
    seqlens_list_k = np.array(list_seq_kv, dtype=np.int64)

    max_seqlen_q = int(np.max(seqlens_list_q))
    max_seqlen_k = int(np.max(seqlens_list_k))
    cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
    cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
    total_q = int(seqlens_list_q.sum())
    total_k = int(seqlens_list_k.sum())
    cu_seq_len_list = cu_seqlens_q[1:].cpu().numpy().tolist()
    cu_seq_kvlen_list = cu_seqlens_k[1:].cpu().numpy().tolist()
    print(f"[{case_label}] total_q={total_q} total_k={total_k}")
    print(f"[{case_label}] cu_seq_len_list={cu_seq_len_list}")
    print(f"[{case_label}] cu_seq_kvlen_list={cu_seq_kvlen_list}")

    if is_deterministic:
        torch.use_deterministic_algorithms(True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed"):
        torch.npu.manual_seed(seed)

    pttype = _dtype_from_str(dtype)
    limit = 2
    q = (limit * (torch.rand([total_q, nheads, headdim]) - 0.5)).to(pttype)
    k = (limit * (torch.rand([total_k, nheads_k, headdim]) - 0.5)).to(pttype)
    v = (limit * (torch.rand([total_k, nheads_k, headdim]) - 0.5)).to(pttype)
    dout = (limit * (torch.rand([total_q, nheads, headdim]) - 0.5)).to(pttype)
    print(f"[{case_label}] q.shape={q.shape} k.shape={k.shape} v.shape={v.shape} dout.shape={dout.shape}")

    pre_tocken = 65536
    next_tocken = 0
    if is_causal:
        atten_mask_npu = (torch.triu(torch.ones([2048, 2048]), diagonal=1)).to(torch.bool).npu()
        sparse_mode = 2
        causal_switch = True
    else:
        atten_mask_npu = None
        sparse_mode = 0
        causal_switch = False

    q = q.npu()
    k = k.npu()
    v = v.npu()
    dout = dout.npu()

    q_g = q.detach().clone().requires_grad_(True)
    k_g = k.detach().clone().requires_grad_(True)
    v_g = v.detach().clone().requires_grad_(True)
    torch.npu.synchronize()

    start_memery = _reset_mem_stats()
    print(f"[{case_label}] start_memery: {start_memery} B")
    npu_rst = torch_npu.npu_fusion_attention(
        q_g, k_g, v_g, nheads,
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask_npu,
        scale=scale,
        keep_prob=float(keep_prob),
        input_layout="TND",
        actual_seq_qlen=tuple(cu_seq_len_list),
        actual_seq_kvlen=tuple(cu_seq_kvlen_list),
        pre_tockens=pre_tocken,
        next_tockens=next_tocken,
        inner_precise=0,
        sparse_mode=sparse_mode,
        prefix=None,
    )
    out_npu = npu_rst[0]
    torch.npu.synchronize()
    fwd_memery = torch.npu.max_memory_allocated()
    print(f"[{case_label}] fwd_memery: {fwd_memery} B")

    x_max_npu = npu_rst[1]
    x_sum_npu = npu_rst[2]
    softmax_lse = lse_from_golden_max_sum(x_max_npu, x_sum_npu, "tnd")
    out_npu.backward(dout)
    dq_golden_npu = q_g.grad
    dk_golden_npu = k_g.grad
    dv_golden_npu = v_g.grad
    torch.npu.synchronize()

    max_memery = torch.npu.max_memory_allocated()
    torch_bwd_mem = _mb_used(max_memery, fwd_memery)
    print(f"[{case_label}] torch_npu 反向显存消耗: {torch_bwd_mem} MB")

    cu_seqlens_q_npu = torch.tensor(cu_seqlens_q, dtype=torch.int32).npu()
    cu_seqlens_k_npu = torch.tensor(cu_seqlens_k, dtype=torch.int32).npu()

    start_memery = _reset_mem_stats()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _run_flash_attn_backward_tnd(
        dout, q, k, v, out_npu.detach(), softmax_lse,
        cu_seqlens_q_npu, cu_seqlens_k_npu, max_seqlen_q, max_seqlen_k,
        dq, dk, dv, scale, causal_switch, window_size_left, window_size_right,
        bool(is_deterministic),
    )
    tridao_bwd_mem = _mb_used(torch.npu.max_memory_allocated(), start_memery)
    print(f"[{case_label}] tridao 反向显存消耗: {tridao_bwd_mem} MB")

    dq_cmp = _compare_grad("dq", dq, dq_golden_npu)
    dk_cmp = _compare_grad("dk", dk, dk_golden_npu)
    dv_cmp = _compare_grad("dv", dv, dv_golden_npu)
    output_nan = _outputs_have_nan(dq, dk, dv)
    if output_nan:
        print(f"[{case_label}] test fail, output have nan")

    deterministic_ok = True
    if is_deterministic:
        print(f"[{case_label}] start deterministic FA bwd check, repeat={DETERMINISTIC_REPEAT}")

        def _run_once():
            dq1 = torch.empty_like(q)
            dk1 = torch.empty_like(k)
            dv1 = torch.empty_like(v)
            _run_flash_attn_backward_tnd(
                dout, q, k, v, out_npu.detach(), softmax_lse,
                cu_seqlens_q_npu, cu_seqlens_k_npu, max_seqlen_q, max_seqlen_k,
                dq1, dk1, dv1, scale, causal_switch, window_size_left, window_size_right,
                True,
            )
            return dq1, dk1, dv1

        deterministic_ok = _check_deterministic_repeat(_run_once)

    perf_result = {}
    if enable_perf:
        perf_base = prof_base_path or DEFAULT_PROF_BASE
        os.makedirs(perf_base, exist_ok=True)
        print(f"[{case_label}] start performance profiling")
        perf_result = _run_perf_profiling_tnd(
            case_label, perf_base, q, k, v, dout, nheads, scale, keep_prob,
            atten_mask_npu, sparse_mode, causal_switch, window_size_left, window_size_right,
            is_deterministic, out_npu, softmax_lse,
            cu_seqlens_q_npu, cu_seqlens_k_npu, max_seqlen_q, max_seqlen_k,
            cu_seq_len_list, cu_seq_kvlen_list,
        )
        print(
            f"[{case_label}] perf torch_bwd={perf_result['torch_bwd_kernel_ms']} ms, "
            f"tridao_bwd={perf_result['tridao_bwd_kernel_ms']} ms"
        )

    cmp_results = {"dq": dq_cmp, "dk": dk_cmp, "dv": dv_cmp, "output_nan": output_nan}
    result = _build_test_result(
        nan_ok=nan_ok,
        torch_bwd_mem=torch_bwd_mem,
        tridao_bwd_mem=tridao_bwd_mem,
        cmp_results=cmp_results,
        deterministic_ok=deterministic_ok,
        **perf_result,
    )
    print("=" * 72)
    print(
        f"[{case_label}] TND bwd-only overall: {'PASS' if result['overall_success'] else 'FAILED'}  "
        f"nan_ok={result['nan_ok']} compare_ok={result['compare_ok']} "
        f"is_deterministic={is_deterministic} deterministic_ok={deterministic_ok}  {cmp_results}"
    )
    print("=" * 72)
    return result


def test_bsnd_bwd_only_npu(
    nheads,
    nheads_k,
    headdim,
    batch,
    seq_q,
    seq_k,
    *,
    dtype="half",
    keep_prob=1.0,
    is_causal=False,
    is_deterministic=False,
    window_size_left=-1,
    window_size_right=-1,
    case_name=None,
    seed=3,
    enable_perf=False,
    prof_base_path=None,
):
    case_label = case_name or f"BSND(batch={batch},seq_q={seq_q},seq_k={seq_k})"
    nan_ok = _run_nan_check()
    scale = 1 / (headdim ** 0.5)

    if is_deterministic:
        torch.use_deterministic_algorithms(True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed"):
        torch.npu.manual_seed(seed)

    pttype = _dtype_from_str(dtype)
    limit = 2
    q = (limit * (torch.rand([batch, seq_q, nheads, headdim]) - 0.5)).to(pttype)
    k = (limit * (torch.rand([batch, seq_k, nheads_k, headdim]) - 0.5)).to(pttype)
    v = (limit * (torch.rand([batch, seq_k, nheads_k, headdim]) - 0.5)).to(pttype)
    dout = (limit * (torch.rand([batch, seq_q, nheads, headdim]) - 0.5)).to(pttype)
    print(f"[{case_label}] q.shape={q.shape} k.shape={k.shape} v.shape={v.shape} dout.shape={dout.shape}")

    pre_tocken = 65536
    next_tocken = 0
    if is_causal:
        atten_mask_npu = (torch.triu(torch.ones([2048, 2048]), diagonal=1)).to(torch.bool).npu()
        sparse_mode = 2
        causal_switch = True
    else:
        atten_mask_npu = None
        sparse_mode = 0
        causal_switch = False

    q = q.npu()
    k = k.npu()
    v = v.npu()
    dout = dout.npu()

    q_g = q.detach().clone().requires_grad_(True)
    k_g = k.detach().clone().requires_grad_(True)
    v_g = v.detach().clone().requires_grad_(True)
    torch.npu.synchronize()

    start_memery = _reset_mem_stats()
    print(f"[{case_label}] start_memery: {start_memery} B")
    npu_rst = torch_npu.npu_fusion_attention(
        q_g, k_g, v_g, nheads,
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask_npu,
        scale=scale,
        keep_prob=float(keep_prob),
        input_layout="BSND",
        pre_tockens=pre_tocken,
        next_tockens=next_tocken,
        inner_precise=0,
        sparse_mode=sparse_mode,
        prefix=None,
    )
    out_npu = npu_rst[0]
    torch.npu.synchronize()
    fwd_memery = torch.npu.max_memory_allocated()
    print(f"[{case_label}] fwd_memery: {fwd_memery} B")

    x_max_npu = npu_rst[1]
    x_sum_npu = npu_rst[2]
    softmax_lse = lse_from_golden_max_sum(x_max_npu, x_sum_npu, "bsnd")
    out_npu.backward(dout)
    dq_golden_npu = q_g.grad
    dk_golden_npu = k_g.grad
    dv_golden_npu = v_g.grad
    torch.npu.synchronize()

    max_memery = torch.npu.max_memory_allocated()
    torch_bwd_mem = _mb_used(max_memery, fwd_memery)
    print(f"[{case_label}] torch_npu 反向显存消耗: {torch_bwd_mem} MB")

    start_memery = _reset_mem_stats()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _run_flash_attn_backward_bsnd(
        dout, q, k, v, out_npu.detach(), softmax_lse, dq, dk, dv,
        scale, causal_switch, window_size_left, window_size_right, bool(is_deterministic),
    )
    tridao_bwd_mem = _mb_used(torch.npu.max_memory_allocated(), start_memery)
    print(f"[{case_label}] tridao 反向显存消耗: {tridao_bwd_mem} MB")

    dq_cmp = _compare_grad("dq", dq, dq_golden_npu)
    dk_cmp = _compare_grad("dk", dk, dk_golden_npu)
    dv_cmp = _compare_grad("dv", dv, dv_golden_npu)
    output_nan = _outputs_have_nan(dq, dk, dv)
    if output_nan:
        print(f"[{case_label}] test fail, output have nan")

    deterministic_ok = True
    if is_deterministic:
        print(f"[{case_label}] start deterministic FA bwd check, repeat={DETERMINISTIC_REPEAT}")

        def _run_once():
            dq1 = torch.empty_like(q)
            dk1 = torch.empty_like(k)
            dv1 = torch.empty_like(v)
            _run_flash_attn_backward_bsnd(
                dout, q, k, v, out_npu.detach(), softmax_lse, dq1, dk1, dv1,
                scale, causal_switch, window_size_left, window_size_right, True,
            )
            return dq1, dk1, dv1

        deterministic_ok = _check_deterministic_repeat(_run_once)

    perf_result = {}
    if enable_perf:
        perf_base = prof_base_path or DEFAULT_PROF_BASE
        os.makedirs(perf_base, exist_ok=True)
        print(f"[{case_label}] start performance profiling")
        perf_result = _run_perf_profiling_bsnd(
            case_label, perf_base, q, k, v, dout, nheads, scale, keep_prob,
            atten_mask_npu, sparse_mode, causal_switch, window_size_left, window_size_right,
            is_deterministic, out_npu, softmax_lse,
        )
        print(
            f"[{case_label}] perf torch_bwd={perf_result['torch_bwd_kernel_ms']} ms, "
            f"tridao_bwd={perf_result['tridao_bwd_kernel_ms']} ms"
        )

    cmp_results = {"dq": dq_cmp, "dk": dk_cmp, "dv": dv_cmp, "output_nan": output_nan}
    result = _build_test_result(
        nan_ok=nan_ok,
        torch_bwd_mem=torch_bwd_mem,
        tridao_bwd_mem=tridao_bwd_mem,
        cmp_results=cmp_results,
        deterministic_ok=deterministic_ok,
        **perf_result,
    )
    print("=" * 72)
    print(
        f"[{case_label}] BSND bwd-only overall: {'PASS' if result['overall_success'] else 'FAILED'}  "
        f"nan_ok={result['nan_ok']} compare_ok={result['compare_ok']} "
        f"is_deterministic={is_deterministic} deterministic_ok={deterministic_ok}  {cmp_results}"
    )
    print("=" * 72)
    return result


def copy_case_file_with_timestamp(src_path):
    base, ext = os.path.splitext(src_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_path = f"{base}_{timestamp}{ext}"
    shutil.copy2(src_path, dst_path)
    print(f"Copied case file to {dst_path}")
    return dst_path


def _row_case_name(row, case_idx):
    if "case_name" in row and pd.notna(row["case_name"]):
        return str(row["case_name"])
    return f"case_{case_idx}"


def _common_row_params(row, case_idx):
    return {
        "nheads": int(row["nheads"]),
        "nheads_k": int(row["nheads_k"]),
        "headdim": int(row["headdim"]),
        "dtype": row["dtype"],
        "keep_prob": float(row.get("keep_prob", 1.0)),
        "is_causal": _parse_flag(row.get("is_causal", 0)),
        "is_deterministic": _parse_flag(row.get("is_deterministic", 0)),
        "window_size_left": int(row.get("window_size_left", -1)),
        "window_size_right": int(row.get("window_size_right", -1)),
        "case_name": _row_case_name(row, case_idx),
        "seed": 3 + case_idx,
    }


def _write_row_result(df, index, run_result, result_path):
    df.at[index, "nan_result"] = "success" if run_result.get("nan_ok") else "fail"
    df.at[index, "torch_bwd_mem"] = run_result.get("torch_bwd_mem", "")
    df.at[index, "tridao_bwd_mem"] = run_result.get("tridao_bwd_mem", "")
    df.at[index, "compare_result"] = "success" if run_result.get("compare_ok") else "fail"
    df.at[index, "torch_bwd_kernel_ms"] = run_result.get("torch_bwd_kernel_ms", "")
    df.at[index, "tridao_bwd_kernel_ms"] = run_result.get("tridao_bwd_kernel_ms", "")
    df.at[index, "torch_prof_path"] = run_result.get("torch_prof_path", "")
    df.at[index, "tridao_prof_path"] = run_result.get("tridao_prof_path", "")
    df.to_excel(result_path, index=False)


def _failed_result(exception):
    return _build_test_result(
        nan_ok=False,
        torch_bwd_mem="",
        tridao_bwd_mem="",
        cmp_results={},
        deterministic_ok=False,
        exception=str(exception),
        torch_bwd_kernel_ms="",
        tridao_bwd_kernel_ms="",
        torch_prof_path="",
        tridao_prof_path="",
    )


def run_bsnd_cases_from_xlsx(xlsx_path, enable_perf=False, prof_base_path=None):
    result_path = copy_case_file_with_timestamp(xlsx_path)
    df = pd.read_excel(result_path)
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    if enable_perf:
        run_prof_base = prof_base_path or DEFAULT_PROF_BASE
        run_prof_base = os.path.join(
            run_prof_base,
            f"BSND_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(run_prof_base, exist_ok=True)
        print(f"Performance profiling enabled, output base: {run_prof_base}")
    else:
        run_prof_base = None

    success_count = 0
    failed_cases = []
    print(f"Loaded {len(df)} BSND cases from {result_path}")

    for index, row in df.iterrows():
        case_idx = index + 1
        run_result = None
        params = _common_row_params(row, case_idx)
        params.update({
            "batch": int(row["batch"]),
            "seq_q": int(row["seq_q"]),
            "seq_k": int(row["seq_k"]),
            "enable_perf": enable_perf,
            "prof_base_path": run_prof_base,
        })
        try:
            print(f"\n[case {case_idx}/{len(df)}] params: {params}")
            run_result = test_bsnd_bwd_only_npu(**params)
            if run_result["overall_success"]:
                success_count += 1
                print(f"[case {case_idx}] success")
            else:
                failed_cases.append((case_idx, params, run_result))
                print(f"[case {case_idx}] failed: {run_result}")
        except Exception as exc:
            run_result = _failed_result(exc)
            failed_cases.append((case_idx, params, run_result))
            print(f"[case {case_idx}] failed with exception: {exc}")
        finally:
            if run_result is not None:
                _write_row_result(df, index, run_result, result_path)
            npu_between_runs_cleanup()

    return _summarize_run(len(df), success_count, failed_cases, result_path)


def run_tnd_cases_from_xlsx(xlsx_path, enable_perf=False, prof_base_path=None):
    result_path = copy_case_file_with_timestamp(xlsx_path)
    df = pd.read_excel(result_path)
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    if enable_perf:
        run_prof_base = prof_base_path or DEFAULT_PROF_BASE
        run_prof_base = os.path.join(
            run_prof_base,
            f"TND_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(run_prof_base, exist_ok=True)
        print(f"Performance profiling enabled, output base: {run_prof_base}")
    else:
        run_prof_base = None

    success_count = 0
    failed_cases = []
    print(f"Loaded {len(df)} TND cases from {result_path}")

    for index, row in df.iterrows():
        case_idx = index + 1
        run_result = None
        params = _common_row_params(row, case_idx)
        params.update({
            "list_seq_q": _parse_list(row["list_seq_q"]),
            "list_seq_kv": _parse_list(row["list_seq_kv"]),
            "enable_perf": enable_perf,
            "prof_base_path": run_prof_base,
        })
        try:
            print(f"\n[case {case_idx}/{len(df)}] params: {params}")
            run_result = test_tnd_bwd_only_npu(**params)
            if run_result["overall_success"]:
                success_count += 1
                print(f"[case {case_idx}] success")
            else:
                failed_cases.append((case_idx, params, run_result))
                print(f"[case {case_idx}] failed: {run_result}")
        except Exception as exc:
            run_result = _failed_result(exc)
            failed_cases.append((case_idx, params, run_result))
            print(f"[case {case_idx}] failed with exception: {exc}")
        finally:
            if run_result is not None:
                _write_row_result(df, index, run_result, result_path)
            npu_between_runs_cleanup()

    return _summarize_run(len(df), success_count, failed_cases, result_path)


def _summarize_run(total, success_count, failed_cases, result_path):
    failed_count = total - success_count
    print("=" * 80)
    print(f"XLSX summary: total={total}, success={success_count}, failed={failed_count}")
    print(f"Result file: {result_path}")
    if failed_cases:
        print("Failed cases detail:")
        for case_idx, params, detail in failed_cases:
            print(f"  - case {case_idx}, params={params}, detail={detail}")
    print("=" * 80)
    return {
        "total": total,
        "success": success_count,
        "failed": failed_count,
        "failed_cases": failed_cases,
        "result_path": result_path,
    }


def run_cases_from_xlsx(xlsx_path, test_layout, enable_perf=False, prof_base_path=None):
    layout = test_layout.upper()
    if layout == "BSND":
        return run_bsnd_cases_from_xlsx(xlsx_path, enable_perf, prof_base_path)
    if layout == "TND":
        return run_tnd_cases_from_xlsx(xlsx_path, enable_perf, prof_base_path)
    raise ValueError(f"unsupported test_layout: {test_layout}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run FA bwd-only tests from xlsx cases")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NPU device id (default: 0)",
    )
    parser.add_argument(
        "--test_layout",
        type=str,
        choices=["BSND", "TND"],
        default="BSND",
        help="test layout branch (default: BSND)",
    )
    parser.add_argument(
        "--case_file",
        type=str,
        default=None,
        help="xlsx case file path (default: fag_cases_<layout>_drop.xlsx)",
    )
    parser.add_argument(
        "--enable_perf",
        action="store_true",
        help="enable torch_npu profiler for npu_fusion_attention bwd and _flash_attn_backward",
    )
    parser.add_argument(
        "--prof_path",
        type=str,
        default=DEFAULT_PROF_BASE,
        help=f"profiler output base path (default: {DEFAULT_PROF_BASE})",
    )
    return parser.parse_args()


def resolve_case_file(case_file, test_layout):
    if case_file is None:
        case_file = DEFAULT_BSND_XLSX if test_layout == "BSND" else DEFAULT_TND_XLSX
    if os.path.isabs(case_file):
        return case_file
    if os.path.exists(case_file):
        return case_file
    candidate = os.path.join(TESTS_DIR, case_file)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"case file not found: {case_file}")


if __name__ == "__main__":
    args = parse_args()
    setup_npu_device(args.device)
    case_file = resolve_case_file(args.case_file, args.test_layout)
    prof_base_path = args.prof_path
    if not os.path.isabs(prof_base_path):
        prof_base_path = os.path.join(TESTS_DIR, prof_base_path)
    run_cases_from_xlsx(
        case_file,
        args.test_layout,
        enable_perf=args.enable_perf,
        prof_base_path=prof_base_path,
    )
