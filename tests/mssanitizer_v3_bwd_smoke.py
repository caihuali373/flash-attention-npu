"""
msSanitizer 专用 v3 反向 smoke / xlsx 批量测试。

仅调用 flash_attn_npu_v3._flash_attn_backward（FAGGeneral kernel），
不做 golden 前向/精度比对，用于内存/竞争/同步类检测。

用法:
  # 内置 smoke 用例（快速）
  python tests/mssanitizer_v3_bwd_smoke.py --device 0 --smoke

  # 从默认 xlsx 读取用例
  python tests/mssanitizer_v3_bwd_smoke.py --device 1 --test_layout BSND --use_xlsx
  python tests/mssanitizer_v3_bwd_smoke.py --device 1 --test_layout TND --use_xlsx
  python tests/mssanitizer_v3_bwd_smoke.py --device 0 --test_layout both --use_xlsx --max_cases 5

  # 指定 xlsx 文件
  python tests/mssanitizer_v3_bwd_smoke.py --device 1 --test_layout TND --case_file fag_cases_TND_drop.xlsx
"""

from __future__ import annotations

import argparse
import ast
import gc
import glob
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
DEFAULT_BSND_XLSX = os.path.join(TESTS_DIR, "fag_cases_BSND_drop.xlsx")
DEFAULT_TND_XLSX = os.path.join(TESTS_DIR, "fag_cases_TND_drop.xlsx")


def _prepend_local_build_paths() -> None:
    """Prefer this repo's freshly built flash_attn_npu_3 extension."""
    candidates = [PROJECT_DIR]
    candidates.extend(sorted(glob.glob(os.path.join(PROJECT_DIR, "build", "lib.*")), reverse=True))
    for path in candidates:
        if path not in sys.path:
            sys.path.insert(0, path)


def _ensure_v3_bwd_module():
    import flash_attn_npu_3 as ext

    if hasattr(ext, "bwd"):
        return ext

    attrs = [name for name in dir(ext) if not name.startswith("_")]
    module_path = getattr(ext, "__file__", "<unknown>")
    raise RuntimeError(
        "flash_attn_npu_3.bwd is missing. Loaded extension is stale or not built for v3 backward.\n"
        f"  module: {module_path}\n"
        f"  attrs: {attrs}\n"
        "Fix:\n"
        f"  cd {PROJECT_DIR}\n"
        "  FLASH_ATTN_BUILD_VERSION=v3 python setup.py install\n"
        "Or:\n"
        "  ./scripts/run_mssanitizer_v3_bwd.sh --build ..."
    )


_prepend_local_build_paths()

import pandas as pd
import torch
import torch_npu

_ensure_v3_bwd_module()

from flash_attn_npu_v3.flash_attn_interface import _flash_attn_backward


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="msSanitizer smoke/xlsx test for v3 backward")
    parser.add_argument(
        "--device",
        "--device_id",
        dest="device",
        type=int,
        default=int(os.environ.get("NPU_DEVICE", "0")),
        help="NPU device id (default: NPU_DEVICE env or 0)",
    )
    parser.add_argument(
        "--test_layout",
        "--layout",
        dest="test_layout",
        choices=("BSND", "TND", "both", "bsnd", "tnd"),
        default=os.environ.get("FA_BWD_LAYOUT", "both").upper(),
        help="BSND, TND, or both (default: both)",
    )
    parser.add_argument(
        "--case_file",
        type=str,
        default=None,
        help="xlsx case file path (implies --use_xlsx)",
    )
    parser.add_argument(
        "--use_xlsx",
        action="store_true",
        help="Load cases from default xlsx for --test_layout (or --case_file)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run built-in minimal smoke cases (default when neither --smoke nor xlsx mode)",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "half"),
        default=os.environ.get("FA_BWD_DTYPE", None),
        help="dtype for built-in smoke only (xlsx uses per-row dtype)",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override deterministic flag for built-in smoke only",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable causal mask for built-in smoke only",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Max number of xlsx cases to run (default: all)",
    )
    parser.add_argument(
        "--start_case",
        type=int,
        default=1,
        help="1-based start index for xlsx cases (default: 1)",
    )
    return parser.parse_args()


def _normalize_layout(layout: str) -> str:
    layout = layout.upper()
    if layout not in ("BSND", "TND", "BOTH"):
        raise ValueError(f"unsupported layout: {layout}")
    return layout


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    name = str(dtype_str).lower()
    if name in ("half", "fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_str}")


def _dtype_from_name(name: str | None) -> torch.dtype:
    if name is None:
        return torch.bfloat16
    return _dtype_from_str(name)


def _parse_flag(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes")
    return bool(int(val))


def _parse_list(val) -> list[int]:
    if isinstance(val, str):
        return ast.literal_eval(val)
    return list(val)


def _row_case_name(row, case_idx: int) -> str:
    if "case_name" in row and pd.notna(row["case_name"]):
        return str(row["case_name"])
    return f"case_{case_idx}"


def _npu_cleanup() -> None:
    torch.npu.synchronize()
    gc.collect()
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()


def _cu_seqlens(seqlens_list: list[int], device: torch.device) -> torch.Tensor:
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu.to(dtype=torch.int32, device=device)


def _run_bsnd_case(
    device: torch.device,
    *,
    batch: int,
    seq_q: int,
    seq_k: int,
    nheads: int,
    nheads_k: int,
    headdim: int,
    pttype: torch.dtype,
    causal: bool,
    deterministic: bool,
    window_left: int = -1,
    window_right: int = -1,
    case_name: str = "BSND",
) -> None:
    scale = headdim ** -0.5
    q = torch.randn(batch, seq_q, nheads, headdim, dtype=pttype, device=device)
    k = torch.randn(batch, seq_k, nheads_k, headdim, dtype=pttype, device=device)
    v = torch.randn(batch, seq_k, nheads_k, headdim, dtype=pttype, device=device)
    dout = torch.randn(batch, seq_q, nheads, headdim, dtype=pttype, device=device)
    out = torch.randn(batch, seq_q, nheads, headdim, dtype=pttype, device=device)
    softmax_lse = torch.randn(batch, seq_q, nheads, dtype=torch.float32, device=device)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    print(
        f"[{case_name}] BSND batch={batch} seq_q={seq_q} seq_k={seq_k} "
        f"nheads={nheads} nheads_k={nheads_k} headdim={headdim} "
        f"dtype={pttype} causal={causal} deterministic={deterministic}"
    )
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        None,
        None,
        None,
        None,
        seq_q,
        seq_k,
        dq,
        dk,
        dv,
        scale,
        causal,
        window_left,
        window_right,
        0.0,
        deterministic,
        0,
    )
    torch.npu.synchronize()
    print(f"[{case_name}] BSND v3 backward done")


def _run_tnd_case(
    device: torch.device,
    *,
    list_seq_q: list[int],
    list_seq_kv: list[int],
    nheads: int,
    nheads_k: int,
    headdim: int,
    pttype: torch.dtype,
    causal: bool,
    deterministic: bool,
    window_left: int = -1,
    window_right: int = -1,
    case_name: str = "TND",
) -> None:
    scale = headdim ** -0.5
    total_q = sum(list_seq_q)
    total_k = sum(list_seq_kv)
    max_seqlen_q = max(list_seq_q)
    max_seqlen_k = max(list_seq_kv)

    q = torch.randn(total_q, nheads, headdim, dtype=pttype, device=device)
    k = torch.randn(total_k, nheads_k, headdim, dtype=pttype, device=device)
    v = torch.randn(total_k, nheads_k, headdim, dtype=pttype, device=device)
    dout = torch.randn(total_q, nheads, headdim, dtype=pttype, device=device)
    out = torch.randn(total_q, nheads, headdim, dtype=pttype, device=device)
    softmax_lse = torch.randn(total_q, nheads, dtype=torch.float32, device=device)
    cu_seqlens_q = _cu_seqlens(list_seq_q, device)
    cu_seqlens_k = _cu_seqlens(list_seq_kv, device)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    print(
        f"[{case_name}] TND total_q={total_q} total_k={total_k} "
        f"max_seqlen_q={max_seqlen_q} max_seqlen_k={max_seqlen_k} "
        f"nheads={nheads} nheads_k={nheads_k} headdim={headdim} "
        f"dtype={pttype} causal={causal} deterministic={deterministic}"
    )
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        max_seqlen_q,
        max_seqlen_k,
        dq,
        dk,
        dv,
        scale,
        causal,
        window_left,
        window_right,
        0.0,
        deterministic,
        0,
    )
    torch.npu.synchronize()
    print(f"[{case_name}] TND v3 backward done")


def _common_row_params(row, case_idx: int) -> dict:
    return {
        "nheads": int(row["nheads"]),
        "nheads_k": int(row["nheads_k"]),
        "headdim": int(row["headdim"]),
        "pttype": _dtype_from_str(row["dtype"]),
        "causal": _parse_flag(row.get("is_causal", 0)),
        "deterministic": _parse_flag(row.get("is_deterministic", 0)),
        "window_left": int(row.get("window_size_left", -1)),
        "window_right": int(row.get("window_size_right", -1)),
        "case_name": _row_case_name(row, case_idx),
    }


def resolve_case_file(case_file: str | None, layout: str) -> str:
    if case_file is None:
        return DEFAULT_BSND_XLSX if layout == "BSND" else DEFAULT_TND_XLSX
    if os.path.isabs(case_file) and os.path.exists(case_file):
        return case_file
    if os.path.exists(case_file):
        return case_file
    candidate = os.path.join(TESTS_DIR, case_file)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"case file not found: {case_file}")


def run_bsnd_cases_from_xlsx(
    xlsx_path: str,
    device: torch.device,
    *,
    start_case: int = 1,
    max_cases: int | None = None,
) -> dict:
    df = pd.read_excel(xlsx_path)
    total = len(df)
    print(f"Loaded {total} BSND cases from {xlsx_path}")

    success_count = 0
    failed_cases = []
    run_count = 0

    for index, row in df.iterrows():
        case_idx = index + 1
        if case_idx < start_case:
            continue
        if max_cases is not None and run_count >= max_cases:
            break

        params = _common_row_params(row, case_idx)
        params.update({
            "device": device,
            "batch": int(row["batch"]),
            "seq_q": int(row["seq_q"]),
            "seq_k": int(row["seq_k"]),
        })
        try:
            print(f"\n[BSND case {case_idx}/{total}]")
            _run_bsnd_case(**params)
            success_count += 1
        except Exception as exc:
            failed_cases.append((case_idx, params, str(exc)))
            print(f"[BSND case {case_idx}] failed: {exc}")
        finally:
            run_count += 1
            _npu_cleanup()

    return _summarize_xlsx("BSND", run_count, success_count, failed_cases, xlsx_path)


def run_tnd_cases_from_xlsx(
    xlsx_path: str,
    device: torch.device,
    *,
    start_case: int = 1,
    max_cases: int | None = None,
) -> dict:
    df = pd.read_excel(xlsx_path)
    total = len(df)
    print(f"Loaded {total} TND cases from {xlsx_path}")

    success_count = 0
    failed_cases = []
    run_count = 0

    for index, row in df.iterrows():
        case_idx = index + 1
        if case_idx < start_case:
            continue
        if max_cases is not None and run_count >= max_cases:
            break

        params = _common_row_params(row, case_idx)
        params.update({
            "device": device,
            "list_seq_q": _parse_list(row["list_seq_q"]),
            "list_seq_kv": _parse_list(row["list_seq_kv"]),
        })
        try:
            print(f"\n[TND case {case_idx}/{total}]")
            _run_tnd_case(**params)
            success_count += 1
        except Exception as exc:
            failed_cases.append((case_idx, params, str(exc)))
            print(f"[TND case {case_idx}] failed: {exc}")
        finally:
            run_count += 1
            _npu_cleanup()

    return _summarize_xlsx("TND", run_count, success_count, failed_cases, xlsx_path)


def _summarize_xlsx(layout: str, total: int, success_count: int, failed_cases: list, xlsx_path: str) -> dict:
    failed_count = total - success_count
    print("=" * 80)
    print(f"{layout} xlsx summary: total={total}, success={success_count}, failed={failed_count}")
    print(f"case file: {xlsx_path}")
    if failed_cases:
        print("Failed cases:")
        for case_idx, params, detail in failed_cases:
            print(f"  - case {case_idx}, name={params.get('case_name')}, error={detail}")
    print("=" * 80)
    return {
        "layout": layout,
        "total": total,
        "success": success_count,
        "failed": failed_count,
        "failed_cases": failed_cases,
        "case_file": xlsx_path,
    }


def run_builtin_smoke(args: argparse.Namespace, device: torch.device) -> int:
    layout = _normalize_layout(args.test_layout)
    pttype = _dtype_from_name(args.dtype)
    deterministic = True if args.deterministic is None else args.deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True)

    print(
        f"msSanitizer v3 bwd smoke: device={args.device} layout={layout} "
        f"dtype={pttype} deterministic={deterministic} causal={args.causal}"
    )

    if layout in ("BSND", "BOTH"):
        _run_bsnd_case(
            device,
            batch=2,
            seq_q=128,
            seq_k=128,
            nheads=8,
            nheads_k=8,
            headdim=128,
            pttype=pttype,
            causal=args.causal,
            deterministic=deterministic,
            case_name="smoke_BSND",
        )
    if layout in ("TND", "BOTH"):
        _run_tnd_case(
            device,
            list_seq_q=[512, 33, 1111],
            list_seq_kv=[512, 33, 1111],
            nheads=3,
            nheads_k=3,
            headdim=128,
            pttype=pttype,
            causal=args.causal,
            deterministic=deterministic,
            case_name="smoke_TND",
        )
    print("All smoke cases finished.")
    return 0


def run_xlsx_cases(args: argparse.Namespace, device: torch.device) -> int:
    layout = _normalize_layout(args.test_layout)
    summaries = []

    if args.case_file is not None:
        if layout == "BOTH":
            raise ValueError("--case_file with --test_layout both is ambiguous; specify BSND or TND")
        xlsx_path = resolve_case_file(args.case_file, layout)
        if layout == "BSND":
            summaries.append(run_bsnd_cases_from_xlsx(
                xlsx_path, device, start_case=args.start_case, max_cases=args.max_cases,
            ))
        else:
            summaries.append(run_tnd_cases_from_xlsx(
                xlsx_path, device, start_case=args.start_case, max_cases=args.max_cases,
            ))
    else:
        if layout in ("BSND", "BOTH"):
            summaries.append(run_bsnd_cases_from_xlsx(
                DEFAULT_BSND_XLSX, device, start_case=args.start_case, max_cases=args.max_cases,
            ))
        if layout in ("TND", "BOTH"):
            summaries.append(run_tnd_cases_from_xlsx(
                DEFAULT_TND_XLSX, device, start_case=args.start_case, max_cases=args.max_cases,
            ))

    failed = sum(s["failed"] for s in summaries)
    return 1 if failed else 0


def main() -> int:
    args = _parse_args()
    if not torch.npu.is_available():
        print("ERROR: torch.npu is not available", file=sys.stderr)
        return 1

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    print(f"Using NPU device: {args.device}")

    use_xlsx = args.use_xlsx or args.case_file is not None
    if use_xlsx:
        return run_xlsx_cases(args, device)
    return run_builtin_smoke(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
