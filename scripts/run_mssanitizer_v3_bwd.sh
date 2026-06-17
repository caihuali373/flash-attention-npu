#!/usr/bin/env bash
# One-click msSanitizer workflow for flash_attn_npu v3 backward (FAGGeneral kernel).
#
# Examples:
#   ./scripts/run_mssanitizer_v3_bwd.sh
#   ./scripts/run_mssanitizer_v3_bwd.sh --build
#   ./scripts/run_mssanitizer_v3_bwd.sh --full-build --quick
#   ./scripts/run_mssanitizer_v3_bwd.sh --use-xlsx --test_layout BSND --device 1 --max_cases 5
#   ./scripts/run_mssanitizer_v3_bwd.sh --case_file tests/fag_cases_TND_drop.xlsx --test_layout TND
#   ./scripts/run_mssanitizer_v3_bwd.sh --cann /usr/local/Ascend/cann-9.1.0-beta.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_SCRIPT="${PROJECT_DIR}/tests/mssanitizer_v3_bwd_smoke.py"
LOG_DIR="${PROJECT_DIR}/mssanitizer_logs"
KERNEL_NAME="FAGGeneral"

DO_BUILD=0
AUTO_BUILD=1
FULL_BUILD=0
QUICK=0
LEAK_CHECK=0
USE_XLSX=0
USE_SMOKE=0
CANN_ROOT=""
CASE_FILE=""
MAX_CASES=""
NPU_DEVICE="${NPU_DEVICE:-0}"
TEST_LAYOUT="${FA_BWD_LAYOUT:-both}"
DTYPE="${FA_BWD_DTYPE:-}"
PYTHON="${PYTHON:-python}"

usage() {
    cat <<'EOF'
Usage: run_mssanitizer_v3_bwd.sh [options]

Options:
  --build           Build v3 extension (FLASH_ATTN_BUILD_VERSION=v3)
  --no-auto-build   Do not auto-build when flash_attn_npu_3.bwd is missing
  --full-build      msSanitizer bwd-only build (-g --cce-enable-sanitizer, skips FAInfer fwd)
  --quick           Run memcheck only (default: memcheck + racecheck + initcheck + synccheck)
  --leak-check      Also run device/cann heap leak checks (separate from kernel memcheck)
  --cann PATH       CANN root path (default: ASCEND_HOME_PATH or auto-detect)
  --device ID       NPU device id (alias: --device_id)
  --test_layout MODE
                    BSND | TND | both (default: both)
  --layout MODE     Alias of --test_layout (bsnd/tnd/both also accepted)
  --use-xlsx        Load cases from default xlsx for --test_layout
  --case_file PATH  Load cases from specified xlsx (implies --use-xlsx)
  --max_cases N     Limit number of xlsx cases per layout
  --smoke           Run built-in minimal smoke cases (default without --use-xlsx)
  --dtype TYPE      fp16 | bf16 for built-in smoke only
  --python CMD      Python interpreter (default: python)
  -h, --help        Show this help

Environment:
  NPU_DEVICE, FA_BWD_LAYOUT, FA_BWD_DTYPE, ASCEND_HOME_PATH
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) DO_BUILD=1; AUTO_BUILD=0; shift ;;
        --no-auto-build) AUTO_BUILD=0; shift ;;
        --full-build) DO_BUILD=1; FULL_BUILD=1; AUTO_BUILD=0; shift ;;
        --quick) QUICK=1; shift ;;
        --leak-check) LEAK_CHECK=1; shift ;;
        --cann) CANN_ROOT="$2"; shift 2 ;;
        --device|--device_id) NPU_DEVICE="$2"; shift 2 ;;
        --test_layout|--layout)
            case "$(echo "$2" | tr '[:lower:]' '[:upper:]')" in
                BSND|TND|BOTH) TEST_LAYOUT="$(echo "$2" | tr '[:lower:]' '[:upper:]')" ;;
                *) echo "Invalid layout: $2" >&2; exit 1 ;;
            esac
            shift 2
            ;;
        --use-xlsx) USE_XLSX=1; shift ;;
        --case_file) CASE_FILE="$2"; USE_XLSX=1; shift 2 ;;
        --max_cases) MAX_CASES="$2"; shift 2 ;;
        --smoke) USE_SMOKE=1; shift ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

detect_cann_root() {
    if [[ -n "${CANN_ROOT}" ]]; then
        echo "${CANN_ROOT}"
        return
    fi
    if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
        echo "${ASCEND_HOME_PATH}"
        return
    fi
    local candidates=(
        /usr/local/Ascend/cann-9.1.0-beta.1
        /usr/local/Ascend/cann-8.5.1
        /usr/local/Ascend/ascend-toolkit/latest
    )
    for c in "${candidates[@]}"; do
        if [[ -f "${c}/set_env.sh" && -x "${c}/tools/mssanitizer/bin/mssanitizer" ]]; then
            echo "${c}"
            return
        fi
    done
    echo ""
}

CANN_ROOT="$(detect_cann_root)"
if [[ -z "${CANN_ROOT}" ]]; then
    echo "ERROR: Cannot find CANN root. Set ASCEND_HOME_PATH or pass --cann PATH." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${CANN_ROOT}/set_env.sh"
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export PATH="${CANN_ROOT}/tools/mssanitizer/bin:${PATH}"
export LD_LIBRARY_PATH="${CANN_ROOT}/lib64:${CANN_ROOT}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"

MSSAN="${CANN_ROOT}/tools/mssanitizer/bin/mssanitizer"
if [[ ! -x "${MSSAN}" ]]; then
    echo "ERROR: mssanitizer not found at ${MSSAN}" >&2
    exit 1
fi

prepend_pythonpath() {
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
    local build_lib
    build_lib="$(find "${PROJECT_DIR}/build" -maxdepth 1 -type d -name 'lib.*' 2>/dev/null | head -1 || true)"
    if [[ -n "${build_lib}" ]]; then
        export PYTHONPATH="${build_lib}:${PYTHONPATH}"
    fi
}

build_v3_extension() {
    cd "${PROJECT_DIR}"
    export FLASH_ATTN_BUILD_VERSION=v3
    if [[ "${FULL_BUILD}" -eq 1 ]]; then
        export FLASH_ATTN_ENABLE_MSSANITIZER=TRUE
        echo "[build] FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_ENABLE_MSSANITIZER=TRUE ${PYTHON} setup.py install"
    else
        unset FLASH_ATTN_ENABLE_MSSANITIZER || true
        echo "[build] FLASH_ATTN_BUILD_VERSION=v3 ${PYTHON} setup.py install"
    fi
    "${PYTHON}" setup.py install
    prepend_pythonpath
}

check_v3_bwd() {
    prepend_pythonpath
    "${PYTHON}" - <<'PY'
import sys
try:
    import flash_attn_npu_3 as m
except Exception as exc:
    print(f"IMPORT_FAIL: {exc}", file=sys.stderr)
    sys.exit(2)
if not hasattr(m, "bwd"):
    attrs = [name for name in dir(m) if not name.startswith("_")]
    print(
        "MISSING_BWD: module={} attrs={}".format(getattr(m, "__file__", m), attrs),
        file=sys.stderr,
    )
    sys.exit(1)
print("OK:", getattr(m, "__file__", m))
PY
}

if [[ "${DO_BUILD}" -eq 1 ]]; then
    build_v3_extension
fi

if ! check_v3_bwd; then
    rc=$?
    if [[ "${rc}" -ne 0 && "${AUTO_BUILD}" -eq 1 && "${DO_BUILD}" -eq 0 ]]; then
        echo "[preflight] flash_attn_npu_3.bwd missing, auto-building v3 extension..." >&2
        build_v3_extension
    else
        echo "ERROR: flash_attn_npu_3.bwd is not available in ${PYTHON}." >&2
        echo "Run: ./scripts/run_mssanitizer_v3_bwd.sh --build --device ${NPU_DEVICE} ..." >&2
        exit 1
    fi
    if ! check_v3_bwd; then
        echo "ERROR: flash_attn_npu_3.bwd still missing after build." >&2
        exit 1
    fi
fi

mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_DIR}/mssanitizer_v3_bwd_summary_${TS}.md"

TEST_ARGS=(
    "${TEST_SCRIPT}"
    --device "${NPU_DEVICE}"
    --test_layout "${TEST_LAYOUT}"
)

if [[ "${USE_SMOKE}" -eq 1 ]]; then
    TEST_ARGS+=(--smoke)
elif [[ "${USE_XLSX}" -eq 1 ]]; then
    TEST_ARGS+=(--use_xlsx)
    if [[ -n "${CASE_FILE}" ]]; then
        TEST_ARGS+=(--case_file "${CASE_FILE}")
    fi
    if [[ -n "${MAX_CASES}" ]]; then
        TEST_ARGS+=(--max_cases "${MAX_CASES}")
    fi
else
    TEST_ARGS+=(--smoke)
fi

if [[ -n "${DTYPE}" ]]; then
    TEST_ARGS+=(--dtype "${DTYPE}")
fi

run_check() {
    local name="$1"
    shift
    local log_file="${LOG_DIR}/${name}_${TS}.log"
    echo ""
    echo "========== ${name} =========="
    set +e
    prepend_pythonpath
    "${MSSAN}" "$@" --log-file="${log_file}" -- env PYTHONPATH="${PYTHONPATH:-}" "${PYTHON}" "${TEST_ARGS[@]}"
    local rc=$?
    set -e
    local size
    size="$(wc -c < "${log_file}" | tr -d ' ')"
    if [[ "${rc}" -eq 0 && "${size}" -eq 0 ]]; then
        echo "[${name}] PASS (exit=0, log empty)"
        echo "- **${name}**: PASS (log empty)" >> "${SUMMARY}"
    elif [[ "${rc}" -eq 0 ]]; then
        echo "[${name}] DONE (exit=0, see ${log_file})"
        echo "- **${name}**: DONE — see \`${log_file}\`" >> "${SUMMARY}"
    else
        echo "[${name}] FAIL (exit=${rc}, see ${log_file})" >&2
        echo "- **${name}**: FAIL (exit=${rc}) — see \`${log_file}\`" >> "${SUMMARY}"
        return "${rc}"
    fi
    return 0
}

{
    echo "# msSanitizer v3 backward summary"
    echo ""
    echo "- time: ${TS}"
    echo "- cann: ${CANN_ROOT}"
    echo "- device: ${NPU_DEVICE}"
    echo "- test_layout: ${TEST_LAYOUT}"
    if [[ -n "${DTYPE}" ]]; then echo "- dtype: ${DTYPE}"; fi
    if [[ "${USE_XLSX}" -eq 1 ]]; then
        echo "- mode: xlsx"
        if [[ -n "${CASE_FILE}" ]]; then echo "- case_file: ${CASE_FILE}"; fi
        if [[ -n "${MAX_CASES}" ]]; then echo "- max_cases: ${MAX_CASES}"; fi
    else
        echo "- mode: smoke"
    fi
    echo "- kernel: ${KERNEL_NAME}"
    echo ""
    echo "## Results"
} > "${SUMMARY}"

COMMON_ARGS=(-t memcheck --kernel-name="${KERNEL_NAME}")
FAIL=0

if ! run_check "memcheck" "${COMMON_ARGS[@]}"; then FAIL=1; fi

if [[ "${QUICK}" -eq 0 ]]; then
    if ! run_check "racecheck" -t racecheck --kernel-name="${KERNEL_NAME}"; then FAIL=1; fi
    if ! run_check "initcheck" -t initcheck --kernel-name="${KERNEL_NAME}"; then FAIL=1; fi
    if ! run_check "synccheck" -t synccheck --kernel-name="${KERNEL_NAME}"; then FAIL=1; fi
fi

if [[ "${LEAK_CHECK}" -eq 1 ]]; then
  # leak-check modes disable kernel-internal detection; run separately from memcheck above.
  if ! run_check "leak_device" -t memcheck --leak-check=yes --check-device-heap=yes; then FAIL=1; fi
  if ! run_check "leak_cann" -t memcheck --leak-check=yes --check-cann-heap=yes; then FAIL=1; fi
fi

echo ""
echo "Summary written to: ${SUMMARY}"
if [[ "${FAIL}" -ne 0 ]]; then
    echo "Some checks failed. Inspect logs under: ${LOG_DIR}" >&2
    exit 1
fi
echo "All checks completed."
