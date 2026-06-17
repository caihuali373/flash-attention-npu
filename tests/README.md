# Flash Attention NPU v3 反向测试使用说明

本目录提供 **用例生成** 与 **批量回归测试** 两套脚本，用于验证 `flash_attn_npu_v3._flash_attn_backward`（tridao 路径）与 `torch_npu.npu_fusion_attention` autograd 反向（golden 路径）的一致性，并支持 nan 检测、显存统计、确定性校验及 profiler 性能采集。

## 目录结构

| 文件 | 说明 |
|------|------|
| `generate_cases.py` | 随机生成 BSND / TND 用例 xlsx |
| `test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py` | 从 xlsx 读取用例并执行测试 |
| `fag_cases_BSND_drop.xlsx` | BSND 定长用例（示例/生成产物） |
| `fag_cases_TND_drop.xlsx` | TND 变长用例（示例/生成产物） |
| `prof_bwd/` | 开启 `--enable_perf` 后的 profiler 输出目录（自动创建） |

## 环境要求

- 已安装并配置 CANN、`torch`、`torch_npu`
- 已编译安装 `flash-attention-npu_final`（可 `import flash_attn_npu_v3`）
- Python 依赖：`pandas`、`openpyxl`（读写 xlsx）
- 性能采集需 NPU 环境支持 `torch_npu.profiler`

建议在 `tests/` 目录下执行命令。

---

## 一、生成用例

```bash
cd /data/lfz/test/test_/flash-attention-npu_final/tests

# 生成 BSND 用例（默认 220 条）
python3 generate_cases.py BSND

# 生成 TND 用例
python3 generate_cases.py TND
```

输出文件：

- `fag_cases_BSND_drop.xlsx`
- `fag_cases_TND_drop.xlsx`

### 用例字段说明

**公共列**

| 列名 | 说明 |
|------|------|
| `Enable` | `enable` 执行该用例，`disable` 跳过 |
| `case_name` | 用例名，如 `BSND_case_001` |
| `batch` | batch 大小（1~18） |
| `nheads` | Q head 数 |
| `nheads_k` | KV head 数（`nheads` 为其整数倍） |
| `headdim` | head 维度（64/128/192/256） |
| `window_size_left` / `window_size_right` | 滑动窗口，当前固定为 -1 |
| `layout` | `BSND` 或 `TND` |
| `dtype` | `half` 或 `bf16` |
| `is_deterministic` | 0/1，是否做 FA 反向确定性循环比对 |
| `keep_prob` | 传给 `npu_fusion_attention` 的 dropout 保留概率（默认 1.0） |
| `is_causal` | 0/1，是否 causal mask |

**BSND 专用列**

| 列名 | 说明 |
|------|------|
| `seq_q` | Q 序列长度 |
| `seq_k` | KV 序列长度 |

**TND 专用列**

| 列名 | 说明 |
|------|------|
| `list_seq_q` | 每个 batch 的 Q 真实长度 list，长度为 `batch` |
| `list_seq_kv` | 每个 batch 的 KV 真实长度 list，长度为 `batch` |

---

## 二、执行测试

### 基本用法

```bash
# BSND 批量测试（默认 layout 为 BSND）
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py \
  --device 4 \
  --test_layout BSND \
  --case_file fag_cases_BSND_drop.xlsx

# TND 批量测试
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py \
  --device 4 \
  --test_layout TND \
  --case_file fag_cases_TND_drop.xlsx
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | NPU 设备 ID |
| `--test_layout` | `BSND` | 测试分支：`BSND` 或 `TND` |
| `--case_file` | `fag_cases_<layout>_drop.xlsx` | 用例 xlsx 路径（支持相对路径，会在 `tests/` 下查找） |
| `--enable_perf` | 关闭 | 开启 `torch_npu.profiler` 性能采集 |
| `--prof_path` | `tests/prof_bwd` | profiler trace 输出根目录 |
| `--case_names` | 全部 | 仅运行指定 `case_name`，可传多个或逗号分隔 |
| `--isolate_cases` | 开启 | 每条用例在独立 Python 子进程中执行（默认开启） |
| `--no_isolate_cases` | — | 在当前进程内连续跑用例（快，但可能残留 NPU/profiler 状态） |

### 用例进程隔离（默认开启）

批跑时若所有用例在同一 Python 进程内连续执行，可能出现：

- NPU 显存碎片 / 缓存未完全释放
- `torch.use_deterministic_algorithms(True)` 从前一条用例残留
- `torch_npu.profiler` 全局状态污染
- `check_nan()` 后的显存布局影响后续用例

这会导致 **同一条用例** 在「单独跑 `--case_names TND_case_003`」与「全表批跑中的 TND_case_003」结果不一致。

**默认行为（`--isolate_cases`）**：主进程编排，每条用例使用独立子进程，并在用例之间做 **NPU 设备级 reset**：

```
主进程
  ├─ 复制 xlsx → fag_cases_TND_drop_<timestamp>.xlsx
  ├─ [device-reset]
  ├─ functional worker: 精度 + nan + 显存 + 确定性（不跑 profiler）
  ├─ [device-reset]
  ├─ perf worker（仅 --enable_perf）: 单独进程采集性能
  ├─ [device-reset]
  └─ 下一条用例 ...
```

批跑与单跑不一致的常见根因是：**前序用例在 NPU 设备上残留状态**（仅靠 Python 子进程隔离无法清除），或 profiler 污染确定性测试。当前方案将功能测试与性能采集拆到不同子进程，并在每条用例前后重置设备。

确定性比对以**首次** `_flash_attn_backward` 的 `dq/dk/dv` 为基准，后续 10 次均在 `torch.npu.manual_seed(2)` 后重跑比对。

调试时可临时关闭隔离（不推荐用于正式批跑）：

```bash
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py \
  --device 5 --test_layout TND --enable_perf --no_isolate_cases
```

### 按 case_name 筛选运行

```bash
# 单条用例
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py \
  --device 4 --test_layout BSND --case_names BSND_case_001

# 多条用例（空格分隔）
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py \
  --device 4 --test_layout BSND \
  --case_names BSND_case_001 BSND_case_005 BSND_case_010

# 多条用例（逗号分隔）
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py \
  --device 4 --test_layout TND \
  --case_names TND_case_001,TND_case_003
```

- 需 xlsx 中存在 `case_name` 列（由 `generate_cases.py` 生成）
- 仅 `Enable=enable` 的用例会执行；`disable` 行会被跳过（无 `Enable` 列时默认执行）
- 仍会复制整表 xlsx 为时间戳副本，但只执行并回写匹配行的结果
- 若某个 `case_name` 不存在，会打印 warning；全部未匹配则报错退出

### 保存日志示例

```bash
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py \
  --device 4 \
  --test_layout BSND \
  --case_file fag_cases_BSND_drop.xlsx \
  --enable_perf \
  2>&1 | tee test_tridao_fag_v3_bsnd_performence_nan_memory.log
```

---

## 三、测试内容说明

每条用例依次执行以下检查（BSND / TND 均支持）。

### 1. 精度比对（核心）

- **Golden**：`npu_fusion_attention` 前向 + `out.backward(dout)` 得到 `dq/dk/dv`
- **被测**：仅调用 `_flash_attn_backward`，输入 golden 的 `out` 与 `softmax_lse`
- 使用 `precision_compare.data_compare` 比对 `dq`、`dk`、`dv`

### 2. NaN 检测

- 用例开始前执行 `check_nan()`：在剩余显存中写入 NaN 后释放，用于暴露显存污染问题
- 反向结束后检查 `dq/dk/dv` 是否含 NaN

### 3. 显存统计

| 指标 | 含义 |
|------|------|
| `torch_bwd_mem` | golden 路径反向阶段峰值显存增量（MB） |
| `tridao_bwd_mem` | `_flash_attn_backward` 反向峰值显存增量（MB） |

### 4. Causal 适配（`is_causal=1`）

- Golden：`atten_mask = triu(ones[2048,2048], diagonal=1)`，`sparse_mode=2`
- FA bwd：`is_causal=True`

### 5. 确定性校验（`is_deterministic=1`）

- 开启 `torch.use_deterministic_algorithms(True)`
- `_flash_attn_backward(..., deterministic=True)` 首次执行后，再循环 **10 次**
- 每次与首次 `dq/dk/dv` 做 `torch.equal` 比对

### 6. 性能采集（`--enable_perf`）

仿照 `dailyauto_fusion_common_D1D2.py`，使用 `torch_npu.profiler`：

- **torch_bwd**：循环 10 次 `npu_fusion_attention` 前向 + `backward`，解析 `FlashAttentionScoreGrad` kernel 耗时
- **tridao_bwd**：循环 10 次仅 `_flash_attn_backward`，解析 tridao/FA 反向 kernel 耗时

Profiler 配置要点：`Level2`、`PipeUtilization`、`warmup=1, active=1, skip_first=5`。

---

## 四、结果回写

运行前会将输入 xlsx **复制一份带时间戳的副本**，例如：

```
fag_cases_BSND_drop_20260616_154516.xlsx
```

每跑完一条用例，向该副本追加/更新结果列：

| 结果列 | 说明 |
|--------|------|
| `nan_result` | `success` / `fail` |
| `torch_bwd_mem` | golden 反向显存（MB） |
| `tridao_bwd_mem` | tridao 反向显存（MB） |
| `compare_result` | 精度（及确定性）比对 `success` / `fail` |
| `torch_bwd_kernel_ms` | golden 反向 kernel 耗时（ms，需 `--enable_perf`） |
| `tridao_bwd_kernel_ms` | tridao 反向 kernel 耗时（ms，需 `--enable_perf`） |
| `torch_prof_path` | golden profiler 输出目录 |
| `tridao_prof_path` | tridao profiler 输出目录 |

终端结束时会打印汇总：`total / success / failed` 及失败用例详情。

---

## 五、性能 Profiler 输出路径

开启 `--enable_perf` 后，trace 目录结构示例：

```
prof_bwd/
└── BSND_20260616_154516/          # 或 TND_<timestamp>
    ├── BSND_case_001_torch_bwd/
    └── BSND_case_001_tridao_bwd/
```

可用 TensorBoard 打开对应目录进一步分析；`kernel_details.csv` 位于各子目录下的 `ASCEND_PROFILER_OUTPUT/`。

---

## 六、参数注意事项

1. **`keep_prob`** 仅传给 `npu_fusion_attention`；`_flash_attn_backward` 无此参数。
2. **`--test_layout`** 必须与 xlsx 中 `layout` 列一致；脚本按 layout 分支读取不同列（BSND 读 `seq_q/seq_k`，TND 读 `list_seq_q/list_seq_kv`）。
3. **`--case_file`** 省略时默认使用 `fag_cases_BSND_drop.xlsx` 或 `fag_cases_TND_drop.xlsx`（由 `--test_layout` 决定）。
4. 单条用例异常不会中断整表执行；异常用例结果列记为 fail，并继续下一条。

---

## 七、典型工作流

```bash
cd /data/lfz/test/test_/flash-attention-npu_final/tests

# 1. 生成用例
python3 generate_cases.py BSND
python3 generate_cases.py TND

# 2. 功能 + nan + 显存 + 确定性
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py --device 4 --test_layout BSND
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py --device 4 --test_layout TND

# 3. 附加性能采集
python3 -u test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py \
  --device 4 --test_layout BSND --enable_perf --prof_path ./prof_bwd \
  2>&1 | tee bsnd_perf.log
```

---

## 八、常见问题

**Q: 日志出现 `nan check failed: 'module' object is not callable`**

A: 已修复：应使用 `torch.npu.empty_cache()`，勿写 `torch_npu.npu().empty_cache()`。请使用最新版 `test_flash_attn_npu_v3_bwd_only_for_dtm_causal.py`。

**Q: `compare_result=fail` 但显存正常**

A: 检查 `dq/dk/dv` 精度是否达标；若 `is_deterministic=1`，还需确定性 10 次循环全部 `torch.equal` 通过。

**Q: profiler 耗时为 0**

A: 确认 `--enable_perf` 已开启、NPU profiler 环境无冲突，并检查 `torch_prof_path` 下是否存在 `kernel_details.csv`。
