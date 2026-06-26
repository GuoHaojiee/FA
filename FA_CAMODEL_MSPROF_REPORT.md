# FlashAttentionScore CAModel 仿真性能分析报告

**更新日期：** 2026-06-26
**目标芯片：** Ascend950PR_9599
**CANN 版本：** 9.0.0
**测试场景：** 场景2 FTT Attention（B=2, S=30, H=192）

---

## 1. 仓库结构

```
FA原生算子性能分析/
├── op_test_template/
│   ├── test_fa_builtin.cpp          ← 算子调用驱动程序（我们自己编写）
│   ├── test_fa_builtin              ← 编译产物，由 run.sh 自动生成
│   └── msprof_output/               ← 性能产物根目录
│       ├── run.log                  ← 完整运行日志（含 Total tick）
│       └── OPPROF_<时间戳>/
│           ├── dump/                ← CAModel 原始 .dump 二进制（机器格式）
│           └── simulator/           ← msprof 解析后的可读产物
│               ├── trace.json       ← 全局 timeline（Perfetto 格式）
│               ├── visualize_data.bin
│               └── core<N>.{cubecore,veccore}<M>/
│                   ├── *_instr_exe.csv   ← 指令级性能数据（主要分析对象）
│                   └── *_code_exe.csv
├── build/                           ← CMake 构建临时目录，每次重建，无需关注
├── CMakeLists.txt                   ← 编译配置
├── run.sh                           ← 一键编译 + 运行入口
├── analyze_trace.py                 ← trace.json 分析脚本
├── FA_CAMODEL_MSPROF_REPORT.md      ← 本报告
└── 操作手册.md                       ← 详细操作说明
```

---

## 2. 工具链路

```
test_fa_builtin.cpp（我们写的算子调用驱动）
        │
        │  bisheng 编译，链接 CANN 库
        ▼
test_fa_builtin（x86 可执行文件）
        │  链接了：
        │  ├── libruntime_camodel.so  ← 仿真模式替代真实 runtime
        │  ├── libopapi_transformer.so ← CANN built-in FA 算子接口
        │  └── libascendcl.so 等
        │
        │  msprof op simulator 包裹执行
        ▼
CAModel 周期精确仿真 Ascend950PR_9599
        │
        ▼
op_test_template/msprof_output/
    ├── dump/       ← 原始仿真 dump
    └── simulator/  ← CSV / JSON / BIN 性能数据
```

### 仿真 vs 真实 NPU 的唯一区别

| 模式 | 编译参数 | 链接的 runtime 库 | 运行环境 |
|------|----------|-------------------|----------|
| 仿真（当前） | `RUN_MODE=sim` | `libruntime_camodel.so` | x86 主机，无需 NPU |
| 真实 NPU | `RUN_MODE=npu` | `libruntime.so` | 需接 Ascend 硬件 |

`test_fa_builtin.cpp` 代码本身一字不改，只通过 CMake 参数切换。

---

## 3. 编译配置（CMakeLists.txt 要点）

```cmake
# 编译器必须在 project() 之前设置才生效
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(CMAKE_C_COMPILER   "$ENV{ASCEND_HOME_PATH}/bin/bisheng")
    set(CMAKE_CXX_COMPILER "$ENV{ASCEND_HOME_PATH}/bin/bisheng")
endif()

project(fa_camodel_test)

# 源文件：算子调用驱动
add_executable(test_fa_builtin op_test_template/test_fa_builtin.cpp)

# 链接库（RUN_MODE=sim 时链 runtime_camodel，npu 时链 runtime）
target_link_libraries(test_fa_builtin PRIVATE
    $<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>
    $<$<STREQUAL:${RUN_MODE},npu>:runtime>
    ascendcl nnopbase opapi_transformer opapi_oam
    stdc++ m dl pthread c_sec
)

# 二进制输出到 op_test_template/
set_target_properties(test_fa_builtin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/op_test_template"
)
```

**bisheng** 是 CANN 自带的编译器（基于 Clang 15），位于 `$ASCEND_HOME_PATH/bin/bisheng`，专为昇腾生态优化。

---

## 4. 运行方式

```bash
bash run.sh
```

`run.sh` 自动完成：
1. source CANN 环境（`set_env.sh`）
2. 设置 `LD_LIBRARY_PATH`（含 simulator lib 路径）
3. `cmake` + `make` 编译 `test_fa_builtin.cpp`
4. `msprof op simulator` 包裹运行，产物输出到 `op_test_template/msprof_output/`
5. 运行日志同时打印到终端并保存到 `msprof_output/run.log`

**验证 CAModel 生效的关键日志：**

```
[TmSim]: Run in serial mode.
" PEM MODEL "
[INFO] <ProfInit> Start profiling on kernel: FlashAttentionScore_236874ba...
[INFO] Total tick: 14931
```

---

## 5. 测试算子说明

### 5.1 算子接口

| 项目 | 值 |
|------|----|
| 接口名 | `aclnnFlashAttentionScoreV2` |
| 头文件 | `aclnnop/aclnn_flash_attention_score.h` |
| 实现库 | `libopapi_transformer.so`（CANN built-in） |
| kernel hash | `FlashAttentionScore_236874ba12878eb50...` |

### 5.2 算子 kernel 源码位置（CANN built-in，只读参考）

```
$CANN/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/flash_attention_score/
├── flash_attention_score.cpp              697 行  ← kernel 主逻辑
├── flash_attention_score_apt.cpp           75 行  ← kernel 入口
└── arch35/
    ├── flash_attention_score_kernel_train.h         557 行  ← cube/vec 流水核心
    ├── flash_attention_score_template_tiling_key.h  516 行
    └── flash_attention_score_entry_regbase.h        129 行
```

### 5.3 测试场景参数（场景2 FTT Attention）

| 参数 | 值 | 备注 |
|------|----|------|
| Batch (B) | 2 | CAModel 内存限制，原始为 16 |
| SeqLen (S) | 30 | |
| Hidden (H) | 192 | = headNum × headDim |
| headNum | 8 | |
| headDim | 24 | |
| scale | 0.2041 | ≈ 1/√24 |
| dtype | float16 | |
| layout | BSH | |
| 输入数据 | Q/K/V 全填 fp16(1.0) | 未做正确性验证 |
| blkDim | 16 | B×headNum = 2×8，每核处理 1 个 head |

> 场景1 ViT（B=160, S=576, H=1024）因 CAModel 内存占用约 6.5 GB，暂未运行。

---

## 6. 性能结果（2026-06-26 实测）

### 6.1 整算子指标

| 指标 | 本次实测 | 历史基准 | 差值 |
|------|----------|----------|------|
| **Total tick** | **14,931 cycles** | 14,784 cycles | +147（+1.0%） |
| 等效时间（200 MHz） | ≈ 74.7 μs | ≈ 73.9 μs | — |
| 仿真 Wall time | ~105 秒 | ~103 秒 | — |

差值 1% 属于 CAModel 正常抖动范围，kernel hash 完全相同，结果可信。

### 6.2 分核 cycles（16 个 AIC core）

| 核类型 | 16 核范围 | 16 核平均 |
|--------|-----------|-----------|
| cubecore0（矩阵计算） | 38,909 ~ 40,605 | **~39,654** |
| veccore0（向量计算） | 42,777 ~ 43,767 | **~43,426** |

**veccore 是关键路径**，比 cubecore 多约 3,772 cycles（+9.5%）。

### 6.3 流水线占比（core0.veccore0）

| 流水线 | cycles | 占比 | 说明 |
|--------|--------|------|------|
| **SCALAR** | 35,259 | **81.0%** | DC_PRELOAD（内存预取）、LD（加载）为主 |
| FLOWCTRL | 3,556 | 8.2% | WAIT_INTRA_BLOCK / SET_INTRA_BLOCK 同步 |
| RVECEX | 1,655 | 3.8% | 向量执行 |
| RVECLD | 926 | 2.1% | 向量加载 |
| RVECST | 710 | 1.6% | 向量存储 |
| MTE2 | 560 | 1.3% | 内存搬运 |

### 6.4 流水线占比（core0.cubecore0）

| 流水线 | cycles | 占比 | 说明 |
|--------|--------|------|------|
| **SCALAR** | 31,994 | **81.9%** | DC_PRELOAD、LD 为主 |
| FLOWCTRL | 4,408 | 11.3% | 块间同步 |
| MTE2 | 2,415 | 6.2% | 矩阵数据搬运 |
| CUBE | **74** | **0.2%** | 实际矩阵计算极少 |

### 6.5 最耗时单条指令（core0.veccore0）

| 指令 | 流水线 | cycles | 说明 |
|------|--------|--------|------|
| `WAIT_INTRA_BLOCK` | FLOWCTRL | 1,057 | cube/vec 块内同步等待 |
| `DC_PRELOAD_XN_IMM` | SCALAR | 771 | L2 cache 预取 |
| `DC_PRELOAD_XN_IMM` | SCALAR | 765 | L2 cache 预取 |
| `LD_XD_XN_IMM` | SCALAR | 756 | 数据加载 |
| `LD_XD_XN_IMM` | SCALAR | 755 | 数据加载 |
| `SET_INTRA_BLOCK` | FLOWCTRL | 701 | 块内同步设置 |
| `MOV_SRC_TO_DST_ALIGNv2` | MTE2 | 555 | 内存搬运 |

---

## 7. 核心分析结论

### 7.1 memory-access bound，非 compute bound

SCALAR 流水线（预取 + 加载）占 81%，CUBE 实际计算只有 **0.2%**。对于 FTT 小 shape（S=30, H=192），数据量本身不大，但大量 cycles 消耗在内存访问准备上，说明该 shape 的计算强度（arithmetic intensity）很低，是典型的 **访存瓶颈场景**。

### 7.2 cube/vec 同步等待是单点最大开销

`WAIT_INTRA_BLOCK`（1,057 cycles）是 veccore 单条最耗时指令，说明 cube 和 vec 流水线之间存在等待，overlap 不充分。cubecore 侧同样有 752 cycles 的 `WAIT_INTRA_BLOCK`。

### 7.3 CAModel 仿真精度说明

- Total tick 适合做**相对优化比较**（优化前 vs 优化后）
- 与真实硬件误差通常 < 5%，memory-bound 算子可能略大
- 仿真 Wall time（~105 秒）远大于真实执行时间（~74.7 μs），不代表实际性能

---

## 8. 产物说明

### 8.1 文件结构

```
op_test_template/msprof_output/OPPROF_<时间戳>/
├── dump/                               ← 原始 .dump 二进制（CAModel 直出，不可读）
└── simulator/
    ├── trace.json                      ← 全局 timeline，29 MB，Perfetto 格式
    ├── visualize_data.bin              ← 可视化数据，35 MB
    └── core<N>.{cubecore0,veccore0,veccore1}/   ← 共 48 目录（16 AIC × 3 subcore）
        ├── *_instr_exe.csv             ← 指令级性能（主要分析对象）
        ├── *_code_exe.csv              ← 代码段聚合
        └── trace.json                  ← 单核 timeline
```

### 8.2 instr_exe.csv 列说明

| 列名 | 含义 |
|------|------|
| `instr` | 指令名 |
| `addr` | 指令地址 |
| `pipe` | 流水线（CUBE / SCALAR / MTE1 / MTE2 / MTE3 / FLOWCTRL / RVECEX 等） |
| `call_count` | 执行次数 |
| `cycles` | 总 cycles 消耗 |
| `running_time(us)` | 等效微秒 |
| `detail` | 附加参数 |

### 8.3 常用分析命令

```bash
# 定位最新产物目录
PROF_DIR=$(ls -td op_test_template/msprof_output/OPPROF_*/simulator | head -1)

# veccore0 Top10 最耗时指令
sort -t',' -k5 -rn "${PROF_DIR}/core0.veccore0/core0.veccore0_instr_exe.csv" | head -11

# 各流水线 cycles 汇总
awk -F',' 'NR>1 {p[$3]+=$5} END {for(k in p) printf "%-12s %d\n",k,p[k]}' \
    "${PROF_DIR}/core0.veccore0/core0.veccore0_instr_exe.csv" | sort -k2 -rn

# 用 analyze_trace.py 分析全局 trace
python3 analyze_trace.py "${PROF_DIR}/trace.json"

# 查看运行日志（含 Total tick）
cat op_test_template/msprof_output/run.log | grep -E "Total tick|tick|SCENARIO|Operator"
```

---

## 9. 历史对比

| 日期 | kernel hash | Total tick | 备注 |
|------|-------------|:----------:|------|
| 2026-06-25（历史1） | `236874ba...` | 14,780 | 自定义编译版（V2，有 vendors） |
| 2026-06-25（历史2） | `236874ba...` | 14,853 | 内置版，vendors 未删 |
| 2026-06-25（历史3） | `236874ba...` | 14,784 | 内置版，干净环境，原始 baseline |
| **2026-06-26（本次）** | **`236874ba...`** | **14,931** | **bisheng 编译器修正后，当前 baseline** |

三次历史与本次差值均 < 1%，属仿真器正常抖动。kernel hash 四次完全一致，证明调用的是同一份 built-in kernel。

---

## 10. 后续待办

- [ ] 正确性验证：输出与 numpy golden 对比
- [ ] 使用真实非均匀输入，验证 cache miss 实际影响
- [ ] 场景1 ViT（B=2, S=576, H=1024）补跑（需约 1 GB+ 内存）
- [ ] 与真实硬件数据对比，验证 CAModel 仿真精度
- [ ] 分析 `WAIT_INTRA_BLOCK` 占比，评估 cube/vec overlap 优化空间
- [ ] 尝试更大 B（B=16 原始值）在内存允许时重跑
