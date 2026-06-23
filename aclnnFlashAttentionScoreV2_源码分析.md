# aclnnFlashAttentionScoreV2 源码分析

> 基于 flash_attention_score 仓库，全面分析 `aclnnFlashAttentionScoreV2` 接口的源码实现，澄清其是否为 AscendC 实现，以及实现的细节。

---

## 目录

1. [接口定义与版本演进](#1-接口定义与版本演进)
2. [整体架构分层](#2-整体架构分层)
3. [V2 接口源码逐层分析](#3-v2-接口源码逐层分析)
   - [3.1 Level-2 API 层（aclnn 层）](#31-level-2-api-层aclnn-层)
   - [3.2 Level-0 API 层（l0op 层）](#32-level-0-api-层l0op-层)
   - [3.3 Host 层（Tiling + InferShape）](#33-host-层tiling--infershape)
   - [3.4 Kernel 层（AscendC 核函数）](#34-kernel-层ascendc-核函数)
4. [是否是 AscendC 实现？](#4-是否是-ascendc-实现)
5. [AscendC 内核的实现细节](#5-ascendc-内核的实现细节)
   - [5.1 多模板策略（Tiling Key）](#51-多模板策略tiling-key)
   - [5.2 双矩阵乘（BMM1 + BMM2）流水线](#52-双矩阵乘bmm1--bmm2流水线)
   - [5.3 在线 Softmax（FlashV2 算法）](#53-在线-softmaxflashv2-算法)
   - [5.4 数据布局处理](#54-数据布局处理)
   - [5.5 稀疏掩码支持（Sparse Mode）](#55-稀疏掩码支持sparse-mode)
   - [5.6 Dropout 掩码预处理](#56-dropout-掩码预处理)
6. [V2 相对 V1 的关键差异](#6-v2-相对-v1-的关键差异)
7. [两段式调用流程（完整调用链）](#7-两段式调用流程完整调用链)
8. [关键文件索引](#8-关键文件索引)

---

## 1. 接口定义与版本演进

`aclnnFlashAttentionScoreV2` 是华为 CANN 平台上 FlashAttention 前向算子的第二个版本接口，声明在：

```
op_api/aclnn_flash_attention_score.h (L62-L96)
```

接口分为两段（CANN 的标准双段式 API 设计）：

```c
// 第一段：计算 workspace 大小，同时构建计算图
aclnnStatus aclnnFlashAttentionScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,       // PSE（Position Shift Encoding）
    const aclTensor *dropMaskOptional,        // Dropout 掩码
    const aclTensor *paddingMaskOptional,     // Padding 掩码
    const aclTensor *attenMaskOptional,       // Attention 掩码
    const aclIntArray *prefixOptional,        // 前缀稀疏的前缀长度
    const aclIntArray *qStartIdxOptional,     // ★ V2 新增：Q 起始索引数组
    const aclIntArray *kvStartIdxOptional,    // ★ V2 新增：KV 起始索引数组
    double scaleValue,                        // 缩放因子（通常为 1/sqrt(d)）
    double keepProb,                          // Dropout 保留概率
    int64_t preTokens,                        // 窗口前向 token 数
    int64_t nextTokens,                       // 窗口后向 token 数
    int64_t headNum,                          // Q 的 head 数（N1）
    char *inputLayout,                        // 输入布局：BSH/BSND/SBH/BNSD/TND
    int64_t innerPrecise,                     // 内部精度模式
    int64_t sparseMode,                       // 稀疏模式（0-9）
    int64_t pseType,                          // ★ V2 新增：PSE 类型
    const aclTensor *softmaxMaxOut,           // 输出：softmax 行最大值
    const aclTensor *softmaxSumOut,           // 输出：softmax 行求和
    const aclTensor *softmaxOutOut,           // 输出：softmax 全量（通常为空）
    const aclTensor *attentionOutOut,         // 输出：最终 attention 结果
    uint64_t *workspaceSize,                  // 输出：workspace 大小
    aclOpExecutor **executor);                // 输出：执行器

// 第二段：执行计算
aclnnStatus aclnnFlashAttentionScoreV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

各版本功能演进：

| 版本 | 新增特性 |
|------|---------|
| V1 (`aclnnFlashAttentionScore`) | 基础 FlashAttention 前向，支持 BSH/BSND/SBH/BNSD/TND |
| V2 (`aclnnFlashAttentionScoreV2`) | 新增 `qStartIdxOptional`/`kvStartIdxOptional`（滑窗/自定义索引）；新增 `pseType`（PSE 类型控制） |
| V3 | 新增 `sinkOptional`（Attention Sink 支持） |
| V4 | 新增 `queryRope`/`keyRope`（RoPE 融合）、量化输入、`outDtype`、Dropout seed/offset 等 |

---

## 2. 整体架构分层

整个算子由四层组成，自上而下调用：

```
用户代码
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  Level-2 API 层（aclnn 层）                              │
│  op_api/aclnn_flash_attention_score.cpp                 │
│  · 参数校验、内存连续性保证                               │
│  · 输入预处理（Reshape / Pad / Transpose）               │
│  · 调用 l0op::FlashAttentionScore 构建计算图             │
│  · 输出后处理（Transpose / Slice / Reshape）             │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│  Level-0 API 层（l0op 层）                               │
│  op_api/flash_attention_score.cpp                       │
│  · 处理可选输入（分配空 Tensor 占位）                      │
│  · 调用 INFER_SHAPE（形状推导）                           │
│  · 调用 ADD_TO_LAUNCHER_LIST_AICORE（注册 AiCore 算子）   │
└────────────────────────────┬───────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
┌─────────────────────┐              ┌──────────────────────────┐
│  Host 层 - InferShape│              │  Host 层 - Tiling         │
│  op_host/flash_      │              │  op_host/flash_attention_ │
│  attention_score_    │              │  score_tiling.cpp         │
│  infershape.cpp      │              │  · 解析硬件信息            │
│  · 推导输出 Shape    │              │  · 分 arch 选择 tiling 策略│
│  · 推导输出 DataType │              │  · 写 Tiling 参数到 GM    │
└─────────────────────┘              └──────────────┬───────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────┐
│  Kernel 层（AscendC 核函数）                             │
│  op_kernel/flash_attention_score.cpp                    │
│  op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h  │
│  op_kernel/arch22/flash_attention_score_s1_bn2gs1.h    │
│  op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h   │
│  op_kernel/arch22/flash_attention_var_len_score.h      │
│  op_kernel/arch35/flash_attention_score_kernel_train.h │
│  · 真正在 AICore 上执行的 AscendC 代码                   │
│  · BMM1 + Online Softmax + BMM2 流水线                  │
└────────────────────────────────────────────────────────┘
```

---

## 3. V2 接口源码逐层分析

### 3.1 Level-2 API 层（aclnn 层）

源文件：`op_api/aclnn_flash_attention_score.cpp`，L1289-L1385。

`aclnnFlashAttentionScoreV2GetWorkspaceSize` 实现如下（精简伪代码）：

```cpp
aclnnStatus aclnnFlashAttentionScoreV2GetWorkspaceSize(...) {
    // 1. 基础参数空指针检查
    CHECK_RET(CheckFaParam(...));

    // 2. DFX 诊断日志（记录入参）
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreV2, DFX_IN(...), DFX_OUT(...));

    // 3. 创建执行器
    auto uniqueExecutor = CREATE_EXECUTOR();

    // 4. 空 Tensor 快速返回（b/n/s 有维度为0）
    if (softmaxMaxOut->IsEmpty() && ...) {
        *workspaceSize = 0;
        return ACLNN_SUCCESS;
    }

    // 5. 格式检查（硬件限制：不允许 NZ 格式）
    if (StrideLimited()) {
        CHECK_RET(CheckFormat(...));
    }

    // 6. 数据类型检查（Q/K/V 必须相同；PSE 类型与 dtype 联动校验）
    CHECK_RET(InputDtypeCheck(..., pseType, ...));

    // 7. 解析输入布局，推导 B/N/S/D 轴信息
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, value, inputLayout, headNum, shapeInfo));
    // shapeInfo.axes.{b, n1, n2, s1, s2, d, dk, dv}
    // shapeInfo.{needPad, needTranspose, needReshape, needPadValue, padNum, padNumv}

    // 8. 保证内存连续（Contiguous）
    CHECK_RET(Contiguous(query, key, value, ...));

    // 9. 输入预处理（仅在 stride-limited 硬件上执行）
    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor));
    //   · needReshape → l0op::Reshape (BSH→BSND, SBH→SBND 等)
    //   · needPad     → l0op::Pad (对 D 维度 padding 到 16/128 对齐)
    //   · needTranspose → l0op::Transpose (BSH/SBH → BNSD)

    // 10. 调用 L0 接口，构建计算图（V2 传入 qStartIdxOptional/kvStartIdxOptional）
    auto l0Outs = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional,
        paddingMaskOptional, attenMaskOptional,
        /*sink=*/nullptr,
        prefixOptional,
        /*actualSeqQLen=*/nullptr, /*actualSeqKvLen=*/nullptr,
        qStartIdxOptional,   // ★ V2 新增
        kvStartIdxOptional,  // ★ V2 新增
        /*dScaleQ/K/V=*/nullptr, nullptr, nullptr,
        /*queryRope/keyRope=*/nullptr, nullptr,
        scaleValue, keepProb, preTokens, nextTokens, headNum,
        shapeInfo.l0InputLayoutStr.c_str(),
        innerPrecise, sparseMode,
        pseType,  // ★ V2 新增
        /*seed=*/0, /*offset=*/0, /*outDtype=*/0, /*softmaxOutLayout=*/"",
        l0Executor);

    // 11. 输出后处理（逆向变换，与预处理对称）
    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor));
    //   · needTranspose → l0op::Transpose 逆变换
    //   · padNumv != 0  → l0op::Slice 去掉 padding
    //   · needReshape   → l0op::Reshape 还原原始 shape

    // 12. 将 L0 输出 ViewCopy 到用户输出 Tensor
    l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);

    // 13. 收集 workspace 大小，交付执行器
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

第二段函数极简：

```cpp
aclnnStatus aclnnFlashAttentionScoreV2(void *workspace, uint64_t workspaceSize,
                                       aclOpExecutor *executor, const aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    // CommonOpExecutorRun 是框架函数，将执行器中记录的所有 kernel 提交到 stream 执行
}
```

**关键辅助函数 `AnalysisInput` 做了什么（L551-L646）：**

该函数解析五种布局，并决策是否需要 Pad/Transpose/Reshape：

| 布局 | Query 形状 | Key/Value 形状 | 关键映射 |
|------|-----------|---------------|---------|
| BSH  | (B, S1, N1\*D) | (B, S2, N2\*D) | d = H/N1 |
| BSND | (B, S1, N1, D) | (B, S2, N2, D) | 直接读 dim |
| SBH  | (S1, B, N1\*D) | (S2, B, N2\*D) | d = H/N1 |
| BNSD | (B, N1, S1, D) | (B, N2, S2, D) | 直接读 dim |
| TND  | (T, N1, D)    | (T, N2, D)    | 变长序列 |

当满足以下条件时触发 **Pad**（D 维度对齐）：
- 硬件为 stride-limited（DAV_2201）
- D 不是 16 或 128 的整数倍（根据 D < 196 选 16 对齐，否则 128 对齐）
- 特殊地，D=72 或 D=88 不需要 Pad

当满足以下条件时触发 **Transpose**：
- BSH/BSND 布局且 `n1 * (d + padNum) > 65535`（超出 stride 硬件限制）
- SBH 布局且 `b * n1 * (d + padNum) > 65535`

### 3.2 Level-0 API 层（l0op 层）

源文件：`op_api/flash_attention_score.cpp`。

`l0op::FlashAttentionScore` 是 aclnn 层和 host 层之间的胶水代码，主要做：

1. **填充空 Tensor**：可选输入如果为 nullptr，分配空的占位 Tensor（形状为 [0]）

2. **IntArray → Tensor 转换**：`qStartIdxOptional`、`kvStartIdxOptional`、`prefixOptional` 等 IntArray 通过 `executor->ConvertToTensor` 转为 Tensor 传给核函数

3. **InferShape**：调用 `INFER_SHAPE(FlashAttentionScore, ...)` 推导所有输出的 shape（分发到 `op_host/flash_attention_score_infershape.cpp`）

4. **注册 AiCore 算子**：调用 `ADD_TO_LAUNCHER_LIST_AICORE(FlashAttentionScore, ...)` 把算子加入执行队列，框架在 Phase-2 时提交到 AICore 执行

```cpp
OP_TYPE_REGISTER(FlashAttentionScore);  // 算子类型注册

const std::array<const aclTensor *, 4> FlashAttentionScore(...) {
    // 1. 填充空 Tensor
    if (realShiftOptional == nullptr)
        realShiftOptional = executor->AllocTensor(query->GetDataType(), ...);
    // ... 其他可选输入同理

    // 2. IntArray → Tensor（qStartIdx/kvStartIdx）
    qStartIdxOptionalTensor = executor->ConvertToTensor(qStartIdxOptional, DT_INT64);

    // 3. 推导输出 Tensor 的 shape/dtype
    auto softmaxMaxOut = executor->AllocTensor(DT_FLOAT, ...);
    auto attentionOutOut = executor->AllocTensor(outputDtype, ...);

    // 4. InferShape 调用
    ret = INFER_SHAPE(FlashAttentionScore,
        OP_INPUT(query, key, value, ..., qStartIdxOptionalTensor, kvStartIdxOptionalTensor, ...),
        OP_OUTPUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut),
        OP_ATTR(scaleValue, keepProb, preTockens, nextTockens, headNum, inputLayout,
                innerPrecise, sparseMode, pseType, seed, offset, outDtype, softmaxOutLayout));

    // 5. 注册 AiCore 核函数启动
    ret = ADD_TO_LAUNCHER_LIST_AICORE(FlashAttentionScore, OP_INPUT(...), OP_OUTPUT(...), OP_ATTR(...));

    return {softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut};
}
```

### 3.3 Host 层（Tiling + InferShape）

#### InferShape（`op_host/flash_attention_score_infershape.cpp`）

根据输入布局和 headNum 推导输出形状：

```
softmax_max  shape: (B, N1, S1, 8)      // 每 8 个 S1 对应一个 FP32 行最大值
softmax_sum  shape: (B, N1, S1, 8)      // 同上，行求和
softmax_out  shape: (0, 0, 0, 0)        // 未实际使用，填空
attention_out shape: 与 query 相同，仅 D 维替换为 value 的 Dv
```

TND 布局时 softmax_max/sum shape 为 `(T, N1, 8)`。

#### Tiling（`op_host/flash_attention_score_tiling.cpp`）

Tiling 是 AscendC 算子的核心步骤，决定如何在 AICore 上切分数据：

```cpp
ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionScore(gert::TilingContext *context) {
    // 1. 基础 shape 参数合法性校验
    CheckParams(context);

    // 2. 获取硬件信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    // 3. 空输入特殊处理（某维度为0时直接输出全零）
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        IsEmptyInputRegbase(context);  // arch35 路径
    } else {
        IsEmptyInput(context);         // arch22 路径
    }

    // 4. 分发到各 arch 的专用 Tiling 实现
    TilingRegistryArch::GetInstance().DoTilingImpl(context);
}
```

Tiling 解析阶段（`TilingPrepareForFlashAttentionScore`）从硬件平台获取：
- `aivNum`：AIV 核数
- `aicNum`：AIC 核数
- `ubSize`：UB（Unified Buffer）大小
- `l1Size`：L1 缓存大小
- `l0cSize`：L0C 缓存大小

并写入 `FlashAttentionScoreCompileInfo` 供 Tiling 函数使用。

算子注册：

```cpp
IMPL_OP_OPTILING(FlashAttentionScore)
    .Tiling(TilingFlashAttentionScore)
    .TilingInputsDataDependency({7, 8, 9, 10, 11})  // prefix/actualSeqQLen等为值依赖输入
    .TilingParse<FlashAttentionScoreCompileInfo>(TilingPrepareForFlashAttentionScore);
```

#### 算子定义（`op_host/flash_attention_score_def.cpp`）

```cpp
// 针对 ascend910b/910_93（arch22）使用标准实现
aicore_config
    .ExtendCfgInfo("opFile.value", "flash_attention_score")
    .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
this->AICore().AddConfig("ascend910b", aicore_config);
this->AICore().AddConfig("ascend910_93", aicore_config);

// 针对 ascend950（arch35）使用 APT 实现
aicore_config_95
    .ExtendCfgInfo("opFile.value", "flash_attention_score_apt")
this->AICore().AddConfig("ascend950", aicore_config_95);
```

### 3.4 Kernel 层（AscendC 核函数）

源文件：`op_kernel/flash_attention_score.cpp`。

核函数签名：

```cpp
template<uint8_t KernelTypeKey, uint8_t UB0, uint8_t UB1, uint8_t Block,
         uint8_t ImplMode, uint8_t DataType, uint8_t Layout, uint8_t Bmm1Format,
         uint8_t Bmm2Source, uint8_t Sparse, uint8_t BigDoubleBuffer,
         bool HasDropOut, bool HasAttenMask, bool HasPse, bool EnableL1Reuse, bool HasRope,
         uint8_t MatmulPolicyType, uint8_t S1TemplateType, uint8_t S2TemplateType,
         uint8_t dTemplateSize>
__global__ __aicore__ void flash_attention_score(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx,   // ← V2 传入
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *sink,
    __gm__ uint8_t *pScale,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
```

`__global__ __aicore__` 是 AscendC 标准核函数标识，表示该函数运行在 **AICore** 上。

---

## 4. 是否是 AscendC 实现？

**是的，底层计算内核完全是用 AscendC 编写的。**

证据：

1. **核函数标识**：`__global__ __aicore__` 是 AscendC 编程模型的核函数声明关键字（类比 CUDA 的 `__global__`）。

2. **AscendC 头文件**：内核文件 include 了：
   ```cpp
   #include "kernel_operator.h"    // AscendC 核心算子库
   #include "lib/matmul_intf.h"   // 矩阵乘接口
   ```

3. **AscendC 内置类型和 API**：代码大量使用 AscendC 专有 API：
   ```cpp
   using AscendC::TPipe;
   using AscendC::SoftmaxFlashV2;
   using AscendC::GetBlockIdx;
   using AscendC::GetUserWorkspace;
   AscendC::SetMaskNorm();
   ```

4. **矩阵乘宏**：`REGIST_MATMUL_OBJ` 是 AscendC 注册 Cube 矩阵乘的宏，对应 Ascend 硬件的 Cube 单元（类比 NVIDIA 的 Tensor Core）。

5. **硬件存储层次**：代码显式操作 Ascend 特有的存储层次：
   - `GM`（Global Memory，DDR）
   - `L1`（L1 Cache）
   - `L0C`（L0C Buffer，Cube 输出缓存）
   - `UB`（Unified Buffer，向量计算缓存）
   - `TSCM`（Tensor Shared Cache Memory）

6. **编译配置**：`op_host/flash_attention_score_def.cpp` 中配置：
   ```cpp
   .ExtendCfgInfo("coreType.value", "AiCore")
   .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
   ```
   表示使用 AiCore 编译，采用静态编译（提前编译内核二进制）。

---

## 5. AscendC 内核的实现细节

### 5.1 多模板策略（Tiling Key）

内核通过 20 个编译期模板参数区分不同的实现路径，核心是 `(UB0, UB1, Block, Layout)` 四个字段组成 **Tiling Key**，对应不同的计算模板：

| UB0 | UB1 | Block | Layout | 对应模板 | 适用场景 |
|-----|-----|-------|--------|---------|---------|
| 3 | 9 | 9 | ≠4 | `FlashAttentionScoreS1s2Bn2gs1SameAB` | SameAB（S2>1024，适合大序列）|
| 3 | 4 | 9 | ≠4 | `FlashAttentionScoreS1s2Bn2gs1` | S1S2 通用模板 |
| 3 | 5 | 9 | - | `FlashAttentionScoreS1Bn2gs1` | S1 模板（超长序列）|
| 9 | 9 | 0 | - | `FlashAttentionScoreBn2gs1s2B` | B 模板（小 batch，大 S） |
| 3 | 9 | 9 | 4 | `FlashAttentionVarLenScoreSameAB` | TND SameAB |
| 3 | 4 | 9 | 4 | `FlashAttentionVarLenScore` | TND 变长序列 |
| 1 | - | - | - | `FlashAttentionScoreEmptyTensor` | 空 Tensor 快速路径 |

Tiling 还决定另外两个重要参数：
- **`Bmm1Format`**：BMM1 结果格式，`ND(0)` 或 `NZ(1)`
- **`MatmulPolicyType`**：矩阵乘策略，`NORMAL(0)` 或 `UNSPLITK(1)`

### 5.2 双矩阵乘（BMM1 + BMM2）流水线

FlashAttention 的计算分两个矩阵乘阶段：

```
BMM1: S_ij = Q_i · K_j^T     形状: (S1_block, S2_block)
      ↓
Online Softmax(S_ij)          行方向 exp/max/sum
      ↓
BMM2: O_i = P_ij · V_j        形状: (S1_block, Dv)
```

在 AscendC 实现中，通过 `REGIST_MATMUL_OBJ` 宏注册两个矩阵乘对象：

```cpp
REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(),
    op.bmm1, bmm1tiling,    // 第一个矩阵乘 Q·K^T
    op.bmm2, bmm2tiling);   // 第二个矩阵乘 P·V
```

Cube 核（`__DAV_C220_CUBE__`）和 Vector 核（AIV）协同工作：
- **Cube 核**：执行矩阵乘（BMM1/BMM2），结果写入 L0C 或 GM
- **Vector 核**：执行 Softmax、掩码加法、Dropout 等逐元素操作

通过 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 声明为 **1 AIC + 2 AIV** 的混合模式，实现 Cube 和 Vector 流水并行。

### 5.3 在线 Softmax（FlashV2 算法）

使用 AscendC 内置的 `AscendC::SoftmaxFlashV2` 实现 **Online Softmax**，其数学形式为：

```
对 S1 方向的每个 query token：
  遍历所有 S2 块时，维护：
    m_i = max(m_i_prev, max(S_ij))        行最大值（在线更新）
    l_i = l_i_prev * exp(m_i_prev - m_i) + sum(exp(S_ij - m_i))  行求和
    O_i = O_i_prev * exp(m_i_prev - m_i) / exp(...) + exp(S_ij - m_i) · V_j
```

`softmaxMax`（m_i）和 `softmaxSum`（l_i）作为输出，供反向传播使用。

精度模式（`ImplMode`）：
- `AA_HIGH_PRECISION`（0）：高精度模式
- `AA_HIGH_PERFORMANCE`（1）：高性能模式（精度略降）
- `AA_INVALID_LINE_HIGH_PRECISION`（2）：含无效行的高精度模式

### 5.4 数据布局处理

由于 AscendC Cube 单元原生支持 NZ 格式（16×16 块），但 FlashAttention 输入通常为 ND 格式，代码在两种路径之间切换：

```cpp
// BMM1 输出为 NZ 格式路径（适合 S2 很大时减少转置开销）
INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ...
    half, float, true, CubeFormat::NZ, ...);

// BMM1 输出为 ND 格式路径（通用路径）
INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ...
    half, float, true, CubeFormat::ND, ...);
```

当 `Bmm2Source = TSCM`（非零）时，BMM2 的 P 矩阵暂存于 TSCM（昇腾特有的共享 L2 片内存储），减少 GM 访问延迟。

内核中的 `NzToNd` 函数（`flash_attention_score_common.h` L231-L293）实现了 NZ 到 ND 的转置：通过 `DataCopy` 搬运 + `vcopy` 实现 16×16 块的转置。

### 5.5 稀疏掩码支持（Sparse Mode）

`sparseMode` 参数支持 10 种稀疏模式（在 `SparseModeEnum` 枚举中定义）：

| 值 | 模式名 | 含义 |
|----|-------|------|
| 0 | ALL | 全注意力 |
| 1 | NONE | 无掩码 |
| 3 | CAUSAL | 因果掩码（下三角）|
| 4 | BAND | 带状掩码（preTokens/nextTokens 控制窗口）|
| 5 | PREFIX | 前缀注意力（配合 prefix 数组）|
| 6 | BAND_COMPRESS | 压缩带状掩码 |
| 7 | RIGHT_DOWN_CAUSAL | 右下因果掩码 |

`AttenMaskComputeMode` 进一步细化掩码计算策略，包括 `CAUSAL_OR_NEXT_ONLY_MODE`、`PREFIX_COMPUTE_MODE` 等，由 Tiling 阶段计算并写入 Tiling Data，内核按需读取。

### 5.6 Dropout 掩码预处理

当输入 `dropMask` 为 bit 位掩码（非字节掩码）时，需要在正式计算前做预处理：

```cpp
// tiling 阶段判断是否需要 DropMask 转换
if (tilingData->inputParams.needDropMaskOp) {
    FlashAttentionScoreDropMaskAdapter dropMaskAdapter;
    dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);
    dropMaskAdapter.Process();   // 将 bit mask 展开为 byte mask，写入 workspace
    tPipe.Reset();               // 复位 pipeline 后继续主计算
}
```

`FlashAttentionScoreDropMaskAdapter` 使用 `SelectWithBytesMask` AscendC API 完成 bit→byte 的格式转换。

---

## 6. V2 相对 V1 的关键差异

| 差异点 | V1 | V2 |
|-------|----|----|
| `qStartIdxOptional` | 无 | 有（Q 的起始 token 索引，每个 batch 一个值）|
| `kvStartIdxOptional` | 无 | 有（KV 的起始 token 索引，每个 batch 一个值）|
| `pseType` | 固定为 1（外部 PSE，add+mul 模式）| 可配置：1=外部; 2=Inner AliBI(mul+add); 3=Inner AliBI(mul+add+sqrt) |

**`qStartIdxOptional` / `kvStartIdxOptional` 的作用**：

允许对每个 batch 内的序列指定自定义的注意力窗口起始位置，实现**滑窗注意力**（Sliding Window Attention）或**分块注意力**。配合 `preTokens`/`nextTokens` 参数，可精确控制每个 query token 能看到的 key/value 范围：

```
有效注意力范围 = [qStartIdx[b] - preTokens, qStartIdx[b] + nextTokens]
```

**`pseType` 的作用**：

| 值 | 名称 | 含义 | PSE 输入要求 |
|----|------|-----|------------|
| 1 | PSE_TYPE_V1 | 外部加法 PSE | dtype 与 attention_out 相同 |
| 2 | PSE_INNER_MUL_ADD | Inner AliBI: score = score * pse + pse | PSE dtype 必须为 FP32 |
| 3 | PSE_INNER_MUL_ADD_SQRT | Inner AliBI with sqrt: `score * pse + pse / sqrt(d)` | PSE dtype 必须为 FP32 |

---

## 7. 两段式调用流程（完整调用链）

```
用户代码
│
├── [Phase 1] aclnnFlashAttentionScoreV2GetWorkspaceSize(...)
│       │
│       ├── CheckFaParam()               // 空指针检查
│       ├── InputDtypeCheck()            // 数据类型校验
│       ├── AnalysisInput()              // 解析布局，决定 Pad/Transpose
│       ├── Contiguous()                 // 保证内存连续
│       ├── PreprocessQKV()             // Reshape / Pad / Transpose
│       ├── l0op::FlashAttentionScore() // 构建计算图
│       │       ├── INFER_SHAPE()       // → InferShapeFlashAttentionScore()
│       │       │       └── 推导 softmax_max/sum/out/attention_out shape
│       │       └── ADD_TO_LAUNCHER_LIST_AICORE()  // 注册内核启动参数
│       │               └── → TilingFlashAttentionScore()
│       │                       ├── CheckParams()     // shape 校验
│       │                       ├── IsEmptyInput()    // 空 Tensor 快速路径
│       │                       └── DoTilingImpl()    // 分 arch 计算 Tiling
│       │                               ├── arch22: flash_attention_score_tiling_general.cpp
│       │                               └── arch35: flash_attention_score_tiling_regbase.cpp
│       ├── Postprocess()               // 逆 Transpose / Slice / Reshape
│       ├── l0op::ViewCopy(×3)          // 输出 Tensor 挂接
│       └── 返回 workspaceSize + executor
│
└── [Phase 2] aclnnFlashAttentionScoreV2(workspace, workspaceSize, executor, stream)
        │
        └── CommonOpExecutorRun()
                └── 将 executor 中所有注册的内核提交到 stream
                        └── 在 AICore 上执行 flash_attention_score kernel
                                ├── 根据 Tiling Key 选择模板路径
                                ├── [可选] DropMaskAdapter 预处理
                                ├── BMM1: Q·K^T
                                ├── Online Softmax (SoftmaxFlashV2)
                                ├── 掩码加法 / Dropout
                                └── BMM2: P·V → attention_out
```

---

## 8. 关键文件索引

| 文件 | 层次 | 作用 |
|------|------|------|
| `op_api/aclnn_flash_attention_score.h` | Level-2 API | V2 接口声明（L62-L96）|
| `op_api/aclnn_flash_attention_score.cpp` | Level-2 API | V2 实现（L1289-L1385）|
| `op_api/flash_attention_score.h` | Level-0 API | l0op 接口声明 |
| `op_api/flash_attention_score.cpp` | Level-0 API | l0op 实现，含 InferShape/Launcher 注册 |
| `op_host/flash_attention_score_def.cpp` | Host | 算子定义，支持的 dtype/format，硬件配置 |
| `op_host/flash_attention_score_infershape.cpp` | Host | 输出 shape/dtype 推导 |
| `op_host/flash_attention_score_tiling.cpp` | Host | Tiling 入口，分 arch 分发 |
| `op_host/arch22/flash_attention_score_tiling_general.cpp` | Host | arch22（910B）Tiling 实现 |
| `op_host/arch35/flash_attention_score_tiling_regbase.cpp` | Host | arch35（950）Tiling 实现 |
| `op_kernel/flash_attention_score.cpp` | Kernel | AscendC 核函数入口，多模板分发 |
| `op_kernel/arch22/flash_attention_score_common.h` | Kernel | 公共类型、常量、工具函数 |
| `op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h` | Kernel | S1S2 通用计算模板 |
| `op_kernel/arch22/flash_attention_score_s1s2_bn2gs1_sab.h` | Kernel | SameAB 计算模板 |
| `op_kernel/arch22/flash_attention_score_s1_bn2gs1.h` | Kernel | S1 超长序列模板 |
| `op_kernel/arch22/flash_attention_score_bn2gs1s2_b.h` | Kernel | B 模板（小 batch）|
| `op_kernel/arch22/flash_attention_var_len_score.h` | Kernel | TND 变长序列模板 |
| `op_kernel/arch35/flash_attention_score_kernel_train.h` | Kernel | arch35（950）BaseAPI 实现 |
| `op_kernel/arch35/flash_attention_score_entry_regbase.h` | Kernel | arch35 核函数入口 |
