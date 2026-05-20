# Attention Daily — 2026-05-20

## TL;DR

- **SGLang 新增 SM90 FP8 Sparse MLA Prefill JIT 内核**：PR [#25751](https://github.com/sgl-project/sglang/pull/25751) 引入 Q8KV8 格式稀疏 MLA prefill JIT kernel（Hopper），相比 BF16 基准实现 1.15–1.31x 吞吐提升，为 MLA 内核路线图（#25746）第 1/2 步。
- **TensorRT-LLM DSv4 MLA 解码延迟降低 26–29%**：PR [#14254](https://github.com/NVIDIA/TensorRT-LLM/pull/14254) 在 SM100（GB200）上默认开启 DSv4 MLA `o_a_proj` 路径的 fused inv-RoPE+FP8 BMM，大批量（M≥2048）场景延迟显著下降。
- **vLLM 修复 FlashInfer/Triton metadata 中 `num_qo_heads` 取值 bug**：PR [#42650](https://github.com/vllm-project/vllm/pull/42650) 修正非标准 head 配置（如 MLA 变体）下 illegal memory access 问题，从 Attention 层直接读取而非推断自 `model_config`。
- **SGLang 将 NSA 正式重命名为 DSA**：PR [#25821](https://github.com/sgl-project/sglang/pull/25821) 将 Native Sparse Attention 全面更名为 DeepSeek Sparse Attention，含 CLI flags、环境变量、文件名，保留向后兼容别名。
- **TRT-LLM FP8 KV cache 布局与上游 FlashMLA/SGLang 存在差异**：Issue [#14327](https://github.com/NVIDIA/TensorRT-LLM/issues/14327) 社区详细指出 TRT-LLM 的 per-tensor FP8 方案与 DeepSeek-V4 参考实现（MX-FP8 + BF16 rope）不一致，已分配维护者跟进。

---

## vLLM

### Open PRs 值得关注

- **[#42650](https://github.com/vllm-project/vllm/pull/42650)** FlashInfer/Triton attention metadata 修复 — `num_qo_heads`/`num_heads_q` 改为从各 Attention 层直接读取，修复非标准 head 数模型（如 MLA 变体）下的 illegal memory access。
- **[#42956](https://github.com/vllm-project/vllm/pull/42956)** [ROCm][MLA] 修复 gfx950 上 FP8 KV cache 稀疏 MLA decode 产生空/乱输出的问题 — AITER decode kernel 缺失 `q.fp8 + kv.fp8` 路径（MI35x, TP=4, DeepSeek V3.2）。
- **[#43162](https://github.com/vllm-project/vllm/pull/43162)** [DSV4] Q pad 操作融入 `qk_rope_norm` CUDA kernel — 消除 TP 场景 head 数低于 kernel 最小值时的额外 `F.pad` 调用。
- **[#43149](https://github.com/vllm-project/vllm/pull/43149)** DeepSeek V4 稀疏 MLA 实现迁入 model 目录 — 将 V4 sparse-MLA forward 从 `vllm.v1.attention.backends` 移到 `vllm/models/deepseek_v4/attention/`，平台检测提前到 `__init__`。
- **[#39168](https://github.com/vllm-project/vllm/pull/39168)** [ROCm] 扩展稀疏 MLA 支持 — 修复非 16 倍数 block stride 的 cache kernel bug，新增 block-size 16 及支持任意 block size 的 Triton kernel。
- **[#41834](https://github.com/vllm-project/vllm/pull/41834)** SM12x Blackwell 消费卡（RTX 5090、DGX Spark）适配 DeepSeek V4 Flash/FlashMLA — 回退至不依赖 TMEM 的替代路径，绕过 DeepGEMM/FlashMLA/Marlin FP8 的 Blackwell 数据中心限制。
- **[#43142](https://github.com/vllm-project/vllm/pull/43142)** KV offloading connector 支持 DSv4 — 修复多 KV cache group 时的 block size 解析及空 `shared_by` 字段的 tensor 过滤。
- **[#42749](https://github.com/vllm-project/vllm/pull/42749)** [AMD] Qwen3-30B-A3B 在 ROCm AITER 后端启用 QK Norm + RoPE + KV Cache 运行时 fusion（Inductor，覆盖 AITER FA 和 AITER Unified 两种后端）。

### Open Issues 值得关注

- **[#43094](https://github.com/vllm-project/vllm/issues/43094)** 双向 KV transfer 在推理链被截断时产生静默错误 — 思维链 token 被裁剪后，KV cache 对应 token 与实际输入不匹配。
- **[#42426](https://github.com/vllm-project/vllm/issues/42426)** Kimi-K2.6 高并发下输出乱码 — 疑似 FA4 MLA prefill 或 FLASHINFER_MLA decode 在高并发下触发。
- **[#42803](https://github.com/vllm-project/vllm/issues/42803)** MiMo-V2 fused QKV weight loader 将 Q 权重误置于 K/V slot — 8 TP rank 中 7 个 rank 的 attention 计算结果严重错误。

### Commits 直推 main

（本日无法通过 API 直接枚举 commits，上述 merged/open PR 记录为等效替代。）

---

## SGLang

### Open PRs 值得关注

- **[#25751](https://github.com/sgl-project/sglang/pull/25751)** SM90 Q8KV8 FP8 Sparse MLA Prefill JIT Kernel — Hopper 新增 FP8 稀疏 MLA prefill kernel，Q/KV 均为 INT8，相比 BF16 基准加速 1.15–1.31x；MLA kernel 路线图 [#25746](https://github.com/sgl-project/sglang/issues/25746) 第 1 步。
- **[#25821](https://github.com/sgl-project/sglang/pull/25821)** NSA → DSA 全局重命名 — 所有 CLI flags、env vars、注册表项、文件名及类名从 NSA 改为 DSA（DeepSeek Sparse Attention），保留 NSA 向后兼容别名。
- **[#25824](https://github.com/sgl-project/sglang/pull/25824) / [#25825](https://github.com/sgl-project/sglang/pull/25825)** KV cache pool 抽象重构（步骤 2c/2d）— 将 `swa_loc`、`kv_cache_dtype`、`is_swa` 提升到 `ForwardBatch`，`start_layer` 改为构造注入，使 attention 层彻底脱离对 KV pool 对象的直接引用。
- **[#25741](https://github.com/sgl-project/sglang/pull/25741)** 修复 chunked prefill 调度 bug — 批次中已有请求占用部分 chunk budget 时，调度器未能允许新请求入队，影响实际吞吐。
- **[#25173](https://github.com/sgl-project/sglang/pull/25173)** NIXL HiCache KV 卸载重构 + O_DIRECT 支持 — 修复线程同步和对象存储 bug，O_DIRECT I/O 显著提升 KV cache 卸载吞吐。
- **[#24376](https://github.com/sgl-project/sglang/pull/24376)** 修复 NIXL HiCache MLA key 分母错误和 backup 跳过 bug — 防止 MLA backup 流程中的静默 cache 污染。
- **[#25776](https://github.com/sgl-project/sglang/pull/25776)** Ascend NPU 后端新增 Gemma4 Sliding Window Attention 支持。

### Open Issues 值得关注

- **[#25823](https://github.com/sgl-project/sglang/issues/25823)** SM100 上 `--attention-backend flashinfer` + `--enable-deterministic-inference` + Qwen3-VL 崩溃 — 视觉编码器触发 DeepGEMM TMA descriptor 非标 shape（N=1076）导致 `CUDA_ERROR_INVALID_VALUE`。
- **[#25672](https://github.com/sgl-project/sglang/issues/25672)** GLM-5 在 AMD MI325X + aiter 后端编译报错 — MLA-style `qk_nope_head_dim=192`（非 2 的幂）触发 Triton `arange range must be a power of 2`。

### Commits 直推 main

（同上，以 merged PR 记录为准。）

---

## TensorRT-LLM

### Merged PRs

- **[#14276](https://github.com/NVIDIA/TensorRT-LLM/pull/14276)** 修复 ADP router 中 `attention_dp_relax` 为 None 时的崩溃 — `DefaultADPRouter` 和 `KVCacheAwareADPRouter` 在该字段未设置时因 NoneType 比较崩溃，现改为安全默认（允许 relaxed routing）。
- **[#13630](https://github.com/NVIDIA/TensorRT-LLM/pull/13630)** AutoDeploy 支持 Gemma3n/4 E2B 变体 — 新增 shared KV attention 支持，Triton paged attention 增加 per-sequence KV 长度处理与内存优化。

### Open PRs 值得关注

- **[#14254](https://github.com/NVIDIA/TensorRT-LLM/pull/14254)** DSv4 MLA `o_a_proj` 默认开启 fused inv-RoPE+FP8 BMM — SM100（GB200）大批量（M≥2048）解码延迟降低 26–29%；同步重调 Triton kernel 的 BTM-dispatch。
- **[#12525](https://github.com/NVIDIA/TensorRT-LLM/pull/12525)** trtllm-gen FlashInfer FMHA 优化 — 将 KV metadata 构建移入 C++，消除 Python host-side 开销，默认启用 trtllm-gen attention，相比先前 trtllm-gen 基准提升约 19% 吞吐。
- **[#14297](https://github.com/NVIDIA/TensorRT-LLM/pull/14297)** 修复 DSv4 稀疏 attention 索引器 CUDA Graph 内存安全 bug — 预分配持久 `radix_aux_indices`/`radix_aux_logits` 缓冲区，避免 `torch.empty` 临时指针在 CUDA Graph replay 时失效导致崩溃。
- **[#13821](https://github.com/NVIDIA/TensorRT-LLM/pull/13821)** Ring Attention 及统一 Context Parallel 用于视觉生成（WAN 2.2）— 近线性扩展至 16 个 GB200。
- **[#13815](https://github.com/NVIDIA/TensorRT-LLM/pull/13815)** 精确多模态 KV block hash — 修复非连续多模态 token span（如交错视频帧）的 KV prefix cache 哈希，提升 VLM 生产场景缓存命中率。
- **[#14060](https://github.com/NVIDIA/TensorRT-LLM/pull/14060)** 混合 Mamba 模型支持 prefix reuse 的 disaggregated serving — 扩展 C++ transceiver 在 TP 不匹配节点间传输 KV cache block 和 RNN 状态。

### Open Issues 值得关注

- **[#14327](https://github.com/NVIDIA/TensorRT-LLM/issues/14327)** DeepSeek-V4 SWA/COMPRESS pool FP8 KV cache 布局与上游参考不一致 — TRT-LLM 使用 per-tensor FP8（rope 也量化），而 FlashMLA/SGLang 参考使用 MX-FP8（nope FP8 E4M3 + rope BF16 + per-tile UE8M0 scale，每 token 584B vs 512B），社区关注长上下文精度影响，维护者已认领跟进。

---

## Blogs & 长文

- **[A First Comprehensive Study of TurboQuant: Accuracy and Performance](https://vllm.ai/blog/2026-05-11-turboquant)** — vLLM 官方 benchmark，覆盖四种 TurboQuant KV cache 量化变体（k8v4、4bit_nc 等）与 BF16/FP8 基准对比；KV cache 压缩至 3–4 bit 可显著节省显存，但 attention 计算仍在 BF16 上进行，吞吐与精度有取舍。（2026-05-11）

- **[DeepSeek V4 in vLLM: Efficient Long-context Attention](https://vllm-project.github.io/2026/04/24/deepseek-v4.html)** — 深入分析 DeepSeek-V4 混合 attention（shared KV、c4a/c128a 压缩 token pooling、128-token SWA）；1M 上下文 KV 仅 9.62 GiB（vs V3.2 的 83.9 GiB，减少 8.7x）；详述 inverse RoPE 实现与 disaggregated serving。（2026-04-24）

- **[HiSparse: Turbocharging Sparse Attention with Hierarchical Memory](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)** — SGLang 团队介绍层级稀疏 attention，结合 FlashMLA 和 FlashAttention-3 Sparse 后端，NSA 后端专为 DeepSeek MLA 式稀疏工作负载设计。（2026-04-10）

- **[Accelerating Long-Context Inference with Skip Softmax in NVIDIA TensorRT-LLM](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/)** — Skip Softmax Attention（稀疏 attention 变体）融入 Hopper/Blackwell FlashAttention kernel，50% 稀疏度下 TTFT 和 TPOT 最高提升 1.4x，无需重训练，支持自动阈值校准。（2026-03-16）

---

## 横向观察

今日三个框架集中在 **DeepSeek-V4 MLA 的工程化与硬件适配**上同步推进：vLLM 在修复 ROCm FP8 路径并重构 V4 模块目录，SGLang 在推出 Hopper FP8 MLA prefill kernel 并完成 NSA→DSA 品牌化重命名，TRT-LLM 在优化 GB200 上的 MLA decode 延迟并修复 CUDA Graph 内存安全 bug。三家均在布局 **Blackwell（SM100/SM120）消费卡和数据中心卡的双线适配**，DeepSeek-V4 的稀疏 MLA 已成为 2026 年 attention 实现的核心工程压力测试场景。TRT-LLM FP8 KV 布局讨论（#14327）也暗示各框架在 MX-FP8 和 per-tensor FP8 之间的路线选择将出现分化。
