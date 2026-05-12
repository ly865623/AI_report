# Attention Daily — 2026-05-12

## TL;DR

- **vLLM #42294**：ROCm MI355X 上 DeepSeek V3.2 的 MLA FP8 汇编级 prefill 核，实测吞吐量提升 40%、TTFT 降低 49%，正在评审 [→](https://github.com/vllm-project/vllm/pull/42294)
- **vLLM #42316**：将 DeepSeek V4 的 FlashInfer 稀疏 MLA 核移植进 vLLM 主线，支持 per-tensor FP8 KV cache，正在评审 [→](https://github.com/vllm-project/vllm/pull/42316)
- **SGLang #25006**：将 trtllm_mha 设为 Gemma4 在 SM100（Blackwell）上的默认注意力后端，吞吐提升 22%，待 FlashInfer 版本升级后合入 [→](https://github.com/sgl-project/sglang/pull/25006)
- **SGLang #25001**：为 DeepSeek 风格的 MLA 补全 LoRA 支持（q_b_proj / kv_b_proj），使 Kimi-K2.5 类适配器可正常工作 [→](https://github.com/sgl-project/sglang/pull/25001)
- **TRT-LLM #13996**：DFlash 投机解码引入持久化 K/V Cache，Qwen3-8B 在 8×B200 上吞吐提升 14.5% [→](https://github.com/NVIDIA/TensorRT-LLM/pull/13996)

---

## vLLM

### Merged PRs

（时间窗口内无 attention 相关 PR 完成合并）

### Open PRs / Issues 值得关注

- **[#42294](https://github.com/vllm-project/vllm/pull/42294)** `[ROCm][MLA]` FP8 ASM prefill + 稀疏 MLA on Aiter — 在 gfx950/MI355X 上为 DeepSeek V3.2 引入 FP8 汇编级 prefill 核（`mla_prefill_ps_asm_fwd` + `mla_reduce_v1`），稀疏 MLA decode 重构为继承 dense AiterMLA 基类，同步修复 paged-MQA-logits indexer 的缓冲区尺寸和布局错误；实测 output throughput **+40.4%**，TTFT **-49.2%**，TPOT **-18.0%**。

- **[#42316](https://github.com/vllm-project/vllm/pull/42316)** Port DeepSeek V4 FlashInfer 稀疏 MLA kernels — 将 DSV4 专用的 FlashInfer 稀疏 MLA 核纳入 vLLM，新增 Triton 核融合 Q 归一化、RoPE 和 KV cache 写入；默认 BF16，可选 per-tensor FP8 KV cache（显式指定时启用），141 项 pytest 全通过。

- **[#42258](https://github.com/vllm-project/vllm/pull/42258)** `[Core][DSV4]` 跳过永远无法命中的 SWA prefix cache 块 — 对 DSV4 混合 SWA/全注意力架构，在 `cache_blocks()` 中新增 `alignment_tokens` 掩码，过滤掉 SWA 右扫窗口之外永不可达的块，减少无效缓存占用；非混合模型行为不变。

- **[#42309](https://github.com/vllm-project/vllm/pull/42309)** `[Core][Feat]` 可插拔 KVCacheConfigBuilder — 以策略模式将 KV cache 规划逻辑抽出，允许平台层或模型层按三级优先级（平台 > 模型 > 默认）覆盖 KV cache spec → groups → tensors 的全流程。

- **[#42345](https://github.com/vllm-project/vllm/pull/42345)** `[KV Cache]` 支持将 SWA 层排除在 NVFP4 KV cache 之外（Draft）— 对混合架构模型，允许仅对全注意力层应用 NVFP4 量化，SWA 层沿用原有精度。

### Commits 直推 main

（时间窗口内未发现绕过 PR 直推 main 的 attention 相关 commit）

---

## SGLang

### Merged PRs

- **[#25013](https://github.com/sgl-project/sglang/pull/25013)**（merged May 11）EAGLE speculative decoding `hidden_size` 路由重构 — 将 EAGLE Draft 的 `hidden_size`/`dtype` 内联查找改为通过 dataclass classmethod 统一解析，消除 EAGLE-3 `*3` 宽度扩展的隐式分支，对齐 CUDA graph 缓冲区分配行为。

- **[#25010](https://github.com/sgl-project/sglang/pull/25010)**（merged May 11）trtllm_mla 内核参数清理 — 删除 `fused_recurrent_kda_fwd` 中未使用的参数，修正 `trtllm_mla_backend` 中两处过时注释（误将 `seq_lens_q` 描述为 accept length）；无行为变更。

### Open PRs / Issues 值得关注

- **[#25006](https://github.com/sgl-project/sglang/pull/25006)** Enable trtllm_mha 为 Gemma4 在 SM100 的默认 attention backend — 已获批准，待 FlashInfer 0.6.10.post1（headdim=512 支持）合并后入库；SM100 上 trtllm_mha 较 triton 后端文本吞吐提升 **22.5%**，端到端延迟降低 12.2%。

- **[#25001](https://github.com/sgl-project/sglang/pull/25001)** `[LoRA]` MLA attention LoRA：q_b_proj / kv_b_proj 支持 — 将 DeepSeek 风格 MLA 的 LoRA 可覆盖投影从 2 个扩展至全部 4 个；`kv_b_proj` 在 absorbed MLA 路径下采用 Triton 分解核沿 LoRA-A/B 边界因式化，避免大中间矩阵实例化，支持混合秩批次。

- **[#25022](https://github.com/sgl-project/sglang/pull/25022)** `[Bugfix, NSA HiCache]` 修复 attach_hybrid_nsa_pool 中缺失的 kv_cache_dim 覆盖 — NSA 与 HiCache 混用时，`attach_hybrid_nsa_pool_to_hiradix_cache` 未传递 `override_kv_cache_dim`，导致 KV cache 维度形状错误。

- **[#24993](https://github.com/sgl-project/sglang/pull/24993)** `[HiCache]` Agent-Aware KV Cache 驱逐策略 — 在 RadixCache TreeNode 中注入多智能体工作流元数据（workflow ID、执行步距等），构建 AgentStepGraph（BFS 计算步间距离），指导缓存驱逐优先保留即将被调度的请求的 KV cache，含 18 项单元测试。

- **[#24982](https://github.com/sgl-project/sglang/pull/24982)** 修复 AMX GQA extend attention — Intel AMX CPU 后端 GQA extend 路径中 softmax 概率写入布局错误，导致 brgemm 消耗错误列数据、输出重复 token；修复为仅转换 brgemm 实际消耗的列，并附回归测试。

### Commits 直推 main

（时间窗口内未发现相关 commit 直推）

---

## TensorRT-LLM

### Merged PRs

（时间窗口内无 attention 相关 PR 完成合并；#13995 初版已关闭，被 #13996 取代）

### Open PRs / Issues 值得关注

- **[#13996](https://github.com/NVIDIA/TensorRT-LLM/pull/13996)** `[perf]` DFlash 投机解码性能优化 — 用固定大小的持久化 KV 缓冲池替代逐层上下文重算，目标模型 forward 的 token 数由 2K 压缩为 K+1，并新增 Q/K 归一化+RoPE 融合核；Qwen3-8B 在 8×B200 batch=64 时吞吐 4422→5064 tok/s（**+14.5%**）；CI 因版权头缺失暂未通过，等待修复。

- **[#14005](https://github.com/NVIDIA/TensorRT-LLM/pull/14005)**（Draft）向 KV cache connector 暴露已存储的 block hash chain — 在 C++ `commitAndGetBlockHashesForRequest` 中首次访问时缓存 block hash，避免 `storeBlocks` 的二次计算；仅支持 beam_width=1，适用于 prefix sharing 场景。

- **[#14003](https://github.com/NVIDIA/TensorRT-LLM/pull/14003)** `[fix]` KVCacheManager 为 CUDA graph padding 预留递归状态槽 — 防止 recurrent 架构（如 Mamba）在启用 CUDA graph 时因 recurrent state 槽不足发生 OOM。

- **[#13992](https://github.com/NVIDIA/TensorRT-LLM/pull/13992)** 更新 FlashInfer-python 0.6.10 → 0.6.11 — 例行依赖升级，跟进 FlashInfer 最新版本。

### Commits 直推 main

（时间窗口内未发现相关 commit 直推）

---

## Blogs & 长文

- **[The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)**（vLLM Blog，2026-04-22）— 系统梳理 FP8 KV cache 与注意力量化的精度/性能取舍：128k+ 上下文时 KV cache 可降至 BF16 的 54% 内存，精度损失 ≤0.7 点；FlashAttention-3 支持 per-head FP8 scale，提供 `--kv-cache-dtype-skip-layers` 实现分层混合精度。

- **[DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)**（LMSYS Blog，2026-04-25）— SGLang 与 Miles 在 DSV4 发布当天完成推理与 RL 训练全栈支持，重点介绍对混合稀疏注意力（NSA）、FP4 专家权重和超连接（mHC）的系统级优化。

---

## 横向观察

本时间窗口内，三个框架均在高强度推进 **DeepSeek V3.2/V4 的 MLA 注意力优化**：vLLM 同时进行 ROCm 汇编核移植（#42294）与 FlashInfer 稀疏 MLA 集成（#42316）；SGLang 在完善 MLA LoRA 支持的同时即将落地 trtllm_mha 后端（#25006）；TRT-LLM 则在 DFlash 投机解码中引入持久化 KV 缓存以减少冗余计算（#13996）。DeepSeek V4 混合 SWA 架构对 prefix cache 友好性的破坏也成为 vLLM #42258 的直接驱动——三家均面临相同的「SWA 块无效缓存」问题。此外，FlashInfer 作为跨框架的底层 attention kernel 库，其版本升级（0.6.10→0.6.11）正在被 TRT-LLM 和 SGLang 同步跟进，体现出生态层面统一依赖的趋势。
