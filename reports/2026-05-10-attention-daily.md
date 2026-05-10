# Attention Daily — 2026-05-10

> 时间窗口：2026-05-09T01:05Z — 2026-05-10T01:05Z

## TL;DR

- **TensorRT-LLM 重磅合并**：DeepSeekV4 专属 attention kernel 正式入库（[#13652](https://github.com/NVIDIA/TensorRT-LLM/pull/13652)），同日还合并了 Cute-DSL FP8 Paged MQA decode kernel（[#13219](https://github.com/NVIDIA/TensorRT-LLM/pull/13219)），Blackwell 系推理生态进一步夯实。
- **SGLang 修复 MLA 路径上的 KV skip 逻辑**：[#24097](https://github.com/sgl-project/sglang/pull/24097) 将 `fa_skip_kv_cache` 限制为非 MLA 后端，避免 MLA 模型的 KV 被错误跳过。
- **vLLM 两份 RFC 值得持续关注**：KV cache 布局标准化（[#42082](https://github.com/vllm-project/vllm/issues/42082)）和 V1 调度器缓存亲和性排序（[#42185](https://github.com/vllm-project/vllm/issues/42185)）均在本日活跃讨论。
- **SGLang HISA 稀疏 attention PR 仍在 review**：声称在 DeepSeek Native Sparse Attention 上实现 1.65× 加速（[#24672](https://github.com/sgl-project/sglang/pull/24672)）。
- **TensorRT-LLM MLA 回退 bug**：Issue [#13939](https://github.com/NVIDIA/TensorRT-LLM/issues/13939) 披露 MLA fallback 路径将 KV cache reuse 关闭后未同步更新 attention 运行时标志，形成 split-brain 状态。

---

## vLLM

### Merged PRs

（今日无可确认在 24 小时窗口内 merge 的 attention 相关 PR）

### Open PRs / Issues 值得关注

- **[#42082](https://github.com/vllm-project/vllm/issues/42082)** `[RFC] Standardize KV-cache Layouts` — 提议统一不同 attention 后端（FlashAttention、FlashInfer、Triton MLA 等）的 KV cache 内存布局，解决当前各后端布局不一致带来的互操作性问题。本日活跃讨论。

- **[#42185](https://github.com/vllm-project/vllm/issues/42185)** `[RFC] Cache-affinity-aware request ordering for the V1 scheduler` — 提议在 V1 调度器中按前缀缓存命中率对请求重排序，降低 prefix cache miss 率、提升吞吐。

- **[#41803](https://github.com/vllm-project/vllm/pull/41803)** `[Kernel][MLA] Triton-fused TurboQuant decode backend` — 为 DeepSeek-V2-Lite 设计的 Triton 融合 TurboQuant 量化 MLA decode kernel，4×RTX 4090 + TP=4 基准已就绪，等待 directional review。

- **[#42175](https://github.com/vllm-project/vllm/pull/42175)** `[Core][Model] Gemma4: Unified FA4 for all layers + FlashAttention mm_prefix support` — 将 Gemma4 所有层统一到 FlashAttention-4，并为 multimodal prefix 添加 FlashAttention 支持。

- **[#41797](https://github.com/vllm-project/vllm/pull/41797)** `[Attention] add triton diff-kv backend for mimo` — 为 MIMO 模型新增 Triton diff-KV attention 后端。

- **[#41093](https://github.com/vllm-project/vllm/pull/41093)** `[P/D][Mooncake] Add cross-layer KV cache support to MooncakeConnector` — 在分布式 Prefill/Decode 场景下，MooncakeConnector 支持跨层 KV cache 传输，降低 P/D 互联带宽压力。

- **[#42172](https://github.com/vllm-project/vllm/pull/42172)** `[Bugfix][KVConnector] Include prompt_embeds digest in LMCacheMP cache key` — 修复 LMCache 在多模态场景下 cache key 未包含 `prompt_embeds` 内容导致的缓存碰撞问题。

- **[#42024](https://github.com/vllm-project/vllm/issues/42024)** `[Bug] NIXL connector silently disables HMA, halving KV cache capacity` — NIXL connector 静默禁用 HMA（Host Memory Allocator），导致 KV cache 容量减半，属于高影响 bug。

- **[#42179](https://github.com/vllm-project/vllm/issues/42179)** `[Bug] FP8 KV cache corrupts output in Qwen3.5-397B-NVFP4 Disagg serving` — Qwen3.5 分布式服务时 FP8 KV cache 导致输出损坏，影响 NVFP4 量化 + 分离式推理场景。

### Commits 直推 main

（MCP 访问限制，无法直接获取 commit 列表；上述 PR 均为 PR 视角覆盖）

---

## SGLang

### Merged PRs

- **[#24097](https://github.com/sgl-project/sglang/pull/24097)** `Restrict fa_skip_kv_cache to non-MLA backends` — 将 FlashAttention 的 `skip_kv_cache` 优化路径限制为非 MLA 后端，防止 MLA 模型（DeepSeek 系列）的 KV 被错误跳过导致精度问题。已于 2026-05-09 merge。

### Open PRs / Issues 值得关注

- **[#24737](https://github.com/sgl-project/sglang/pull/24737)** `Support Flashinfer Cute-DSL MLA attention` — 引入基于 FlashInfer Cute-DSL 实现的 MLA attention 后端，利用 CUTLASS DSL 提升 Hopper/Blackwell 上 MLA 的计算效率。

- **[#24672](https://github.com/sgl-project/sglang/pull/24672)** `[Feature] HISA: hierarchical sparse indexer for DeepSeek Sparse Attention (1.65×)` — 分层稀疏索引器，针对 DeepSeek Native Sparse Attention，通过层级化索引减少无效 KV 访问，bench 显示 1.65× 吞吐提升。

- **[#24640](https://github.com/sgl-project/sglang/pull/24640)** `Support spec v2 for FlashMLA speculative decoding` — FlashMLA + speculative decoding v2 协议支持，将推测解码与 MLA 解码路径深度融合。

- **[#23515](https://github.com/sgl-project/sglang/pull/23515)** `[Disagg] Layer-pipelined KV transfer: overlap RDMA with GPU compute` — 分离式推理中，通过逐层流水线化 KV 传输，将 RDMA 网络传输与 GPU 计算重叠，降低 P/D disaggregation 延迟。

- **[#24781](https://github.com/sgl-project/sglang/pull/24781)** `RolloutKV: Trainer-Informed Prefix KV Lifecycle Management` — 为 RL 训练场景设计，训练器主动通知推理引擎 prefix KV 的生命周期，减少冗余重计算和不必要的 KV 驱逐。

- **[#24857](https://github.com/sgl-project/sglang/pull/24857)** `Optimize SWA memory preallocation for disaggregated decode` — 针对 Sliding Window Attention 在分离式 decode 场景下的内存预分配进行优化，减少 OOM 风险。

- **[#24762](https://github.com/sgl-project/sglang/pull/24762)** `[AMD] fix(triton-mla): clamp max_kv_splits by attn_logits buffer budget (Kimi-K2)` — 修复 Kimi-K2 在 AMD 平台 Triton-MLA 后端中 `max_kv_splits` 超出 attn logits 缓冲区预算的问题。

- **[#24758](https://github.com/sgl-project/sglang/issues/24758)** `[Bug] HiCache file/L3 storage can reload KV pages across incompatible model identities` — HiCache 存储在不同模型间复用 KV page 时缺乏模型身份校验，可能加载到不兼容的缓存页（PR #24794 正在修复）。

- **[#24853](https://github.com/sgl-project/sglang/issues/24853)** `[Bug] flashinfer attention backend produces output non-determinism under concurrent requests` — 并发场景下 FlashInfer attention 后端输出不确定，影响需要确定性推理的场景。

### Commits 直推 main

（同上，访问限制）

---

## TensorRT-LLM

### Merged PRs

- **[#13652](https://github.com/NVIDIA/TensorRT-LLM/pull/13652)** `[feat] Add DeepSeekV4 attention kernels` — 正式合并 DeepSeek V4 专属 attention kernel，覆盖 DeepSeek-V4 的 MLA 变体实现，是 TRTLLM 支持 DSv4 高性能推理的核心基础。已于 2026-05-09 merge。

- **[#13219](https://github.com/NVIDIA/TensorRT-LLM/pull/13219)** `[TRTLLM-34871][feat] Add cute dsl FP8 paged MQA logits decode kernel` — 基于 CUTLASS Cute DSL 实现 FP8 精度的 Paged MQA logits decode kernel，提升 Blackwell GPU 上多查询注意力解码的计算效率。已于 2026-05-09 merge。

- **[#12932](https://github.com/NVIDIA/TensorRT-LLM/pull/12932)** `[feat] Add Gemma4 multimodal model support (text + vision + audio)` — Gemma4 多模态支持落地，含文本/视觉/音频 attention 路径。已于 2026-05-09 merge。

### Open PRs / Issues 值得关注

- **[#13929](https://github.com/NVIDIA/TensorRT-LLM/pull/13929)** `[TRTLLM-35237][feat] Add cute dsl FP4 paged MQA logits decode kernel` — FP8 版本合并后，FP4 版本的 Paged MQA kernel 随即跟进，进一步压缩计算精度以换取更高吞吐。

- **[#13937](https://github.com/NVIDIA/TensorRT-LLM/pull/13937)** `[refactor] Decouple cached prefix from KVSlice token_range` — 将 prefix cache 逻辑从 KVSlice 的 token range 中解耦，改善 KV cache 管理模块的可维护性。

- **[#13938](https://github.com/NVIDIA/TensorRT-LLM/pull/13938)** `[feat] Keep DSv4 o_a_proj as FP8, and port vLLM's fused_inv_rope_fp8_quant` — 保持 DeepSeek V4 输出投影为 FP8，并从 vLLM 移植融合的逆 RoPE FP8 量化 kernel。

- **[#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713)** `[fix] Disaggregated KV transfer: lifecycle, cancellation, and...` — 修复分离式 KV 传输中的生命周期管理和取消逻辑，提升 P/D disaggregation 稳定性。

- **[#13939](https://github.com/NVIDIA/TensorRT-LLM/issues/13939)** `[Bug] MLA fallback disables KV cache reuse in config but leaves attention runtime flag stale` — MLA 回退路径（SM 版本不支持或 KV 量化不兼容时）正确关闭了 `kv_cache_config.enable_block_reuse`，但未同步更新 `attn_runtime_features.cache_reuse`，导致 KV manager 与 attention 运行时形成 split-brain，可能触发错误的 cached-KV 执行路径和静默精度问题。

---

## Blogs & 长文

- **[HiSparse: Turbocharging Sparse Attention with Hierarchical Memory](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)** — SGLang 团队介绍 HiSparse 系统：将非活跃 KV cache 条目主动卸载到 host 内存，GPU HBM 只保留热 buffer，配合 hierarchical memory 层级（GPU→CPU→Disk）动态管理 KV，显著缓解长上下文 GPU 内存压力。（发布于 2026-04-10，本日无新博文）

- **[DeepSeek-V4 on Day 0: Fast Inference to Verified RL with SGLang](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)** — 介绍 SGLang 对 DeepSeek-V4 的 Day-0 支持，包括 ShadowRadix prefix cache、FlashMLA 推测解码、HiSparse CPU 扩展 KV 等 attention 相关优化组合。（发布于 2026-04-25）

---

## 横向观察

**今日最显著的横向趋势是 DeepSeek V4 attention 支持的集体冲刺：**

- TensorRT-LLM 今日合并了 DSv4 attention kernel，是三家中生产化程度最高的动作；
- SGLang 有大量针对 DSv4 MLA、AMD MI300 上 DSv4 KV pool、以及 EAGLE+DSv4 的 open PR 正在 review；
- vLLM 则有多个针对 ROCm 和 NVIDIA GPU 的 DSv4/MLA 变体 PR 处于 open 状态。

**FP8/FP4 KV 量化 kernel 是 TensorRT-LLM 本日的另一条主线**：Cute-DSL FP8 MQA 合并、FP4 版本紧随其后，反映 Blackwell（SM120）上 sub-byte precision attention 的成熟度快速提升。

**SGLang 在 KV cache 分层管理（HiCache/HiSparse/RolloutKV）和 disaggregated prefill 层面的工程积累**较另外两家更为系统化，且 HISA 稀疏 attention 若顺利合并将是近期三家中最具新颖性的 kernel-level 优化。
