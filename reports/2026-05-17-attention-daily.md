# Attention Daily — 2026-05-17

> 时间窗口：2026-05-16T01:00Z — 2026-05-17T01:00Z  
> 数据来源：GitHub PR/Issue 搜索（`updated:>=2026-05-16`）；commit 直推数据因 API 权限限制无法获取；blog 无当日新文。

---

## TL;DR

- **vLLM RFC**: #42826 提议将 FlashAttention forward 拆分为独立的 prefill / decode 两条路径，有望成为后续 kernel 优化的结构性改变。[链接](https://github.com/vllm-project/vllm/issues/42826)
- **SGLang 合并**: #25481 修复了 FlashAttention-3 在复用 radix 前缀时的非确定性输出问题，影响使用 FA3 + RadixAttention 的所有部署。[链接](https://github.com/sgl-project/sglang/pull/25481)
- **vLLM ROCm 融合提速**: 连续两个 PR（#42838、#42832）推进 ROCm 路径的 MLA concat 融合和 RoPE+FP8 联合量化，DeepSeek-V3.2 是主要受益模型。[#42838](https://github.com/vllm-project/vllm/pull/42838) / [#42832](https://github.com/vllm-project/vllm/pull/42832)
- **TensorRT-LLM**: FlashInfer MLA 后端（#13428）仍在评审，同日新开 PR #14200 为 Eagle3 推测解码器启用 sliding window attention。[#13428](https://github.com/NVIDIA/TensorRT-LLM/pull/13428) / [#14200](https://github.com/NVIDIA/TensorRT-LLM/pull/14200)
- **SGLang bug**: DeepSeek-V4-Pro MQA kernel 在 JIT 编译时超出 GPU shared memory 限制（#25484），影响 DeepSeek-V4 系列模型。[链接](https://github.com/sgl-project/sglang/issues/25484)

---

## vLLM

### Merged PRs

- **#42782** 修复 `--kv-cache-dtype` 命令行参数被 checkpoint 中的 `kv_cache_scheme` 覆盖的问题，确保显式 CLI 配置优先级正确。[链接](https://github.com/vllm-project/vllm/pull/42782)

### Open PRs 值得关注

- **#42838** `[ROCm][MLA]` 将 sparse-MLA `forward_mqa` 中的 `torch.cat` 替换为 fused concat 算子，减少 DeepSeek-V3.2 ROCm 路径上的 kernel launch 开销。[链接](https://github.com/vllm-project/vllm/pull/42838)
- **#42832** `[ROCm]` 将 RoPE、静态 Q FP8 量化与 KV cache 写入融合为单一 primitive，服务于 decode graph 场景。[链接](https://github.com/vllm-project/vllm/pull/42832)
- **#42828** `[KVConnector]` 为 MooncakeStoreConnector 增加 HMA（Hybrid Memory Architecture）支持，适配 DSv4 等混合注意力模型。[链接](https://github.com/vllm-project/vllm/pull/42828)
- **#42746** `[Qwen3.5]` 将 GDN linear attention 的 qkv/z/b/a 四个投影权重融合为单一 GEMM，降低显存带宽消耗。[链接](https://github.com/vllm-project/vllm/pull/42746)
- **#42740** `[CPU]` 为 CPU attention backend 显式声明 KV cache 布局要求（HND 格式：Head, N-tokens/block, Dim）。[链接](https://github.com/vllm-project/vllm/pull/42740)
- **#42637** `[TurboQuant]` 为 Gemma 4 模型启用混合注意力 KV 量化（mixed-attention KV quant），支持 per-layer 差异化精度。[链接](https://github.com/vllm-project/vllm/pull/42637)
- **#42792** `[WIP][Model Runner V2]` 为 Mamba hybrid 模型在 Model Runner V2 中对齐 prefix caching 与 spec decode 的支持。[链接](https://github.com/vllm-project/vllm/pull/42792)
- **#41847** 将 KV transfer connector 的 HMA 模式从 opt-in 改为 opt-out，默认为支持的 connector 开启。[链接](https://github.com/vllm-project/vllm/pull/41847)

### Issues 值得关注

- **#42826** `[RFC]` 提议拆分 FlashAttention forward 为 prefill/decode 两条独立路径，为不同阶段的专项优化奠定结构基础。[链接](https://github.com/vllm-project/vllm/issues/42826)
- **#42846** `[Bug]` NIXL + FlashInfer 在 Qwen3 MRV2、`--block-size 128` 时 CI 失败，怀疑 block size 对齐问题。[链接](https://github.com/vllm-project/vllm/issues/42846)
- **#42839** `[Bug]` P2pNcclConnector 在 disaggregated prefill 中因 `request_id` 不匹配导致挂死，KV tensor key 使用了 engine-local ID 而非全局 ID。[链接](https://github.com/vllm-project/vllm/issues/42839)
- **#42808** `[Bug]` TurboQuant attention backend 在 v0.21.0 中启动首个请求时 workspace lock 断言失败，与 spec decode 共用时必现。[链接](https://github.com/vllm-project/vllm/issues/42808)

### Commits 直推 main

（API 权限限制，无法访问，本日跳过。）

---

## SGLang

### Merged PRs

- **#25481** 修复 FlashAttention-3 在 radix cache 复用前缀 KV 做 extend batch 时的非确定性输出，确保 deterministic 模式正确性。（#25479 为同一修复的早期草稿，已关闭。）[链接](https://github.com/sgl-project/sglang/pull/25481)

### Open PRs 值得关注

- **#25418** 将 `flash_mla_sparse_fwd` 集成进 SGLang MLA 路径，启用稀疏 forward 注意力计算。[链接](https://github.com/sgl-project/sglang/pull/25418)
- **#25489** 为 tokenspeed MLA 后端的 draft extend 阶段添加 CUDA graph 支持，降低推测解码的调度开销。[链接](https://github.com/sgl-project/sglang/pull/25489)
- **#25463** `[ROCm][MLA]` 移除 ROCm MLA decode 路径中冗余的 tensor contiguous 拷贝（MXFP4 场景）。[链接](https://github.com/sgl-project/sglang/pull/25463)
- **#25460** 为 MLA prefill 增加 `prepare_prefill_qkv` hook，允许在 kernel 执行前注入 fp8 JIT 量化逻辑。[链接](https://github.com/sgl-project/sglang/pull/25460)
- **#25094** `[AMD]` 通过 XGMI 实现单节点内 GPU-to-GPU KV cache 传输，用于 prefill/decode 解耦场景。[链接](https://github.com/sgl-project/sglang/pull/25094)
- **#24640** 为 FlashMLA 推测解码启用 spec v2 路径，支持多 token 预测（MTP）。[链接](https://github.com/sgl-project/sglang/pull/24640)

### Issues 值得关注

- **#25484** `[Bug]` DeepSeek-V4-Pro `paged_mqa_logits_metadata` kernel 在 JIT 编译时超出 GPU shared memory 上限，阻塞 V4-Pro 推理。[链接](https://github.com/sgl-project/sglang/issues/25484)
- **#22819** `[Bug]` 当 `prefix_len == block_size` 时 radix cache 块边界处 KV cache 数据损坏，deterministic 模式下必现。[链接](https://github.com/sgl-project/sglang/issues/22819)

### Commits 直推 main

（API 权限限制，无法访问，本日跳过。）

---

## TensorRT-LLM

### Merged PRs

（今日无确认合并的相关 PR。）

### Open PRs 值得关注

- **#13428** 实现基于 FlashInfer 的 MLA（Multi-head Latent Attention）后端，针对 DeepSeek 系列模型，使用 `BatchPrefillWithRaggedKVCacheWrapper` 和 `BatchMLAPagedAttentionWrapper`。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13428)
- **#14200** 为 Eagle3 推测解码器启用 sliding window attention，补全 speculator 对 SWA 模型的支持。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14200)
- **#14194** 废弃旧版 Triton attention 实现，将 `triton_paged` 后端统一重命名为 `triton`，精简 AutoDeploy 后端列表。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14194)
- **#13721** 通过 CUTLASS 源码和预编译二进制引入 CuTe DSL fmha，支持 ragged attention（16-bit 及 Qk16Pv8 变体）。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13721)
- **#13821** 为 VisualGen 实现 Ring Attention 与统一 Context Parallelism，支持长序列视觉生成。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13821)
- **#13713** 强化 disaggregated KV transfer 的生命周期管理，增加取消和超时处理以提升可靠性。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13713)
- **#14003** 修复 CppMambaHybridCacheManager 的功能和性能问题，涉及 recurrent-state cache 管理。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14003)

### Issues 值得关注

（今日搜索无 TensorRT-LLM 相关新 issue。）

---

## Blogs & 长文

（过去 24 小时内三家团队均无新博客发布。以下为近期值得补读的文章：）

- [HiSparse: Turbocharging Sparse Attention with Hierarchical Memory](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/) — SGLang 将稀疏注意力与分层 KV 卸载结合，256 并发下吞吐超基线 3×（LMSYS，2026-04-10）
- [DeepSeek-V4 on Day 0 with SGLang](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) — SGLang 首日支持 DeepSeek-V4，覆盖 ShadowRadix、HiSparse、Flash Compressor 等注意力相关优化（LMSYS，2026-04-25）

---

## 横向观察

**三框架同步推进 MLA 生态**：vLLM（ROCm sparse-MLA fused concat #42838）、SGLang（flash_mla_sparse_fwd 集成 #25418）、TensorRT-LLM（FlashInfer MLA 后端 #13428）在同一天均有 MLA 相关 PR 活跃，共同指向 DeepSeek 系列模型的 MLA kernel 深度适配。此外，三家都在独立推进 disaggregated KV transfer 的健壮性（vLLM #42839 bug、SGLang #25094 XGMI、TRT-LLM #13713 lifecycle hardening），预示 prefill/decode 解耦部署正在从实验走向生产。
