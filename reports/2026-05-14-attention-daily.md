# Attention Daily — 2026-05-14

## TL;DR

- **SGLang 合并 TokenSpeed MLA 内核（Blackwell + FP8 KV）**：PR #24925 将 TokenSpeed MLA 的 prefill/decode 内核（支持 FP8 KV cache、SM100 Blackwell）正式并入主线，是本日最重大的 Attention 后端进展。[链接](https://github.com/sgl-project/sglang/pull/24925)
- **vLLM 新增 TOKENSPEED_MLA 后端 PR**：PR #41778 为 vLLM 引入同系列 CuTe DSL MLA 内核，覆盖 Blackwell 上的 DeepSeek-R1 和 Kimi K2.5，两框架在 Blackwell MLA 上共享技术路线。[链接](https://github.com/vllm-project/vllm/pull/41778)
- **vLLM RFC：统一 KV Cache 布局**：Issue #42082 提议标准化 `[num_layers, num_blocks, num_states, num_heads, ...]` 布局，消除 MLA/GQA/Mamba 各后端分歧，具有重要架构意义。[链接](https://github.com/vllm-project/vllm/issues/42082)
- **TensorRT-LLM 新增 FlashInfer MLA 后端**：PR #13428 正在为 DeepSeek 系列模型接入基于 FlashInfer 的 MLA 后端，同时 PR #13929 引入 FP4 paged MQA decode 内核（CuTe DSL）。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13428)
- **三框架同日推进 DSV4 SWA prefix cache 修复**：vLLM PR #42258/#42296 优化 DeepSeek-V4 滑动窗口注意力 prefix cache 命中率；SGLang PR #25088 修复 HiCache 的 SWA 子树回溯节点 bug，均于今日更新。

---

## vLLM

### Merged PRs
（今日检索范围内未发现已合并的 attention/KV cache 相关 PR）

### Open PRs 值得关注

- **#41778** `[MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell` — 将 TokenSpeed MLA CuTe DSL 内核接入 vLLM，作为新的 MLA 后端，覆盖 Blackwell（SM100）上的 prefill 和 decode 阶段，目标模型为 DeepSeek-R1 和 Kimi K2.5。[链接](https://github.com/vllm-project/vllm/pull/41778)

- **#42258** `[Core][DSV4] Skip caching SWA blocks that can never serve a prefix-cache hit` — 针对 DeepSeek-V4 混合全注意力（block_size=256）+ SWA（block_size=64）架构，通过几何分析剔除永远不会产生 prefix cache 命中的 SWA block，减少无效缓存条目。[链接](https://github.com/vllm-project/vllm/pull/42258)

- **#42296** `[Feat][KVConnector] Support DSV4 in SimpleCPUOffloadBackend` — 依赖 #42258，扩展 SimpleCPUOffloadBackend KV Connector 支持 DeepSeek-V4，修复 scheduler/hash block size 解析逻辑。[链接](https://github.com/vllm-project/vllm/pull/42296)

- **#41968** `Add objectstore as a secondary tier to multi-tier kv cache offloading` — 通过 NVIDIA NIXL 将 S3 兼容对象存储作为多层 KV cache 卸载的第二级后端，实现 KV cache 的持久化和超本地内存容量存储。[链接](https://github.com/vllm-project/vllm/pull/41968)

- **#42580** `[Bugfix] Fix fp8 kv cache scaling for the triton attention backend` — 修复 Triton 注意力后端在 FlashInfer + FP8 KV cache + CUDA graphs 路径下 scaling factor 应用错误导致输出损坏的 bug（Blackwell 上复现）。[链接](https://github.com/vllm-project/vllm/pull/42580)

- **#39168** `[ROCm] Expanded sparse MLA support` — 修复 `cache_kernels.cu` 中 block_stride 非 16 倍数时的 bug，为 ROCm sparse MLA 增加 block size 16 支持。[链接](https://github.com/vllm-project/vllm/pull/39168)

- **#41834** `Add SM12x support for DeepSeek V4 Flash` — 为 RTX PRO 6000 / DGX Spark GB10（SM12x，缺少 TMEM/tcgen05 指令）提供纯 PyTorch 和 Triton 的 FlashMLA 回退路径。[链接](https://github.com/vllm-project/vllm/pull/41834)

### Open Issues 值得关注

- **#42082** `[RFC]: Standardize KV-cache Layouts` — 提议统一逻辑 KV cache 布局为 `[num_layers, num_blocks, num_states, num_heads, <state_content>]`，解耦 KV-connector 与后端实现，覆盖 MLA、GQA、Mamba 等异构场景。[链接](https://github.com/vllm-project/vllm/issues/42082)

- **#41962** `[ROCm] DeepSeek-V4-Flash: rocm_dequantize_blocked_k_cache 在 decode 时对整个 KV cache pool 做反量化导致 OOM` — SWA decode fallback 路径将全部 FP8 KV cache 展开为 bfloat16，每次 decode 消耗 13+ GiB HBM，PR #42248/#42576 正在修复中。[链接](https://github.com/vllm-project/vllm/issues/41962)

- **#42571** `[Bug]: KV Block double free when using eager SimpleCPUOffloading + Sliding window attention` — 在 CPU offloading eager 模式与 SWA 共存时触发 KV block 双重释放崩溃。[链接](https://github.com/vllm-project/vllm/issues/42571)

- **#40756** `MTP speculative decoding crash with prefix caching + chunked prefill on long sequences` — Qwen3.6-27B-FP8 在启用 prefix caching 和 chunked prefill 时，约 26k token 后触发 MTP 投机解码非法内存访问崩溃。[链接](https://github.com/vllm-project/vllm/issues/40756)

### Commits 直推 main
（GitHub MCP 工具权限限制，无法直接枚举 commits，此项略过）

---

## SGLang

### Merged PRs

- **#24925** `[attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)` — 将 TokenSpeed MLA prefill 和 decode 内核（FP8 KV cache、SM100 Blackwell）接入 SGLang 注意力后端，于 2026-05-14 00:36 合并。[链接](https://github.com/sgl-project/sglang/pull/24925)

- **#25001** `[LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support` — 为 DeepSeek 风格 MLA 的低秩投影矩阵（`q_b_proj`、`kv_b_proj`）增加 LoRA adapter 支持，于 2026-05-13 22:15 合并。[链接](https://github.com/sgl-project/sglang/pull/25001)

### Open PRs 值得关注

- **#25195** `Support breakable CUDA graph for DeepSeek V4 DP attention` — 让 DeepSeek-V4 数据并行注意力（DP attention）在 mixed/extend batch 场景下支持 piecewise 可中断 CUDA graph，解决之前被迫走 eager 路径的性能回退。[链接](https://github.com/sgl-project/sglang/pull/25195)

- **#25088** `[UnifiedRadixCache] Fix HiCache load back start node` — 修复 UnifiedRadixCache 中 HiCache L2 回写时锚点节点选择错误的 bug，统一 SWA/全注意力/Mamba 子树的 `best_match_node` 逻辑。[链接](https://github.com/sgl-project/sglang/pull/25088)

- **#24953** `[UnifiedTree] feat: add HiCache storage (L3) support` — 为 SGLang HiCache 层级 KV cache 增加 L3（磁盘/远程存储）层，构建 L1（GPU）→ L2（CPU）→ L3（远端）三级 KV 存储体系。[链接](https://github.com/sgl-project/sglang/pull/24953)

- **#24541** `[AMD] feat(attention): pick native vs padded MLA decode heads from num_head` — 在 AMD ROCm 上根据 `num_head` 正确选取 MLA decode 内核变体（native vs padded），修复 MLA 注意力在 ROCm 上的 dispatch 正确性问题。[链接](https://github.com/sgl-project/sglang/pull/24541)

### Open Issues 值得关注

- **#24656** `[RFC] Agent-Aware KV Cache Phase 1` — 提议将 agentic 工作流元数据（角色、工具调用轮次）传入运行时，影响 RadixAttention 的 LRU/LFU/SLRU 淘汰策略，以提升 agent 负载下的缓存命中率。[链接](https://github.com/sgl-project/sglang/issues/24656)

- **#24153** `AssertionError in SWARadixCache.dec_lock_ref under long-context DSV4` — SWARadixCache 在 DSV4-Pro 的 100K–500K token 长上下文共享前缀负载下触发引用计数 bug 崩溃。[链接](https://github.com/sgl-project/sglang/issues/24153)

- **#25118** `DSV4-Flash on MI300x fails with assert store_dtype == uint8` — AMD MI300x 上 DeepSeek-V4-Flash 的 MLA KV cache 存储 dtype 断言失败。[链接](https://github.com/sgl-project/sglang/issues/25118)

- **#23500** `FlashInfer prefill plan() blocking sync causes latency regression in EAGLE` — FlashInfer prefill `plan()` 在 EAGLE 投机解码每个 decode step 时触发阻塞 `cudaMemcpyAsync`，严重影响延迟。[链接](https://github.com/sgl-project/sglang/issues/23500)

### Commits 直推 main
（权限限制，此项略过）

---

## TensorRT-LLM

### Merged PRs

- **#14083** `move 3 python_scheduler chunked_prefill cases to post merge` — CI 维护性改动，将三个 BF16 chunked-prefill 单卡测试用例移至 post-merge gate 以缩减 pre-merge 时间，无功能变更。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14083)

### Open PRs 值得关注

- **#13428** `Add FlashInfer MLA attention backend support` — 为 DeepSeek 系列模型实现基于 FlashInfer 的 MLA 后端，prefill 用 `BatchPrefillWithRaggedKVCacheWrapper`，generation 用 `BatchMLAPagedAttentionWrapper`（paged KV）。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13428)

- **#13929** `Add cute dsl FP4 paged MQA logits decode kernel` — 使用 CuTe DSL 实现 FP4 paged Multi-Query Attention decode 内核，针对 DeepSeek-V4 的 decode 性能优化。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13929)

- **#14029** `DSV4 multistream improvement for attention` — DeepSeek-V4 注意力路径的多流执行性能改进。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14029)

- **#13745** `AutoDeploy: Support Gemma4 mixed-shape pools in KVCacheManager` — 重构 AutoDeploy KV cache 层，将 Gemma-4 的异构 KV pool（SWA 层 head_dim=256，全注意力层 head_dim=512）统一纳入单个 C++ `KVCacheManager`，修复 radix reuse tree、disagg KV transfer 等跨 pool 协作问题。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13745)

- **#14104** `SM120/121-conditional post-processing override for enable_attention_dp` — 修复 Qwen3-235B-A22B-FP8 在 RTX PRO 6000 Blackwell（SM120/121）上启用 attention DP 时的死锁，对该 SM 版本条件性禁用 attention DP，H100/H200/B200 不受影响。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14104)

- **#13815** `Exact multimodal KV blockhashing` — 修复多模态 KV cache prefix reuse 的 block hash 计算，支持真实 VLM 预处理器生成的非连续多 span 多模态 item，替代此前的单连续 span 假设。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13815)

- **#12525** `Disable shared paged index in flashinfer trtllm-gen fmha kernel and unify kv cache buffer calculation` — 将 FMHA 注意力的 shared paged index 和 KV cache buffer 大小计算移入 C++/nanobind，减少 Python host-side 开销。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/12525)

### Open Issues 值得关注

（今日 TensorRT-LLM 的 attention/KV cache 相关 issue 活跃度较低，无新增高优先级问题）

---

## Blogs & 长文

- **[SGLang HiSparse: Turbocharging Sparse Attention with Hierarchical Memory](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)** — SGLang 团队于 4 月 10 日发布 HiSparse 博文，介绍利用 CPU 内存分层存储实现稀疏注意力（NSA/DSA）decode 的系统设计，大幅降低稀疏注意力在长上下文推理中的显存压力。（近期相关博文，非今日发布）

- **[DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)** — SGLang 团队于 4 月 25 日发布 DSV4 day-0 支持博文，覆盖 DP attention、MLA 内核优化及 PD 分离部署细节。（近期相关博文，非今日发布）

今日（2026-05-14）三框架官方博客均无新发布文章。

---

## 横向观察

**三框架今日同步推进 Blackwell MLA 内核落地**：SGLang 已合并 TokenSpeed MLA 内核（#24925），vLLM 的同系列 PR（#41778）正在审查中，TensorRT-LLM 则以 FlashInfer MLA 后端（#13428）和 FP4 MQA 内核（#13929）为主线推进。技术上均以 FP8 KV cache + SM100 为目标，体现出三方在 Blackwell 世代 MLA 优化上的高度同向性，TokenSpeed CuTe DSL 内核已成为事实上的跨框架技术汇聚点。

**DSV4 混合注意力（SWA + 全注意力）的 KV cache 管理是当日主要 bug 集中区**：vLLM 的 #42258、#42571、#41962，SGLang 的 #24153、#25088、#25148，TensorRT-LLM 的 #13745 均与 DeepSeek-V4 的混合注意力 block 管理相关，说明这一架构的 prefix caching 和内存管理尚处于稳定化阶段。
