# Attention Daily — 2026-05-11

## TL;DR

- **vLLM MLA 三线并进**：[#41803](https://github.com/vllm-project/vllm/pull/41803) Triton-fused TurboQuant MLA decode、[#41778](https://github.com/vllm-project/vllm/pull/41778) TOKENSPEED_MLA prefill backend、[#41812](https://github.com/vllm-project/vllm/pull/41812) ROCm sparse MLA 三个 PR 同时处于活跃 review 状态，MLA attention backend 生态快速扩张。
- **SGLang 推进 Blackwell MLA decode**：[#24925](https://github.com/sgl-project/sglang/pull/24925) 集成 tokenspeed_mla decode kernels，支持 FP8 KV cache 和 Blackwell 架构，与 vLLM 同名 backend 同步推进。
- **TRT-LLM 合并 KV cache hashing 重构**：[#13800](https://github.com/NVIDIA/TensorRT-LLM/pull/13800) 今日落地，同时新增 [#13929](https://github.com/NVIDIA/TensorRT-LLM/pull/13929) FP4 paged MQA logits decode kernel PR。
- **SWA bug 修复密集**：vLLM [#42273](https://github.com/vllm-project/vllm/issues/42273) 报告 SWA eviction 遗留脏 block table，[#42276](https://github.com/vllm-project/vllm/pull/42276) 针对性修复正在 review；SGLang [#24664](https://github.com/sgl-project/sglang/pull/24664) 为 EAGLE-3 speculative decoding 补全 SWA 支持。
- **KV cache 多级卸载爆发**：vLLM 同时存在对象存储（[#41968](https://github.com/vllm-project/vllm/pull/41968)）、文件系统（[#41735](https://github.com/vllm-project/vllm/pull/41735)）和 PD 分离（[#42285](https://github.com/vllm-project/vllm/pull/42285)）三条 secondary tier 路线同步开发。

---

## vLLM

### Merged PRs

- [#41188](https://github.com/vllm-project/vllm/pull/41188) **MambaAttentionBackendEnum 重构** — 将 mamba_type 字符串字面量统一替换为枚举类型，已于今日合并，为后续 hybrid attention/Mamba 模型的 backend 路由打基础。

### Open PRs 值得关注

- [#41803](https://github.com/vllm-project/vllm/pull/41803) **[Kernel][MLA] Triton-fused TurboQuant decode backend** — 将 MLA decode 和 TurboQuant（INT4 weight-only 量化）融合为单一 Triton kernel，目标是 DeepSeek R1/V3 量化推理的解码吞吐。
- [#41778](https://github.com/vllm-project/vllm/pull/41778) **TOKENSPEED_MLA attention backend（prefill 向）** — 为 DSR1/Kimi K2.5 prefill 阶段新增专用 MLA backend，与 #41803 的 decode 路径形成互补。
- [#41812](https://github.com/vllm-project/vllm/pull/41812) **[ROCm][DSv4] Flash Sparse MLA Triton kernels** — 在 ROCm 平台实现 Native Sparse Attention（NSA）形式的 MLA，补齐 AMD GPU 上 DeepSeek V4 的稀疏注意力能力。
- [#40392](https://github.com/vllm-project/vllm/pull/40392) **[性能][DSR1] MLA Fused RoPE+KVCache+q_concat** — 将 MLA prefill 中的三步操作融合成单 kernel，减少显存带宽开销。
- [#42276](https://github.com/vllm-project/vllm/pull/42276) **SWA eviction 后 block table 重写** — 修复滑动窗口 eviction 时 worker 侧 block table 未同步更新的问题（对应 Issue [#42273](https://github.com/vllm-project/vllm/issues/42273)）。
- [#42285](https://github.com/vllm-project/vllm/pull/42285) / [#41735](https://github.com/vllm-project/vllm/pull/41735) / [#41968](https://github.com/vllm-project/vllm/pull/41968) **KV cache 多级卸载 secondary tier** — 三条并行路线：PD 分离方案、文件系统 Python 层实现、对象存储适配，均活跃讨论中。
- [#40108](https://github.com/vllm-project/vllm/pull/40108) **TurboQuant for YOCO + sliding-window 模型（Gemma 4 E4B）** — 将量化与滑动窗口 attention 结合，支持 Gemma 4 E4B 等模型。
- [#42236](https://github.com/vllm-project/vllm/pull/42236) **[DSv4] 改进 K cache dequant gather kernel** — 优化 DeepSeek V4 KV cache 的反量化读取路径。

### Open Issues 值得关注

- [#42273](https://github.com/vllm-project/vllm/issues/42273) **[Bug] SWA eviction 留下脏 block table 条目** — 滑动窗口 eviction 后 worker 端状态不一致，已有 PR #42276 跟进。
- [#40244](https://github.com/vllm-project/vllm/issues/40244) **[RFC] KV cache free_block_queue 分配顺序 API** — 为长时间运行任务提供控制 KV 块分配顺序的接口，prefix caching 场景相关。
- [#42271](https://github.com/vllm-project/vllm/issues/42271) **[Bug] MTP + piecewise cudagraph 在 HT batched-decode 时死锁** — speculative decoding 和 CUDA graph 结合的稳定性问题。

---

## SGLang

### Merged PRs

- [#24799](https://github.com/sgl-project/sglang/pull/24799) **[AMD] 修复 DeepSeek import cascade** — 兼容 AMD AITER backend 更新前后两版本的 import 路径，解除 ROCm 平台 DeepSeek 模型加载阻塞。
- [#23819](https://github.com/sgl-project/sglang/pull/23819) **[NPU] 修复 MTP warmup 错误** — `--disable-cuda-graph` + MTP（Multi-Token Prediction）组合下 NPU warmup 崩溃问题已修复。
- [#24540](https://github.com/sgl-project/sglang/pull/24540) **[NPU] Wan 模型量化 bugfix** — 修复 NPU 上 Wan 视频生成模型的量化推理错误。

### Open PRs 值得关注

- [#24925](https://github.com/sgl-project/sglang/pull/24925) **集成 tokenspeed_mla decode kernels（FP8 KV + Blackwell）** — 引入面向 Blackwell 架构优化的 MLA decode kernel，同时支持 FP8 KV cache，与 vLLM #41803 方向一致。
- [#23351](https://github.com/sgl-project/sglang/pull/23351) **NSA（Native Sparse Attention）支持 piecewise CUDA graph** — 为稀疏注意力 forward pass 启用分段 CUDA graph，降低 inference 延迟。
- [#24664](https://github.com/sgl-project/sglang/pull/24664) **EAGLE-3 drafter 支持 SWA** — speculative decoding 的 drafter 模型支持滑动窗口 attention，覆盖 Gemma/Qwen 等有 SWA 需求的模型。
- [#24816](https://github.com/sgl-project/sglang/pull/24816) **FlashInfer SM90 cutlass MXFP4 MoE backend** — 为 GPT-OSS + DeepSeek 提供 SM90（Hopper）上的 MXFP4 块量化 MoE kernel，间接影响 attention 之后的 MoE 吞吐。
- [#24943](https://github.com/sgl-project/sglang/pull/24943) **[UnifiedTree] prefix cache 节点 partial match 修复** — RadixAttention 统一树结构中，允许对已 evict 但有备份的节点做部分匹配，提升 prefix cache 命中率。
- [#24794](https://github.com/sgl-project/sglang/pull/24794) **fix(hicache): 存储页 key 加入模型身份 hash** — 修复 HiCache（分层 KV cache）在多模型共存时存储页 key 碰撞的问题。
- [#23254](https://github.com/sgl-project/sglang/pull/23254) **prefix cache 跳过 chunked prefill 时 ring-buffer 泄漏修复** — 分离 prefill 场景下 KV staging buffer 内存泄漏问题。
- [#22536](https://github.com/sgl-project/sglang/pull/22536) **[Disagg][NIXL] 异构 TP KV 传输 staging buffer** — 为不同 TP 大小的 prefill/decode 实例之间的 KV transfer 增加 staging buffer，支持 KV disaggregation。

### Open Issues 值得关注

- [#24869](https://github.com/sgl-project/sglang/issues/24869) **[Bug] UnifiedTree（HiCache L2）高并发下 worker 崩溃** — Gemma 模型高并发时 Radix Tree 状态损坏，与 #24943 有关联。
- [#24938](https://github.com/sgl-project/sglang/issues/24938) **[Feature] DP attention global_tokens 操作融合** — 提议将 DP attention 中的 `global_tokens.fill_(0)` 和 memcpy 合并，减少内核调用开销。

---

## TensorRT-LLM

### Merged PRs

- [#13800](https://github.com/NVIDIA/TensorRT-LLM/pull/13800) **KV cache hashing 容器类型重构** — 将 KV block hash 相关数据结构迁移到新容器类型，为 prefix caching 的 hash 路径提供更清晰的抽象。
- [#13545](https://github.com/NVIDIA/TensorRT-LLM/pull/13545) **DFlash quickstart 文档更新** — 更新 DFlash（Disaggregated Flash Attention）快速入门文档。
- [#13489](https://github.com/NVIDIA/TensorRT-LLM/pull/13489) **Mamba slot sentinel 内存优化** — 将 Mamba 状态槽的哨兵标记从多个合并为一个，降低 hybrid attention/Mamba 模型的显存占用。

### Open PRs 值得关注

- [#13929](https://github.com/NVIDIA/TensorRT-LLM/pull/13929) **[feat] FP4 paged MQA logits decode kernel（cute DSL）** — 使用 NVIDIA cute DSL 实现 FP4 精度的分页 MQA decode kernel，面向 Blackwell FP4 推理。
- [#13937](https://github.com/NVIDIA/TensorRT-LLM/pull/13937) **[refactor] cached prefix 与 KVSlice token_range 解耦** — 将 prefix caching 逻辑从 KVSlice 的 token 范围抽象中分离，内部架构清晰化。
- [#13805](https://github.com/NVIDIA/TensorRT-LLM/pull/13805) **[fix] V2 延迟批处理中 ctx KV 页释放** — 修复 delay batching 模式下 prefill 阶段 KV 页未及时释放导致的内存泄漏。
- [#13965](https://github.com/NVIDIA/TensorRT-LLM/pull/13965) **[feat] DeepSeek V4 scratch buffer 复用** — 通过复用中间 scratch buffer 降低 DeepSeek V4 MLA 计算的峰值显存。
- [#13687](https://github.com/NVIDIA/TensorRT-LLM/pull/13687) **[perf] ltx2 cross-attention 去除冗余 all-gather** — 去掉视频生成模型 ltx2 cross-attention 中不必要的 positional embedding all-gather，降低通信开销。
- [#13773](https://github.com/NVIDIA/TensorRT-LLM/pull/13773) **FlashInfer NVFP4 MoE backend（SM120/SM121）** — 为 Nemotron 等模型在 SM120（Blackwell Pro）上提供 NVFP4 精度的 MoE kernel，与 attention 量化方向配套。
- [#13075](https://github.com/NVIDIA/TensorRT-LLM/pull/13075) **KV transfer 多线程并行化** — 通过多线程加速 KV cache 传输，降低 disaggregated prefill 场景的传输瓶颈。

### Open Issues 值得关注

- [#13560](https://github.com/NVIDIA/TensorRT-LLM/issues/13560) **AutoDeploy B200 上 Llama-3.1 8B 性能 vs vLLM** — 调查 TRT-LLM AutoDeploy backend 在 B200 低并发下落后 vLLM 的原因，涉及 KV cache 配置和 attention batch size 参数。

---

## Blogs & 长文

今日（2026-05-10 至 2026-05-11）未检索到来自 blog.vllm.ai、lmsys.org/blog 或 developer.nvidia.com/blog 的新发布技术长文。

---

## 横向观察

三个框架今日均在以下方向高度趋同：

1. **MLA + 量化 kernel 竞争**：vLLM（#41803 TurboQuant MLA）与 SGLang（#24925 tokenspeed_mla FP8）同日推进面向 Blackwell 的 MLA decode 量化 kernel，名称和目标高度相似，背后可能引用了同一批上游 tokenspeed 研究成果。TRT-LLM 则以 #13929 FP4 MQA kernel 跟进。

2. **SWA（Sliding Window Attention）稳定性**：vLLM、SGLang 同日各有 SWA 相关 PR/Issue，说明 Gemma 4、Qwen3.5 等采用 SWA 的模型在生产部署中暴露出新问题，三个框架的 SWA 实现均处于持续修缮阶段。

3. **KV cache 分层与卸载**：vLLM 的 multi-tier secondary tier（CPU/文件/对象存储）、SGLang 的 HiCache + disaggregated KV transfer、TRT-LLM 的多线程 KV 传输，均在同一时间窗口内活跃，说明长上下文和大规模部署对 KV cache 管理的压力正在三家同步爆发。
