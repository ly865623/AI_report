# Attention Daily — 2026-05-21

## TL;DR

- **TensorRT-LLM 合并 Ring Attention for VisualGen**：WAN 2.2 14B 在多 GB200 上通过 Ring Attention + Unified Context Parallelism 实现分布式注意力计算，视频生成推理再进一步。([#13821](https://github.com/NVIDIA/TensorRT-LLM/pull/13821))
- **SGLang 合并 MLA fp8 prefill hook**：新增 `prepare_prefill_qkv` 扩展点 + fp8 JIT 量化 kernel，MLA 路径的量化 prefill 正式落地。([#25460](https://github.com/sgl-project/sglang/pull/25460))
- **SGLang 同日修复两处 chunked-prefill bug**：scheduler 批次填满问题 (#25741) 与 ring-allocator 槽位别名 (#25862) 均已合并修复，chunked-prefill 在 SWA + 长上下文场景下的稳定性显著提升。
- **vLLM 紧急修复 FlashInfer GDN kernel 回归**：cutlass-dsl 4.5.1 导致 GB200 上 FlashInfer GDN prefill kernel 崩溃，已通过降级至 4.5.0 修复。([#43230](https://github.com/vllm-project/vllm/pull/43230))
- **SWA KV pool 工程问题三框架同日凸显**：SGLang 开出 3 个 SWA KV pool 相关修复/重构 PR，vLLM 修复 MiMo V2.5 SWA 数据错误，TRT-LLM 新增多 head_dim KV pool 支持 Gemma4 混合注意力。

---

## vLLM

### Merged PRs

- **#43230** 将 nvidia-cutlass-dsl 从 4.5.1 降级至 4.5.0，修复 GB200 上 FlashInfer GDN prefill kernel 链接时崩溃问题。([链接](https://github.com/vllm-project/vllm/pull/43230))

### Open PRs 值得关注

- **#43257** 为 Gemma4 global attention 层（`head_dim=512`）调优 Triton `unified_attention` kernel，原实现在 Hopper 上对大 head_dim 有明显性能劣化。([链接](https://github.com/vllm-project/vllm/pull/43257))
- **#43258** 修复 hybrid Mamba+Attention 模型（如 Jamba）的 prefix-cache KV-event 粒度：按 `hash_block_size` 而非物理 attention block size 发送事件。([链接](https://github.com/vllm-project/vllm/pull/43258))
- **#41834** 为 SM12x 消费级 Blackwell 卡适配 DeepSeek V4 Flash：绕过 TMEM/tcgen05 指令限制，FlashMLA 与 FP8 KV cache 路径均做针对性处理。([链接](https://github.com/vllm-project/vllm/pull/41834))
- **#38822** 在 FlashInfer trtllm 后端新增 `head_dim=512` 支持，解除 Blackwell 上大 head_dim 模型（如 Gemma4）的 FlashInfer 路径限制。([链接](https://github.com/vllm-project/vllm/pull/38822))
- **#42270** 修复 MiMo V2.5-Pro fused QKV FP8 权重加载方式差异，同时消除混合 SWA 层的 wrong-data 问题。([链接](https://github.com/vllm-project/vllm/pull/42270))
- **#43241** Gemma4 MTP speculative decoding：草稿层复用目标模型同 attention group 最后一层的 KV cache，减少显存开销。([链接](https://github.com/vllm-project/vllm/pull/43241))

### Open Issues 值得关注

- **#43263** MLA prefill 路径 (`mla_attention.py L2094`) 在 AWQ 量化 + 长序列场景下报 AttributeError，推测是 #34695 后引入的回归。([链接](https://github.com/vllm-project/vllm/issues/43263))
- **#43235** [RFC] NSA（Native Sparse Attention）在消费级 Blackwell（SM12x）上存在架构不兼容，建议将 MLA 作为该硬件上唯一可行的稀疏 attention 路径。([链接](https://github.com/vllm-project/vllm/issues/43235))
- **#43224** [RFC] 将 MLA/RoPE/KV cache 编译器 fusion 迁移为手动 kernel fusion，与 #42770 架构重构方向配套。([链接](https://github.com/vllm-project/vllm/issues/43224))

### Commits 直推 main

（外部仓库 commit API 本环境不可访问，以上 merged PR 为等效记录。）

---

## SGLang

### Merged PRs

- **#25460** 新增 `prepare_prefill_qkv` hook，允许 attention backend 在 prefill kernel 调用前自定义 Q/K/V 处理（量化、RoPE、打包）；随附 fp8 JIT 量化 kernel，适配 tokenspeed-MLA 路径。([链接](https://github.com/sgl-project/sglang/pull/25460))
- **#25741** 修复 chunked prefill scheduler：当 batch 中已有请求需要 chunking 时，新请求无法入队导致吞吐降低，现已纠正批次填充逻辑。([链接](https://github.com/sgl-project/sglang/pull/25741))
- **#25862** 修复 future token map 中的 ring-allocator 槽位重用 bug（wrap-around 别名、跨流竞争），改为以 request-pool index 为主键。([链接](https://github.com/sgl-project/sglang/pull/25862))
- **#25892** 修复 DSV4/FlashMLA 在 `--load-format dummy` 下 `HashTopK.tid2eid` 未初始化导致的越界问题。([链接](https://github.com/sgl-project/sglang/pull/25892))

### Open PRs 值得关注

- **#25418** 将 MLA decode 从 `flash_mla_with_kvcache` 切换至 `flash_mla_sparse_fwd`，实测 ~1.35× decode 加速，并修复 8192 token 以上 chunked-prefill 失败问题。([链接](https://github.com/sgl-project/sglang/pull/25418))
- **#24737** 新增 FlashInfer Cute-DSL MLA 后端（`cutedsl_mla`），在 SM103（Blackwell）上相比现有 MLA 后端提速约 18%。([链接](https://github.com/sgl-project/sglang/pull/24737))
- **#23292** [1/N] 为 MLA prefill 添加 Context Parallel 支持，覆盖 DeepSeek-V3/R1 等 MLA 模型的多 GPU 长上下文 prefill。([链接](https://github.com/sgl-project/sglang/pull/23292))
- **#25824** 将 SWA token 索引转换封装进 `SWAKVPool`，移除外部 `set_swa_loc` side-channel，引入带 per-batch 缓存失效的内部翻译路径。([链接](https://github.com/sgl-project/sglang/pull/25824))
- **#25389** 修复 SWA token mapping 的陈旧反向所有权问题：防止过期的 full-token 映射释放已被新请求复用的 SWA 槽位。([链接](https://github.com/sgl-project/sglang/pull/25385))
- **#25907** 修复 FlashInfer `FlashinferDispatcher` 中 `max_num_tokens` 被 ep_size 重复乘一次的问题，默认上限同步从 1024 提升至 16384。([链接](https://github.com/sgl-project/sglang/pull/25907))

### Open Issues 值得关注

- **#25914** Gemma 4 E2B merged 模型在 FlashInfer paged prefill 路径下报 "invalid configuration: causal=True with non-uniform sequence lengths"，prefill attention kernel 拒绝该 mask 配置。([链接](https://github.com/sgl-project/sglang/issues/25914))
- **#24523** PD 分离部署下，SWA 上限被错误应用到 prefill 节点，导致输入长度硬性限制在 18432 token，实为调度器中 `is_hybrid_swa` 判断逻辑错误。([链接](https://github.com/sgl-project/sglang/issues/24523))

### Commits 直推 main

（外部仓库 commit API 本环境不可访问，以上 merged PR 为等效记录。）

---

## TensorRT-LLM

### Merged PRs

- **#13821** 为 WAN 2.2 14B VisualGen 管线集成 Ring Attention 与 Unified Context Parallelism，支持多 GB200 节点分布式视频生成推理。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13821))

### Open PRs 值得关注

- **#12947** 将 SkipSoftmax（BLASST）稀疏 attention kernel 接入 VisualGen 管线；引入 `BaseSparseAttentionConfig` 可扩展配置体系，支持未来更多稀疏算法。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/12947))
- **#13745** 重构 KVCacheManager 支持多 `head_dim` pool，允许 Gemma4 MoE 中 SWA 层（`head_dim=256`）与全局 attention 层（`head_dim=512`）共用同一 manager。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13745))
- **#13711** 在 AutoDeploy 的 MTP extend 路径集成 FlashInfer kernel，提升多 token 推测解码的 attention 计算效率。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13711))
- **#14060** 为 Mamba hybrid 模型接入 disaggregated serving KV cache 传输 + block reuse（prefix caching），打通混合架构的 PD 分离。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14060))
- **#14297** 修复 DSv4 indexer 在 chunked-prefill 高并发下 CUDA Graph 捕获/回放时的 scratch buffer 陈旧问题。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14297))
- **#14275** 从 PyTorch attention 接口移除已废弃的 `sink_token_length` 参数，清理 KvCacheConfig API。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14275))

### Commits 直推 main

（外部仓库 commit API 本环境不可访问，以上 merged PR 为等效记录。）

---

## Blogs & 长文

- **[HiSparse: Turbocharging Sparse Attention with Hierarchical Memory](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)** — SGLang 团队介绍分层稀疏 attention（CPU 扩展 KV + PD 分离部署），为长上下文稀疏推理提供系统级方案。（发布于 2026-04-10，近期关注度上升）
- **[DeepSeek-V4 on Day 0 with SGLang](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)** — SGLang 首日支持 DeepSeek-V4，集成 FlashMLA 接口、ShadowRadix prefix cache 与 HiSparse CPU-extended KV，是本周多个 MLA PR 的背景文档。（发布于 2026-04-25）
- **[The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)** — vLLM 团队系统梳理 FP8 KV cache 量化现状，包含 FA3/FA4 prefill 支持与 per-token 动态量化路线图。（发布于 2026-04-22）
- **[Accelerating Long-Context Inference with Skip Softmax in TensorRT-LLM](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/)** — NVIDIA 介绍 SkipSoftmax 稀疏 attention，在 Hopper/Blackwell 上对长序列实现最高 1.4× 加速，与 PR #12947 直接相关。

---

## 横向观察

**SWA KV pool 工程债务在三框架同步爆发**：随着 Gemma4、DeepSeek-V4、MiMo V2.5 等混合 SWA/全局注意力模型普及，三框架在同一天密集处理 SWA 相关问题——SGLang 对 SWA KV pool 进行架构级重构（#25824）并修复两处正确性 bug（#25385/#25889），vLLM 修复 MiMo V2.5 SWA 数据错误（#42270），TRT-LLM 重构 KVCacheManager 以支持同一模型中的多 head_dim SWA（#13745）。SWA 正从"特殊 case"演变为各框架 KV 管理的核心能力。
