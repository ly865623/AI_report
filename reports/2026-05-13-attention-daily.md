# Attention Daily — 2026-05-13

## TL;DR

- **vLLM 合并 MooncakeStoreConnector**：通过 Mooncake 分布式存储实现 KV cache offloading，prefill/decode 分离架构再进一步。([#40900](https://github.com/vllm-project/vllm/pull/40900))
- **SGLang 同步 FlashInfer 0.6.11**：三大框架本周均在推进 FlashInfer 0.6.11 升级，SGLang 已率先合并。([#24452](https://github.com/sgl-project/sglang/pull/24452))
- **vLLM 开出 Attention Backend 大重构 RFC**：提案对当前 attention backend 架构进行系统性整理，值得持续跟踪。([#42449](https://github.com/vllm-project/vllm/issues/42449))
- **SGLang SWARadixCache 存在断言失败 bug**：在 DeepSeek-V4 长上下文场景下，滑窗注意力与 RadixCache 组合触发 `dec_lock_ref` 断言，影响生产稳定性。([#24153](https://github.com/sgl-project/sglang/issues/24153))
- **TensorRT-LLM KV cache 基础设施持续强化**：KV block-hash 暴露、多模态 KV 块哈希、disaggregated KV 传输鲁棒性三个 PR 同期在审。([#14005](https://github.com/NVIDIA/TensorRT-LLM/pull/14005), [#13815](https://github.com/NVIDIA/TensorRT-LLM/pull/13815), [#13713](https://github.com/NVIDIA/TensorRT-LLM/pull/13713))

---

## vLLM

### Merged PRs

- **#40900** 新增 MooncakeStoreConnector，通过 Mooncake 分布式 key-value 存储实现 KV cache offloading，为 prefill/decode 异构分离提供新的传输后端。([链接](https://github.com/vllm-project/vllm/pull/40900))

### Open PRs 值得关注

- **#42449** [RFC] Attention Backend 架构重构提案，计划系统性清理当前多 backend 耦合问题，若落地将影响 FlashAttention/FlashInfer/xformers 的集成方式。([链接](https://github.com/vllm-project/vllm/issues/42449))
- **#42112** 修复 TRTLLM ragged MLA prefill 的 workspace warmup 问题：CUDA graph capture 后 MLA prefill 失败的 bugfix。([链接](https://github.com/vllm-project/vllm/pull/42112))
- **#42472** Model Runner V2 接入 FlashInfer sampler，统一新旧 runner 的采样路径。([链接](https://github.com/vllm-project/vllm/pull/42472))
- **#41711** 将 FlashInfer 依赖从 v0.6.8.post1 升至 v0.6.11。([链接](https://github.com/vllm-project/vllm/pull/41711))
- **#40633** 新增基于 Triton 的 INT4/INT2 per-token-head KV cache 量化，进一步压缩显存占用。([链接](https://github.com/vllm-project/vllm/pull/40633))
- **#39841** 修复 chunked prefill 中 FP8 cast 顺序错误，避免 `flashinfer_concat_mla_k` 崩溃。([链接](https://github.com/vllm-project/vllm/pull/39841))
- **#39168** [ROCm] 扩展稀疏 MLA 支持，修复 cache kernel bug 并新增 block size 16 支持。([链接](https://github.com/vllm-project/vllm/pull/39168))
- **#41946** [ROCm][DSV4] 集成 aiter mhc kernel，优化 DeepSeek V4 在 ROCm 上的 attention 计算。([链接](https://github.com/vllm-project/vllm/pull/41946))
- **#38871** [9/n] 将 attention 及 cache kernel 迁移到 torch stable ABI，提升跨版本兼容性。([链接](https://github.com/vllm-project/vllm/pull/38871))
- **#42444** [Model Runner V2][DSV4] 修复 CUDA graph capture 期间 lazy attention state 初始化顺序问题。([链接](https://github.com/vllm-project/vllm/pull/42444))

### Commits 直推 main

（本日无法通过 API 直接枚举外部仓库 commits，上述 merged PR 即为等效记录。）

---

## SGLang

### Merged PRs

- **#24452** FlashInfer 依赖从 0.6.8post1 升至 0.6.11，包含性能提升与 bug 修复。([链接](https://github.com/sgl-project/sglang/pull/24452))
- **#24856** 修复 TRTLLM MHA 在 draft extend 路径下的 routing 错误，涉及 speculative decoding 中的 attention kernel 路由。([链接](https://github.com/sgl-project/sglang/pull/24856))

### Open PRs 值得关注

- **#24925** 集成 tokenspeed_mla prefill/decode kernel，支持 fp8 KV cache 并适配 Blackwell 架构（SM100+）。([链接](https://github.com/sgl-project/sglang/pull/24925))
- **#22921** 为 SM100+（Blackwell）新增 FlashInfer GDN prefill/extend 路径支持。([链接](https://github.com/sgl-project/sglang/pull/22921))
- **#24857** 优化 disaggregated decode 场景下滑窗注意力（SWA）KV pool 预分配策略，减少显存碎片。([链接](https://github.com/sgl-project/sglang/pull/24857))
- **#24976** [CPU] 为 DeepSeek V4 添加 CPU 推理支持，引入 `flash_mla_with_kvcache` CPU attention 实现。([链接](https://github.com/sgl-project/sglang/pull/24976))
- **#25006** 将 trtllm_mha 设为 Gemma4 默认 attention backend。([链接](https://github.com/sgl-project/sglang/pull/25006))
- **#24089** LMCache 多进程模式支持，针对 KV cache 跨进程共享场景。([链接](https://github.com/sgl-project/sglang/pull/24089))

### Open Issues 值得关注

- **#24153** DeepSeek-V4 长上下文场景下 SWARadixCache 触发 `dec_lock_ref` 断言失败，滑窗注意力与 RadixCache 存在竞态，已有多个用户复现。([链接](https://github.com/sgl-project/sglang/issues/24153))
- **#24869** Hicache L2（Unified Radix Tree）在 Gemma 4 高并发下 worker 崩溃，prefix cache 稳定性问题。([链接](https://github.com/sgl-project/sglang/issues/24869))
- **#23500** FlashInfer attention backend 在 Qwen3.5-397B-A17B-FP8 推理中存在不必要的同步开销，有性能回归。([链接](https://github.com/sgl-project/sglang/issues/23500))

### Commits 直推 main

（同上，以 merged PR 记录为准。）

---

## TensorRT-LLM

### Merged PRs

- **#13936** 为 DSv4 disaggregated serving 新增模块级单元测试，覆盖 KV cache 分离场景。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13936))

### Open PRs 值得关注

- **#14005** 将 per-block hash chain 暴露给 KV cache connector，直接读取已存储的哈希值而非重新计算，为 prefix sharing 提供更精确的缓存索引。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14005))
- **#13815** 精确多模态 KV 块哈希：修复 VLM 预处理器中非连续多模态 token span 的哈希计算，prefix cache 命中率有望提升。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13815))
- **#13713** 加固 disaggregated KV 传输的生命周期管理、取消逻辑与静默等待，提升大规模分离式部署的稳定性。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13713))
- **#13929** 新增基于 CuTe DSL 的 FP4 paged MQA logits decode kernel，针对 paged attention 场景的精度与性能优化。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13929))
- **#12525** 在 FlashInfer TRTLLM-gen fmha kernel 中禁用 shared paged index，将 host-side 初始化融入 C++/nanobind op，减少 Python 端开销。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/12525))
- **#13577** 在构造期显式拒绝不兼容的 KV connector 配置，防止静默错误计算。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13577))
- **#14060** 修复 Mamba hybrid 模型在 disaggregated serving 时 KV cache manager 选择逻辑错误。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14060))

### Open Issues 值得关注

- **#13123** `enable_block_reuse=True`（prefix caching）在 Gemma 3 VSWA inflight batching 下崩溃，错误为 `decoderFinishedEvent must be nullopt`。([链接](https://github.com/NVIDIA/TensorRT-LLM/issues/13123))
- **#12877** FP8 context FMHA 被强制开启，导致 1.2.0 推理结果异常（相对 1.1.0 的回归）。([链接](https://github.com/NVIDIA/TensorRT-LLM/issues/12877))

---

## Blogs & 长文

（今日检索周期内未发现 blog.vllm.ai、lmsys.org/blog 或 developer.nvidia.com/blog 有与 attention 直接相关的新文章发布。）

---

## 横向观察

**FlashInfer 0.6.11 成为本周同步焦点**：vLLM（#41711 open）与 SGLang（#24452 merged）同步推进 FlashInfer 0.6.11 升级，TensorRT-LLM 亦在 PR #12525 中深度整合 FlashInfer。三家在同一版本窗口内密集更新，说明 FlashInfer 0.6.11 修复了对各框架均有影响的关键问题。

**KV disaggregation 基础设施全面铺开**：vLLM 合并了 Mooncake KV offloading connector，SGLang 在优化 SWA disaggregated decode 的预分配，TensorRT-LLM 同时推进 KV 传输鲁棒性与 block-hash chain 暴露。prefill/decode 分离架构已从实验性功能走向各框架的核心工程优先级。

**MLA 支持进入收尾阶段**：vLLM 本日同时有 MLA prefill bugfix（#42112）、ROCm 稀疏 MLA 扩展（#39168）在 review；SGLang 正在集成 tokenspeed_mla Blackwell kernel（#24925）并推进 CPU 端 flash_mla（#24976）。MLA 从功能实现转向多硬件覆盖与 edge-case 修复阶段。
