# Attention Daily — 2026-05-16

## TL;DR

- **SGLang 合并 MLA chunked-prefill Triton 融合算子**：一个 Triton kernel 同时完成 `k_nope/k_pe` 拼接与 FP8 量化，减少显存来回读写，是本日最具实质性能影响的 merge。([#25333](https://github.com/sgl-project/sglang/pull/25333))
- **SGLang 修复 TRTLLM MHA + SWA + 推测解码三方交互 bug**：draft 模型错误继承 target 的 SWA KV pool 标志导致 KV cache 索引错位，已合并修复。([#25103](https://github.com/sgl-project/sglang/pull/25103))
- **vLLM 合并 kv-cache-dtype 覆盖 bug 修复**：compressed-tensors checkpoint 的 `kv_cache_scheme` 会无条件将 KV dtype 强制为 FP8，覆盖用户显式指定的 `--kv-cache-dtype` flag，已修复。([#42782](https://github.com/vllm-project/vllm/pull/42782))
- **vLLM 启动 KV cache 布局统一大重构**：`[Core] Standardize kv layout` 系列 PR（1/N）正式开始，目标是统一 FlexAttention/FlashAttention 等所有 backend 的 KV 张量布局。([#42374](https://github.com/vllm-project/vllm/pull/42374), [#42095](https://github.com/vllm-project/vllm/pull/42095))
- **TensorRT-LLM 推进 Ring Attention 与 Gemma4 chunked-prefill**：面向视觉生成模型（WAN 2.2 14B）的 Ring Attention PR 和支持 Gemma4 全系列 chunked-prefill 的 PR 均在积极 review 中。([#13821](https://github.com/NVIDIA/TensorRT-LLM/pull/13821), [#14134](https://github.com/NVIDIA/TensorRT-LLM/pull/14134))

---

## vLLM

### Merged PRs

- **#42782** `[Bugfix] Respect explicit --kv-cache-dtype over checkpoint kv_cache_scheme` — 修复 per-layer Attention 初始化时无条件读取 compressed-tensors checkpoint 的 `kv_cache_scheme` 并强制写入 FP8 的问题，用户通过 CLI 传入的 `--kv-cache-dtype` 参数被静默忽略，同时导致 draft 模型推测解码在 Blackwell 上异常，已于 2026-05-16 合并。([链接](https://github.com/vllm-project/vllm/pull/42782))

### Open PRs 值得关注

- **#42784** `[Bugfix] Fix SWA cache block mask breaking prefix caching with Eagle/MTP` — `SlidingWindowManager._cache_block_mask()` 在 Eagle/MTP 推测解码激活时过度截断，导致 prefix cache 命中率归零，正在修复中。([链接](https://github.com/vllm-project/vllm/pull/42784))
- **#42374** `[Core][WIP][1/N] Standardize kv layout` — KV cache 布局统一化系列第一 PR，目标是在所有 attention backend 间对齐张量格式，后续改动影响范围很广。([链接](https://github.com/vllm-project/vllm/pull/42374))
- **#42095** `[Attention] Make FlexAttention and FlashAttention use num-blocks first layouts` — 与 #42374 配套，将 FlexAttention 和 FlashAttention 的 KV block 张量改为 num-blocks-first，修复布局不一致 bug（tracking issue #41657）。([链接](https://github.com/vllm-project/vllm/pull/42095))
- **#42689** `[KV Connector] Support disk offloading in MooncakeStoreConnector` — 在 MooncakeStoreConnector 中新增 SSD 磁盘 KV 溢出路径，经 4×GB200 + Qwen3-8B 验证可达 49.6 GB 磁盘读取，进一步延伸 KV offloading 层级。([链接](https://github.com/vllm-project/vllm/pull/42689))
- **#41834** `[New Model] Add SM12x support for DeepSeek V4 Flash` — 为 RTX PRO 6000 / DGX Spark GB10（SM12x Blackwell 消费级）补充 FlashMLA 和 DeepGEMM 的 Triton/PyTorch fallback，包含 MLA prefix-cache reuse 的 KV 管理器修复。([链接](https://github.com/vllm-project/vllm/pull/41834))
- **#31636** `[Frontend] Add FP8 output quantization support to FlashAttention backend` — 在 FlashAttention-3 kernel 路径内融合 FP8 静态量化输出，通过 `output_scale` 直接写出 FP8 激活，减少后续算子的显存带宽消耗。([链接](https://github.com/vllm-project/vllm/pull/31636))

### Issues 值得关注

- **#41515** `[Bug]: [kv_offload+HMA] Fails on chat subsequent request` — KV offload 与 Hybrid Model Architecture 共存时，多轮对话的第二条请求报错，仍在追踪中。([链接](https://github.com/vllm-project/vllm/issues/41515))
- **#38652** ~~`[Bug]: --kv-cache-dtype fp8 produces garbage output on MLA models`~~ — 已关闭，与 #42782 合并修复相关。([链接](https://github.com/vllm-project/vllm/issues/38652))

---

## SGLang

### Merged PRs

- **#25333** `perf(mla): hybrid Triton fused cat+FP8-quantize for MLA chunked-prefill K/V` — **本日亮点**。将 `k_nope/k_pe` 的拼接（concat）与 K、V 的 FP8 量化合并为单一 Triton kernel，消除中间结果的显存 round-trip 和多余 kernel launch，对 MLA 长序列 chunked-prefill 吞吐有直接提升，已于 2026-05-15 合并。([链接](https://github.com/sgl-project/sglang/pull/25333))
- **#25277** `[UnifiedTree]: Fix UnifiedRadixCache device match semantics with HiCache` — 修正 `UnifiedRadixCache` 中 `best_match_node` 与 device-local 命中的混淆逻辑，使 HiCache（host-memory prefix cache）的命中与 GPU 常驻命中正确区分，已于 2026-05-15 合并。([链接](https://github.com/sgl-project/sglang/pull/25277))
- **#25103** `[TRTLLM/SWA/Spec] fix trtllm mha + swa + spec accept length drop` — draft 模型错误继承 target 模型的 SWA KV pool 标志，导致其对 full-layer-only pool 应用 SWA 索引重映射，读取到错误的 KV cache 位置，推测解码接受率下降，已于 2026-05-16 合并。([链接](https://github.com/sgl-project/sglang/pull/25103))

### Open PRs 值得关注

- **#25418** `integrate flash_mla_sparse_fwd` — 将 `flash_mla_sparse_fwd` 稀疏注意力 kernel 接入 SGLang 的 MLA attention backend，为 sparse MLA 推理铺路。([链接](https://github.com/sgl-project/sglang/pull/25418))
- **#24640** `Support spec v2 for FlashMLA speculative decoding` — 在 FlashMLA 多 token 预测路径中启用推测解码 v2，并补充 CUDA CI 测试覆盖。([链接](https://github.com/sgl-project/sglang/pull/24640))
- **#24101** `Enable num_splits > 1 for FA3 deterministic inference` — 解除 FA3 确定性推理中的 `num_splits=1` 限制（依赖上游 sgl-flash-attn 中的 `batch_invariant` 调度修复），将 `num_splits` 提升至 4，prefill 吞吐最高可恢复 2.1×。([链接](https://github.com/sgl-project/sglang/pull/24101))
- **#25090** `[AMD] Support triton backend decode context parallel for Qwen3.5 [WIP]` — 在 AMD 上通过 Triton attention kernel 为 Qwen3.5 支持 decode context parallelism（DCP），采用 zigzag KV sharding，仍在开发中。([链接](https://github.com/sgl-project/sglang/pull/25090))

### Issues 值得关注

- **#25315** `[Bug] DeepSeek-R1 increased kv usage and performance regression due to transformers==5.6.0` — transformers 升级到 5.6.0 后 KV cache 用量异常上升、性能下降，可能影响生产环境升级决策。([链接](https://github.com/sgl-project/sglang/issues/25315))
- **#25388** `[Bug] JSON grammar abort with EAGLE/NEXTN + SPEC_V2 overlap in PD disaggregated decode` — PD disaggregated decode 中 EAGLE/NEXTN 与 SPEC_V2 重叠触发 JSON grammar abort，影响结构化输出场景。([链接](https://github.com/sgl-project/sglang/issues/25388))

---

## TensorRT-LLM

（今日无新合并 PR，以下均为进行中的重点 PR）

### Open PRs 值得关注

- **#13821** `[feat] Ring Attention, Unified Context Parallel for VisualGen` — 为视觉生成模型（WAN 2.2 14B）引入 Ring Attention 与统一 context-parallel 策略，attention 计算可跨多卡近线性扩展。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13821))
- **#14134** `[feat] Add chunked prefill support for Gemma4 (text + vision)` — 为 Gemma4 全系列（E2B、E4B、26B-A4B MoE、31B）支持 chunked prefill，SWA + chunked prefill 路径以 Triton 作为 fallback。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14134))
- **#14200** `[feat] Enable sliding window attention for Eagle3` — 在 Eagle3 推测解码中支持可配置 SWA，补充 SWA 配置透传的测试覆盖。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14200))
- **#12928** `[feat] KV cache manager v2 + python transceiver bug fix` — 重新设计的 KV cache manager（v2），引入 VMM-aware 内存描述符分割与 region 元数据，同步修复 disaggregated KV 传输中的 Python transceiver 错误。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/12928))
- **#13575** `[feat] Integrate FP4 indexer for DSv4` — 将 DeepGEMM 的 FP4 精度 indexer kernel 接入 DeepSeek V4 稀疏注意力，降低 KV index 计算精度以节省带宽。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13575))
- **#14194** `[feat] Autodeploy deprecate the legacy triton attention` — 将 AutoDeploy 中遗留的 `triton_paged` attention backend 标记为废弃，统一为新版 `triton` 后端，Gemma4 配置同步更新。([链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14194))

### Issues 值得关注

- **#10611** ~~`AutoDeploy Revisit the Flashinfer KV cache for FI 0.6.0`~~ — 已关闭（2026-05-15）。FlashInfer 升级后 AutoDeploy FP8 测试在 Hopper 上失败（`No eligible GMMA operator for request configuration`），FP8 KV cache 临时回退 BF16，正式解法已在 PR #14194 中推进。([链接](https://github.com/NVIDIA/TensorRT-LLM/issues/10611))

---

## Blogs & 长文

- **[A First Comprehensive Study of TurboQuant: Accuracy and Performance](https://vllm.ai/blog/2026-05-11-turboquant)** (vLLM Blog, 2026-05-11) — 系统评测极低比特（INT2/INT3/INT4）KV cache 量化方案，结论是 FP8 仍是精度与效率最优权衡，超低比特方案在长上下文场景下精度损失不可忽略。
- **[DeepSeek V4 in vLLM: Efficient Long-context Attention](https://vllm-project.github.io/2026/04/24/deepseek-v4.html)** (vLLM Blog, 2026-04-24) — 详解 DeepSeek V4 混合稀疏注意力在 vLLM 中的实现，覆盖 SWA + sparse top-k + prefix cache 协同。
- **[HiSparse: Turbocharging Sparse Attention with Hierarchical Memory](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)** (LMSYS Blog, 2026-04-10) — SGLang 团队提出 HiSparse，主动将不活跃 KV cache 分层卸载至 CPU DRAM，GPU HBM 仅保留热点 KV，在长上下文稀疏注意力场景下实现显存容量与吞吐的双重提升。
- **[DeepSeek-V4 on Day 0 with SGLang](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)** (LMSYS Blog, 2026-04-25) — 介绍 SGLang 首日支持 DeepSeek-V4 的技术细节，包括为混合注意力（SWA + extra attention over compressed KV）设计的 ShadowRadix prefix cache 机制。

---

## 横向观察

三个框架今日均有 **MLA / SWA / sparse attention** 方向的集中动作：SGLang 合并了 MLA chunked-prefill 的 Triton 融合算子并开始集成 `flash_mla_sparse_fwd`；vLLM 正在为 DeepSeek V4 Flash 的 FlashMLA 补充 SM12x 消费级 Blackwell 支持；TensorRT-LLM 则在推进 DSv4 的 FP4 sparse attention indexer。三者的分工逐渐清晰：SGLang 在算子融合和 kernel 集成上落地最快，vLLM 在做更底层的 KV 布局标准化（`Standardize kv layout` 系列），TensorRT-LLM 则在硬件（Hopper/Blackwell）与量化精度（FP4/FP8）的深度整合上持续发力。
