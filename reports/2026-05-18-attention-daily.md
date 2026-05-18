# Attention Daily — 2026-05-18

## TL;DR

- **vLLM 合并了 DeepSeek V4 MLA 在 AMD MI300X 的重要修复**（#42810），解决了高并发下 MLA 注意力输出错误和精度退化问题，ROCm 用户可直接受益。[链接](https://github.com/vllm-project/vllm/pull/42810)
- **SGLang 集成 `flash_mla_sparse_fwd` 核**（#25418），替换旧的 `flash_mla_with_kvcache`，MLA prefill 吞吐提升约 1.35×，同时修复超大 chunked-prefill 下的崩溃路由。[链接](https://github.com/sgl-project/sglang/pull/25418)
- **SGLang 将 `trtllm_mha` 设为 Gemma 4 默认 attention 后端**（#25006 已合并），SM100/Blackwell 上 Gemma 4 推理性能显著提升。[链接](https://github.com/sgl-project/sglang/pull/25006)
- **TensorRT-LLM 新增 FlashInfer MLA attention 后端**（#13428，open），为 DeepSeek 系列模型带来零拷贝 paged KV cache 视图与外部 RoPE 处理，是继自研 MLA 核后的第二条推理路径。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13428)
- **vLLM 修复消费级 Blackwell（RTX 5090/PRO 6000）FlashMLA OOM**（#42856），将 sparse-MLA workspace 从按 H200 140 GiB 定大小改为按实际显存调整。[链接](https://github.com/vllm-project/vllm/pull/42856)

---

## vLLM

### Merged PRs

- **#42810** [ROCm] 修复 DeepSeek V4 在 MI300X 的功能与精度问题 — 还原了 MLA 注意力输出中的正确性回归，修复高并发场景下的精度退化，ROCm/gfx942 用户可直接更新。[链接](https://github.com/vllm-project/vllm/pull/42810)

### Open PRs 值得关注

- **#42893** [ROCm][DSv4] MI300X 上 DeepSeek V4 的后续修复 — 补丁 #42810 遗留的 eager 和 CUDA graph 模式加载失败，修正 FP8 FNUZ/OCP 类型不匹配及 KV workspace 初始化 bug。[链接](https://github.com/vllm-project/vllm/pull/42893)
- **#41834** 为消费级 Blackwell（SM12x）添加 DeepSeek V4 Flash 支持 — RTX 5090 / GB10 缺少 datacenter TMEM 指令，此 PR 用 Triton sparse-MLA 和 FlashMLA 回退路径实现 MLA 推理。[链接](https://github.com/vllm-project/vllm/pull/41834)
- **#42856** 降低 SM120 上 sparse-MLA + indexer workspace 上限 — 将 FlashMLA workspace 分配从 H200 的 140 GiB 基准调整为消费级 Blackwell 的 95 GiB，解决 RTX 5090 OOM。[链接](https://github.com/vllm-project/vllm/pull/42856)
- **#42828** [KVConnector][DSV4] Mooncake 外部 KV 连接器支持 HMA — 让混合 attention 布局的模型（full + sliding-window，含 MLA）能共享同一 Mooncake KV 池，解决 HMA 未接入时的内存孤岛问题。[链接](https://github.com/vllm-project/vllm/pull/42828)
- **#41968** 多层 KV cache offloading 新增 S3 对象存储层 — 借助 NVIDIA NIXL，将 KV offload 扩展至 S3 兼容对象存储，形成 GPU → CPU → 远端存储三级架构。[链接](https://github.com/vllm-project/vllm/pull/41968)
- **#41847** KV Transfer 将 HMA 默认开启 — 把混合 KV 缓存管理器（HMA）从 opt-in 改为 opt-out，简化 disaggregated prefill 部署配置。[链接](https://github.com/vllm-project/vllm/pull/41847)
- **#42172** [Bugfix][KVConnector] 修复 LMCacheMP 的 prefix cache key 碰撞 — 两个 token 长度相同但 prompt embedding 不同的请求，可能错误共享 LMCache KV 条目；通过将 embedding SHA-256 摘要折入 key 解决。[链接](https://github.com/vllm-project/vllm/pull/42172)
- **#42890** 支持 NVFP4 KV 与 `kv-cache-dtype-skip-layers sliding_window` 共存 — 修复 NVFP4 KV 量化与 sliding window 层跳过选项不兼容的问题，集中化 KV cache 缩放/填充逻辑。[链接](https://github.com/vllm-project/vllm/pull/42890)
- **#39995** [Spec Decode] DFlash + FlashInfer 后端支持 SWA 与 FP8 KV cache — 为 DFlash 投机解码增加 FlashInfer attention 后端，同时支持 sliding-window attention 和 FP8 KV cache（RTX 4090 级 GPU）。[链接](https://github.com/vllm-project/vllm/pull/39995)

### Issues 值得关注

- **#41559** DFlash 投机解码与所有 KV 量化方案（FP8/TurboQuant）根本不兼容 — DFlash draft 模型需要 non-causal attention，但所有 attention 后端在非因果模式下都拒绝量化 KV，被迫使用 BF16 KV cache 导致显存效率减半。[链接](https://github.com/vllm-project/vllm/issues/41559)
- **#41726** TurboQuant KV cache 在大 chunked prefill 续写时崩溃 — 在全局 workspace 已锁定后恢复长提示的 chunked prefill 时，TurboQuant attention 后端触发运行时崩溃（Qwen3.5-9B 复现）。[链接](https://github.com/vllm-project/vllm/issues/41726)
- **#42186** FlashInfer 0.6.8 在 B200（SM100）上引发 worker 无限挂起 — 已二分到 flashinfer-python 0.6.7→0.6.8.post1 之间引入的回归，EP=8 + NVFP4 场景必现。[链接](https://github.com/vllm-project/vllm/issues/42186)

---

## SGLang

### Merged PRs

- **#25006** 将 `trtllm_mha` 设为 Gemma 4 的默认 attention 后端 — 在 SM100/Blackwell 上对比 Triton attention 后端后选定 `trtllm_mha`，Gemma 4 31B 推理吞吐显著提升。[链接](https://github.com/sgl-project/sglang/pull/25006)

### Open PRs 值得关注

- **#25418** 集成 `flash_mla_sparse_fwd` 核（MLA prefill 加速）— 替换 DeepSeek-V4 Flash/Pro 中的 `flash_mla_with_kvcache`，利用 TMA direct load 实现约 1.35× prefill 加速；同时修复超大 chunked-prefill 下的路由 bug。[链接](https://github.com/sgl-project/sglang/pull/25418)
- **#25502** [DSv4] 将大 q 的 prefill 路由至 `flash_mla_sparse_fwd` — 修复 DeepSeek-V4 dense MLA 在超大 `--chunked-prefill-size` 下崩溃的问题，与 #25418 形成配套。[链接](https://github.com/sgl-project/sglang/pull/25502)
- **#23292** [CP] 第 1/N：MLA Prefill 上下文并行支持 — 为 DeepSeek V3/R1、Kimi K2.5 等 MLA 模型实现 prefill 上下文并行，使用 FlashAttention 后端新增 `cp_attn_forward_extend` 封装。[链接](https://github.com/sgl-project/sglang/pull/23292)
- **#25547** 修复 Gemma4 attention 后端用户配置被强制覆盖的 bug — 默认后端选择逻辑无条件覆盖了 `--attention-backend`，导致用户手动指定 Triton 等后端静默失效。[链接](https://github.com/sgl-project/sglang/pull/25547)
- **#25299** [NSA] 避免 NSA MQA logits 内存重复查询 — Native Sparse Attention prefill 路径每次大 chunked-prefill 调用都会触发阻塞式 `mem_get_info`（需 host-device 同步），通过缓存单次查询结果消除该瓶颈。[链接](https://github.com/sgl-project/sglang/pull/25299)
- **#25385** 修复 SWA disagg 模式下的 KV allocator double-free — PD disaggregation + HiCache 组合下，sliding-window attention KV allocator 存在所有权追踪错误，导致 double-free 和 stale mapping。[链接](https://github.com/sgl-project/sglang/pull/25385)

### Issues 值得关注

- **#20820** FA3 后端 + FP8_E4M3 KV cache 产生乱码输出 [已关闭] — FlashAttention-3 与 FP8（e4m3）KV cache 组合使用时输出完全损坏，关闭 FP8 KV cache 后恢复正常，提示 FA3 FP8 路径存在精度 bug。[链接](https://github.com/sgl-project/sglang/issues/20820)

---

## TensorRT-LLM

### Open PRs 值得关注

- **#13428** 新增 FlashInfer MLA attention 后端 — 为 DeepSeek 系列模型实现 FlashInfer-based MLA：prefill 使用 `BatchPrefillWithRaggedKVCacheWrapper`，decode 使用 `BatchMLAPagedAttentionWrapper`，支持零拷贝 paged KV cache 视图和外部 RoPE。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13428)
- **#14120** 默认开启 DeepSeek V4 的三项性能优化 — 将 MLA dependency-aware overlap、比率 4 的 sparse-attention indexer 多流调度、FP8 量化打包三项优化从 opt-in 改为默认开启，降低 DSv4 部署门槛。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14120)
- **#14219** DSv4：为 compress_ratio=4 启用 GVR Heuristic Top-K — 将 GVR Heuristic Top-K decode 路径扩展到 DSv4 sparse attention indexer 的 compress_ratio=4 场景，进一步提升稀疏注意力性能。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14219)
- **#14216** KVCacheManager V2 完整统计指标支持 — 为 KVCacheManagerV2 新增请求级、全局、迭代级、池组等多维统计，以及传输/拷贝遥测，便于生产监控。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14216)
- **#14060** hybrid 模型 disagg serving 下的 KV cache block reuse 支持 — 修复混合架构（Transformer + 循环层）在 disaggregated serving 中 KV cache manager 选择和 head count 处理错误。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14060)
- **#14140** 重构 KVCacheManagerV2 的 prefix reuse 作用域抽象 — 引入统一 `ReuseScope` 抽象，将 LoRA task ID 和 cache salt 路由统一管理，提升基数树前缀复用的可维护性。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/14140)
- **#13513** FP8 KV cache 激活时强制 AutoDeploy MHA kernel 输出 FP8 类型 — 修复 fused MHA kernel 在 FP8 KV cache 模式下仍输出 BF16 的问题，避免 L40S/sm_89 上缺少对应 cubin 的错误。[链接](https://github.com/NVIDIA/TensorRT-LLM/pull/13513)

---

## Blogs & 长文

- **"A First Comprehensive Study of TurboQuant: Accuracy and Performance"**（vLLM Blog，2026-05-11）— 系统评测 TurboQuant：将 KV cache 压缩至 3-4 bit 并在 attention 计算前反量化到 BF16，与 FP8 KV cache 对比精度和吞吐；结合今日多个 TurboQuant 相关 bugfix PR，显示其正在快速走向生产就绪。[链接](https://vllm.ai/blog/2026-05-11-turboquant)
- **"DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles"**（LMSYS Blog，2026-04-25）— SGLang 为 DSv4 混合稀疏 attention 架构实现 Day-0 支持，涵盖 ShadowRadix prefix cache、HiSparse CPU 扩展 KV、Flash Compressor、Lightning TopK 等多项 attention 相关优化；与今日 SGLang 的 `flash_mla_sparse_fwd` 集成 PR 形成工程呼应。[链接](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)

---

## 横向观察

三个框架今日均有 **MLA + Blackwell/SM12x 适配**动作：vLLM 合并了 MI300X MLA 修复并推进消费级 Blackwell 的 sparse-MLA 支持；SGLang 集成更快的 `flash_mla_sparse_fwd` 并将 Gemma 4 切换至 `trtllm_mha`；TensorRT-LLM 则新增 FlashInfer MLA 路径并将 DSv4 三项 MLA 优化默认开启。整体来看，MLA 的多后端工程化（FlashMLA / FlashInfer / trtllm_mha）正在从实验阶段快速进入生产默认，同时 **KV cache 多级 offloading**（CPU→S3）和 **disaggregated prefill 下的 KV 一致性**（HMA、SWA double-free、Mooncake HMA）成为共同关注的稳定性前沿。
