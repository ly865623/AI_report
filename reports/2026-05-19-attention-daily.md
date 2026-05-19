# Attention Daily — 2026-05-19

> 时间窗口：2026-05-18T01:03Z — 2026-05-19T01:03Z  
> 覆盖仓库：vllm-project/vllm · sgl-project/sglang · NVIDIA/TensorRT-LLM

---

## TL;DR

- **SGLang #23515** 实现层间流水线 KV 传输，将 RDMA 与 GPU 计算重叠，PD 分离场景下 ≥1K token prompt 的 TTFT 降低 **48–68%** [→](https://github.com/sgl-project/sglang/pull/23515)
- **TRT-LLM #13428** 为 DeepSeek 系模型添加 **FlashInfer MLA 后端**，context 阶段用 ragged KV wrapper、generation 阶段用 paged MLA wrapper [→](https://github.com/NVIDIA/TensorRT-LLM/pull/13428)
- **TRT-LLM Gemma4 SWA 三件套**（#14134/#14121/#13745）同步推进：chunked prefill 适配、Variable SWA 前驱驱逐 Triton kernel、多 head_dim KV 池统一管理 [→](https://github.com/NVIDIA/TensorRT-LLM/pull/14134)
- **vLLM #42095** 将 FlexAttention 与 FlashAttention 统一到 **num-blocks-first KV 布局**，为 KV Connector 与 speculative decoding 兼容性奠基 [→](https://github.com/vllm-project/vllm/pull/42095)
- 今日三个框架均有密集的 **MLA 多平台修复**（ROCm/XPU/NPU/SM12x），MLA 支持正进入各框架的收尾阶段

---

## vLLM

### Merged PRs

（今日 GitHub 搜索仅返回 open 状态，已合并 PR 暂无数据）

### Open PRs 值得关注

**FP8 / NVFP4 KV Cache**

- **#42080** [Triton attention FP8 per-tensor Q scale 修复](https://github.com/vllm-project/vllm/pull/42080) — 修复 per-tensor KV cache 模式下 Q scale 被忽略导致结果错误的 bug，已标 ready
- **#42890** [NVFP4 KV cache 与 sliding-window 共存](https://github.com/vllm-project/vllm/pull/42890) — 移除阻碍 NVFP4 KV 与 `--kv-cache-dtype-skip-layers sliding_window` 同时生效的 padding 限制
- **#42971** [DFlash prefix cache 数据污染修复](https://github.com/vllm-project/vllm/pull/42971) — speculative decoding 下缺少 lookahead block 导致 KV cache 持续损坏，已标 bug+ready

**MLA / DeepSeek V4 多平台**

- **#42316** [DSv4 FlashInfer 稀疏 MLA kernel](https://github.com/vllm-project/vllm/pull/42316) — 将 DSv4 FlashInfer 稀疏 MLA 分支 rebase 到 main，Triton indexer-Q 保持默认路径
- **#42956** [ROCm MI35x FP8 KV 稀疏 MLA 修复](https://github.com/vllm-project/vllm/pull/42956) — AITER decode kernel 缺少 FP8 路径，DSv3.2 TP=4 下输出空/乱码
- **#41834** [SM12x Blackwell 消费卡 DSv4 Flash 支持](https://github.com/vllm-project/vllm/pull/41834) — RTX 5090/DGX Spark GB10 等无 TMEM/tcgen05 的硬件上启用 DSv4 Flash

**Attention 布局 / Backend**

- **#42095** [FlexAttention + FlashAttention 统一 KV 布局](https://github.com/vllm-project/vllm/pull/42095) — num-blocks-first 重构，KV Connector 与 speculative decoding 兼容前提
- **#42990** [sliding window cache 命中搜索 O(n)→strided probe](https://github.com/vllm-project/vllm/pull/42990) — 替代线性扫描，提升 SWA 场景 prefix cache 查找效率

**KV Connector / 分离推理**

- **#42828** [Mooncake 连接器 HMA hybrid KV 支持](https://github.com/vllm-project/vllm/pull/42828) — 为 DSv4 等混合 layout 模型接入 Hybrid Memory Architecture

### Issues 值得关注

- **#43021** [RFC：融合 QK Norm + mRoPE + KV cache 写 + FP8 量化](https://github.com/vllm-project/vllm/issues/43021) — 消除 Qwen3-VL 等模型多个中间显存 round-trip 的 fused kernel 提案
- **#42948** [Bug：DSv4-Flash hybrid 0% prefix cache 命中](https://github.com/vllm-project/vllm/issues/42948) — 每次调度重分配清空所有 attention group 的 block cache key，影响生产

---

## SGLang

### Merged PRs

（今日 GitHub 搜索仅返回 open 状态，已合并 PR 暂无数据）

### Open PRs 值得关注

**KV 传输 / PD 分离**

- **#23515** [层间流水线 KV 传输，RDMA 与 GPU 计算重叠](https://github.com/sgl-project/sglang/pull/23515) — **本日最重要进展**：逐层组传 KV 同时计算下一组，Qwen2.5-72B ≥1K token prompt TTFT 降幅 48–68%

**MLA 修复与优化**

- **#25556** [AMD AITER MLA page-size>1 正确性修复](https://github.com/sgl-project/sglang/pull/25556) — page size 超过 1 时 ROCm AITER MLA 后端输出错误
- **#25460** [MLA prefill FP8 JIT kernel + prepare_prefill_qkv 扩展点](https://github.com/sgl-project/sglang/pull/25460) — 为 Blackwell 添加 FP8 量化 JIT kernel，在 MLA 后端增加 QKV 预处理钩子
- **#25144** [Ascend NPU 完整 DSv4-Flash 支持](https://github.com/sgl-project/sglang/pull/25144) — 910 A3 NPU 上新增 V4 attention 后端、稀疏注意力自定义算子、c1/c4/c128 KV 压缩路径，GSM8K 97.0%

**KV Cache 量化**

- **#25555** [MXFP4 KV cache BlockFP4-Hadamard 量化路径](https://github.com/sgl-project/sglang/pull/25555) — 扩展 MXFP4 KV cache，增加 BlockFP4/Hadamard 方案提升非 Blackwell 精度

**Speculative Decoding + 长上下文**

- **#22892** [EAGLE/MTP + SWA 在 flashinfer 和 triton 后端修复](https://github.com/sgl-project/sglang/pull/22892) — 修复两个后端 speculative decoding 路径下 sliding-window attention 失效
- **#14194** [DeepSeek v2 MLA Decode Context Parallel](https://github.com/sgl-project/sglang/pull/14194) — 首次为 DSv2 MLA 实现 DCP，8xH20 下支持更长 context

### Issues 值得关注

- **#25512** [EAGLE v2 verify + DSv4 压缩 attention segfault](https://github.com/sgl-project/sglang/issues/25512) — CUDA graph padding 污染 EAGLE v2 verify，8–13 并发请求下生产崩溃
- **#22682** [EAGLE3 draft runner 全局覆写导致 MLA + chunked prefix cache 路由损坏](https://github.com/sgl-project/sglang/issues/22682) — draft model 全局设 `disable_chunked_prefix_cache=True` 破坏 target model dispatch
- **#24656** [RFC：Agent-Aware KV Cache Phase 1](https://github.com/sgl-project/sglang/issues/24656) — 基于 agent session 拓扑的 KV cache 管理与驱逐策略设计提案

---

## TensorRT-LLM

### Merged PRs

- **#14195** `[doc] Update spec dec support matrices` — 纯文档更新，无技术实质，跳过

### Open PRs 值得关注

**FlashInfer MLA**

- **#13428** [FlashInfer MLA 后端](https://github.com/NVIDIA/TensorRT-LLM/pull/13428) — context 阶段用 `BatchPrefillWithRaggedKVCacheWrapper`，generation 阶段用 `BatchMLAPagedAttentionWrapper`，完整覆盖 DeepSeek 系模型两阶段

**Gemma4 SWA 全栈**

- **#14134** [Gemma4 chunked prefill + SWA](https://github.com/NVIDIA/TensorRT-LLM/pull/14134) — 四种 Gemma4 变体支持 chunked prefill；SWA 层在 trtllm-gen 缺 cubin 时回退 Triton；修复 B200 超大 page ID int64 溢出
- **#14121** [Variable SWA 前驱驱逐 Triton kernel](https://github.com/NVIDIA/TensorRT-LLM/pull/14121) — 支持异构 head_dim 与 dtype 的 per-window KV 配置，含 64K token passkey 验证测试
- **#13745** [Gemma4 多 head_dim KV 池统一管理](https://github.com/NVIDIA/TensorRT-LLM/pull/13745) — 重构 KVCacheManager 同时服务 SWA (head_dim=256) + global attention (head_dim=512) 两个池

**Attention API 重构**

- **#14275** [移除 attention 接口 sink_token_length](https://github.com/NVIDIA/TensorRT-LLM/pull/14275) — 清理 StreamingLLM 遗留参数，消除 KV manager 预留 sink block 而 kernel 忽略的静默错误
- **#14279** [thop.attention ~75 位置参数 → 显式 kwargs](https://github.com/NVIDIA/TensorRT-LLM/pull/14279) — 为后续 attention dispatch 重构打基础

**KV Cache Bug 修复**

- **#14245** [KV cache V2 stream fence 修复（4×B300 非法内存）](https://github.com/NVIDIA/TensorRT-LLM/pull/14245) — stream 不匹配的 CUDA Event.record 导致 H2D copy 非法访问
- **#14231** [多模态模型自动禁用 KV block reuse](https://github.com/NVIDIA/TensorRT-LLM/pull/14231) — 相同多模态请求命中错误前缀 KV 块的崩溃，embedding hash 未纳入 block hash
- **#13513** [FP8 KV cache 激活时强制 FP8 输出](https://github.com/NVIDIA/TensorRT-LLM/pull/13513) — 修复 L40S (sm_89) head_dim=192 下缺失 FP8 cubin 的 kernel 失败
- **#13713** [分离 KV 传输取消与生命周期硬化](https://github.com/NVIDIA/TensorRT-LLM/pull/13713) — 修复长 prompt 突发 + 客户端取消时 KV 传输线程永久卡死

**DSv4 / 稀疏注意力**

- **#14219** [DSv4 GVR Heuristic Top-K 支持 compress_ratio=4](https://github.com/NVIDIA/TensorRT-LLM/pull/14219) — 扩展稀疏注意力启发式 Top-K 可用范围从 compress_ratio=1 到 {1,4}
- **#14008** [Eagle cu_seqlen 数据竞争修复](https://github.com/NVIDIA/TensorRT-LLM/pull/14008) — 修复 Llama-3.1-8B+Eagle3 两个 attention 后端上的 cu_seqlen tensor race

---

## Blogs & 长文

今日（2026-05-17 至 19）三个团队均无新博客发布，以下为近期可参考的相关内容：

- [**TurboQuant 首次系统评估**（vLLM，2026-05-11）](https://vllm.ai/blog/2026-05-11-turboquant) — vLLM v0.20.0 集成 2/3-bit KV cache 量化，FA4 重新成为 SM90+ 默认 MLA prefill 后端
- [**HiSparse：稀疏 attention 分层内存加速**（LMSYS，2026-04-10）](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/) — GPU+CPU 分层 KV cache + 稀疏 Top-K，256 并发下 >3x 基线吞吐，近线性扩展
- [**Skip Softmax 加速长上下文推理**（NVIDIA，2026-03-16）](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/) — FlashAttention kernel 内动态跳过低分 attention block，无需重训，最高 1.4x TTFT/TPOT 提升

---

## 横向观察

今日呈现三个清晰的趋同方向：

**1. MLA 多平台收尾**：vLLM（ROCm/XPU/SM12x Blackwell）、SGLang（AMD/Intel XPU/Ascend NPU）、TRT-LLM（FlashInfer MLA 后端）在同一天均有 MLA 后端修复或新平台上线。MLA 的核心算法已稳定，各框架正在密集解决边界平台的正确性与性能问题。

**2. PD 分离 KV 传输成瓶颈**：SGLang 层间流水线（#23515）、TRT-LLM KV 传输生命周期硬化（#13713）、vLLM NIXL KV push（#35264）三条线独立展开，共同指向同一问题：在 PD 分离架构下，KV 搬运延迟已成为 TTFT 的主要来源，且各框架的解法路径不同（流水线重叠 vs. 取消稳健性 vs. prefill 主动推送）。

**3. 低精度 KV cache 长尾 bug 集中爆发**：SGLang MXFP4 Hadamard 路径、vLLM NVFP4+sliding window 组合、TRT-LLM FP8 输出修复（L40S）在同一天活跃，说明 FP4/FP8 KV cache 在非旗舰卡和边界组合配置下的兼容性问题仍在持续修复中。
