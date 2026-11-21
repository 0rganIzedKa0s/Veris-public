# Latency roadmap — cutting time-to-first-spoken-word on the pack

Baseline (measured, docs/orin-benchmarks.md): **3B @ 15W = 12.8s TTFT,
14.5 tok/s**; 7B @ 25W = 17.5s TTFT, 7.4 tok/s. TTFT ≈ vision encode +
prefill of ~1300 vision tokens; decode is memory-bandwidth-bound (68 GB/s
LPDDR5 is the wall — Super mode does NOT raise it; that claim was refuted in
verification). Evidence from an adversarially-verified literature review
(sources inline; every claim was independently re-checked before it was
allowed to stand); the honest caveat up front: **most published speedups were
measured on faster silicon** (Apple M-series, datacenter GPUs, DRIVE Thor),
so every number below needs local re-benchmarking. Directionally they are
consistent and they stack, because they attack different bottlenecks.

## The stack, in order of expected leverage

### 1. Cut vision tokens — the TTFT lever (target: 12.8s → ~4-6s)
- **Cheapest first: lower the image budget.** We ship 1280px ≈ 1300 tokens.
  Qwen2.5-VL's token count scales ~linearly with pixels; 896px ≈ ~640 tokens
  ≈ half the prefill. Run the description-quality eval (style-guide rubric)
  at 1280/1024/896/768 and find the knee. Zero engineering, do this week.
- **Token pruning:** GlimpsePrune drops 92.6% of visual tokens on
  Qwen2.5-VL-7B with *unchanged* free-form VQA accuracy
  (arxiv.org/2508.01548); LUVC prunes ~62% for ~2x LM speedup
  (arxiv.org/2512.09010). Caveats verified: GlimpsePrune is
  query-conditioned (great for on-demand questions, weaker for open scene
  description) and prunes after the encoder pass — it cuts prefill, not
  encoder time. Both are preprints; treat as experiments, not plans.
- **Bigger swing: token-efficient encoder (FastVLM/FastViTHD class)** —
  4-16x fewer tokens, 3.2x TTFT at equal quality (CVPR 2025,
  arxiv.org/2412.13303). This means changing the VLM, so it belongs in the
  fine-tuning decision: if we fine-tune anyway, evaluating a FastVLM-class
  backbone alongside Qwen is nearly free optionality.

### 2. Vision encoder as TensorRT engine (target: encoder ~2x)
NanoLLM does NOT support Qwen2.5-VL (verified from its docs) — the viable
Jetson paths are llama.cpp (current), or vLLM/TensorRT-LLM containers with an
INT8/FP8 TensorRT engine for the ViT. Expect meaningful encoder gains but
budget a week of toolchain pain; do after the free wins.

### 3. Prefix caching — recover the constant system prompt (~0.5-1s)
Our system prompts are ~350 tokens and constant per mode. llama.cpp keeps
the common prefix KV when `cache_prompt` is on and the prefix matches —
but the image sits early in the message, breaking the prefix for new frames.
Restructure the template so [system + mode + instructions] precede the image
tokens, enable cache_prompt in production (we disabled it only for honest
benchmarking). Small but free. (A claimed "5x TTFT from system-prompt KV
reuse in TensorRT-LLM" was REFUTED in verification — expect modest gains
proportional to prefix share.)

### 4. Speculative decoding — the decode lever (measured on our exact models)
ViSpec: **1.87x on Qwen2.5-VL-3B, 1.80x on 7B**, lossless by construction
(arxiv.org/2509.15235). Self-speculative CATS runs on Jetson Orin without a
separate draft model (arxiv.org/2508.16201). Decode at 14.5 tok/s is already
~2x speech rate, so this matters most for long text_reading outputs and for
freeing GPU time for the reflex channel. Medium effort; after TTFT work.

### 5. Frame differencing / encoder-output reuse (situational)
VLCache-style encoder+KV reuse gives 1.2-16x TTFT but only on same-image
hits (arxiv.org/2512.12977) — a live camera rarely repeats frames exactly.
The practical wearable version: IMU-gated frame differencing — if the head
hasn't moved and the scene diff is tiny, skip the generation entirely and
say nothing new. That's a UX win (silence is a feature) and a 100% latency
win on skipped frames.

### 6. Pipeline overlap — make 4s feel like 1s
- **Streaming TTS now:** Piper on Orin reaches ~0.8-0.9s to first audio with
  a warm persistent process (single-source number; verify locally). We
  already stream tokens; speak at first clause boundary. Perceived latency
  becomes TTFT + ~1s, and the first clause should be the mode's
  most-important fact (the prompts already order it that way).
- **Progressive two-stage response:** reflex channel (detector on frame
  arrival, <100ms class words: "person ahead") speaks immediately; VLM
  refines. This is the sensor-fusion architecture doing latency work.
- **Pre-encode while speaking:** start encoding the next frame during TTS
  playback of the previous answer; on-demand requests then hit a warm
  half-finished prefill.

## Composite projection (needs local validation, but the shape is)

| Stage | TTFT | Perceived (first audio) |
|---|---|---|
| Today (3B @ 15W, 1280px) | 12.8s | ~13.5s |
| + 896px budget | ~6-7s | ~7-8s |
| + prefix cache + template reorder | ~5.5-6.5s | ~6.5-7.5s |
| + streaming TTS first-clause | ~5.5-6.5s | **~2-4s + reflex channel at ~0.3s** |
| + pruning or TRT encoder (pick winner) | ~3-4s | ~1.5-2.5s |

The reflex channel (sensors doc) is what makes the product feel instant;
the VLM roadmap makes the description arrive while the moment is still
relevant.

## Measured results — experiments executed 2026-07-04 (3B @ 15W on the Orin)

| Change | Result | Shipped? |
|---|---|---|
| 896px nav budget | 12.8s → ~5.0s TTFT | superseded by 768px |
| **768px nav budget** | prefill 3.5-4.1s, laterality intact (640px loses left/right → rejected) | ✅ per-mode default (text_reading stays 1280px) |
| **cache_prompt=true** | new frame: −0.9s (system prefix reused); repeated/static frame: **0.15s TTFT, 1.1s total** (llama-server holds multiple recent prompt caches); stable over 10+ requests | ✅ hub default |
| Piper TTS on Orin CPU | 0.17-0.19s per clause warm (RTF 0.09), voice load 1.7s once, ~200MB | ready to wire into the pack loop |
| Flash attention (-fa) | no prefill gain (5.17s vs 4.98s); server crash on 2nd request | ❌ rejected |
| Speculative decoding (0.5B draft, --spec-draft-n-max 8) | decode unchanged at 14.2 tok/s — llama.cpp doesn't engage the draft path with mtmd/vision models | ❌ rejected for now; revisit on TensorRT-LLM |

**Net effect through the demo hub (verified end-to-end):** cold new frame
**3.6s TTFT / 4.6s total** (was 13.5s yesterday — 3.7×); repeated frame
**0.15s TTFT / 1.1s total**. With Piper's 0.2s clause synth on-device, first
spoken word on a cold frame lands at ~4s, and the remaining roadmap items
(TensorRT vision encoder, token pruning, FastVLM-class backbone at
fine-tuning time) target the residual ~3.5s of vision-encode+prefill.

Ops note: any llama-server experiment loop MUST drop page cache between
restarts (`prep_mem.sh`) — two "crashes" during these experiments were NvMap
OOM from accumulated cache, not the features under test. Also, one llama.cpp
prefill quirk: prefill rate ~220-260 tok/s includes the ViT encode, so
smaller images help twice (fewer tokens AND faster encode).

## Remaining experiments
1. Image-budget sweep 1280→768 — **DONE (2026-07-04, crossing frame, 3B@15W,
   measured on the Orin):** 1280px = 13.7s, 1024px = 7.8s, 896px = 5.1s,
   768px = 4.2s TTFT, with descriptions essentially identical (one
   truck→"bus" slip at 896px). TTFT scales ~linearly with pixels as
   predicted. Remaining work: verify on text_reading (small text is where
   resolution should matter most — may want per-mode budgets: 768px nav,
   1280px text_reading) and on real head-cam frames.
2. Prompt-template reorder + cache_prompt=true, measure warm-frame TTFT.
3. Piper TTS on the Orin beside the 3B server (RAM fits: ~5.1GB + ~200MB),
   measure true first-audio.

## vLLM container evaluation — FAILURE ANALYSIS (2026-07-05, 3B on the Orin)

Goal was to benchmark the `dustynv/vllm` container running Qwen2.5-VL-3B
(AWQ) against the llama.cpp baseline, targeting prefill/TTFT. **Result: vLLM
never reached a serving state on the 8GB Orin. No benchmark numbers; the
finding is that vLLM 0.9.2 does not fit this model on 8GB unified memory.**
Demo stack restored and verified green (/api/status ok, HTTPS 200, llama
health ok) before and after.

### Live llama.cpp baseline captured this session (for the intended compare)
Veris-3B-v1 Q4_K_M, 15W, 1280px, cold (cache_prompt off), via `bench/client.py`
(raw: `bench/vllm_results.jsonl`):

| Mode | TTFT (s) | Decode (tok/s) |
|---|---|---|
| outdoor_nav | 13.8 | 14.5 |
| indoor_nav | 12.2 | 14.1 |
| text_reading | 13.8 | 14.0 |
| describe | 12.3 | 14.1 |

Median TTFT ~12.9s, decode ~14.1 tok/s — consistent with the 2026-07-04 table.

### Container facts
- **`dustynv/vllm:r36.4.0` (the pulled/instructed image) ships vLLM 0.6.3**,
  which predates Qwen2.5-VL support (`Qwen2_5_VLForConditionalGeneration` not
  in the model registry; arch added in vLLM 0.7.2). **Unusable for this model.**
- Pivoted to `dustynv/vllm:0.9.2-r36.4-cu128-24.04` (also already on the
  board) — vLLM 0.9.2, `qwen2_5_vl` registered. All work below used it.
- Model: pre-downloaded AWQ `Qwen/Qwen2.5-VL-3B-Instruct` at
  `~/models/qwen2.5-vl-3b-awq` (int4 gemm; **visual tower left fp16 /
  unquantized** per `modules_to_not_convert:["visual"]`; 3.4GB safetensors).

### Failure chain — each cause fixed, revealing the next
1. **`vllm serve` CLI crashes** → `ModuleNotFound: pandas` (its CLI eagerly
   imports the `vllm.benchmarks` subcommand, whose datasets need pandas, which
   is absent in the image). Fix: bypass the CLI —
   `python3 -m vllm.entrypoints.openai.api_server ...`.
2. **Vision-encoder `conv3d` in `profile_run`** →
   `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED` at
   `CUDACachingAllocator.cpp:1016`. PyTorch's allocator NVML query is
   unsupported on Tegra/Jetson. Fix:
   `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (avoids the NVML
   code path). This got profiling to run.
3. **KV cache allocation** → `ValueError: No available memory for the cache
   blocks` (`available_memory <= 0`). This is the hard wall. It persisted
   across every lever tried:
   - `--gpu-memory-utilization` 0.60 → 0.85 → 0.88
   - `--swap-space` 4GiB → 0 (the default 4GiB CPU swap double-books unified
     RAM but was not the sole cause)
   - hub + context_engine stopped to free physical RAM (6.9GB free at launch)
   - `--limit-mm-per-prompt image=1` **still profiled a *video*** (2592-token
     dummy) → also disabled with `{"image":1,"video":0}`
   - `--mm-processor-kwargs max_pixels` 1.6M → 1.0M → 262K (shrinks the ViT
     activation peak)
   - `--enforce-eager`, `--max-num-seqs 1`, `--max-model-len 4096` throughout.

### Root cause (fundamental, not a tunable)
On the 8GB unified-memory Orin, vLLM 0.9.2's memory profiler cannot find KV
headroom: **weights 3.32 GiB + Jetson PyTorch/CUDA runtime context
(~1.5-2 GiB; FlashAttention backend, torch kernels) + the fp16 visual-tower
profiling activation peak** together consume the entire
`gpu_memory_utilization × 7.44GiB` budget, so computed available KV ≤ 0 — even
with the demo fully stopped and util at 0.88. llama.cpp fits the *same* 3B in
~5.1 GiB (via `--no-mmap`, `-np 1`, page-cache drop) precisely because it
carries no multi-GiB Python/PyTorch runtime and can keep the vision tower
quantized (Q8_0 mmproj). vLLM's runtime footprint alone is fatal here.

### Conclusion / recommendation
- **vLLM is not a viable on-device engine for Qwen2.5-VL-3B on the 8GB Orin
  as-is.** It does not change the TTFT story on this device — it cannot even
  serve. Keep **llama.cpp as the shipping on-device engine.**
- Making vLLM fit would require: (a) a build/quant that does *not* keep the
  fp16 visual tower resident (quantize or offload the ViT), (b) a slimmer
  runtime, or (c) more memory (Orin NX 16GB / Thor). The TensorRT-LLM ViT
  engine (roadmap §2) is the remaining container path for encoder speedups and
  should be evaluated on larger-memory silicon, not chased further on 8GB.
- Ops note for any retry: `dustynv/vllm:r36.4.0` is too old (0.6.3); use the
  0.9.2 image, `python3 -m vllm.entrypoints.openai.api_server`, and
  `expandable_segments:False`. `bench/client.py --model <name> --base-url
  http://192.168.55.1:8001` benchmarks vLLM identically to llama.cpp once a
  server is up (OpenAI-compatible; `--model` set triggers the vLLM path).
