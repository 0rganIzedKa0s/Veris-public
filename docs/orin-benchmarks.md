# Orin Nano 8GB benchmarks — Qwen2.5-VL on the real target (2026-07-04)

Device: Jetson Orin Nano Devkit 8GB, JetPack 6.2 (L4T r36.4.7, Super profile
enabled via FDT), root on NVMe (Samsung 990 PRO, 1.8 GB/s), GUI disabled.
Stack: llama.cpp (CUDA sm_87), Q4_K_M GGUF weights + Q8_0 mmproj, `-np 1`,
`--no-mmap`, 1280px frames (~1300 vision tokens), prompt cache off (every
real frame is new), streaming. Metrics: median over 2 runs × 4 frame/mode
cells; power/RAM/thermals merged from tegrastats during generation windows.

| Config | TTFT median (s) | Decode tok/s | Avg draw (W) | Peak RAM (MB) | Max temp (°C) |
|---|---|---|---|---|---|
| 3B @ 7W | 33.7 | 5.4 | 7.0 | 5,169 | 54 |
| 3B @ 15W | 12.8 | 14.5 | 10.6 | 5,133 | 59 |
| 3B @ 25W | 12.6 | 14.8 | 10.3 | 5,171 | 59 |
| 3B @ MAXN_SUPER | 12.6 | 14.8 | 10.7 | 5,146 | 60 |
| 7B @ 25W | 17.5 | 7.4 | 10.9 | 7,525 | 60 |
| 7B @ MAXN_SUPER | 18.9 | 6.3 | 10.9 | 7,522 | 60 |
| 7B @ 15W / 7W | — crashes — | | | | |

## Findings

1. **15W is the operating point.** For the 3B, 25W and MAXN buy nothing over
   15W: the workload draws only ~10.5W and is memory-bandwidth-bound
   (68 GB/s LPDDR5 is the wall, not compute). Battery math can assume
   ~10.5W under load, ~60°C worst case — thermally trivial.
2. **7W mode is not a usable inference mode** (2.6× worse TTFT, 5 tok/s) —
   it's the idle/standby mode between requests.
3. **7B requires ≥25W — hard constraint.** At 15W and 7W the server dies
   mid-generation with kernel `nvgpu` channel faults (TSG unbind errors),
   reproduced across fresh boots, with and without jetson_clocks. Combined
   with 7,525MB peak RAM (of 7,620 usable), the 7B has zero headroom: no
   detector, no ASR, no TTS beside it. **The shipping-device model is a 3B;
   7B is the dev-box teacher for distillation.**
4. **TTFT is the product problem.** ~12.6s cold-frame at 15W (vision encode +
   prefill dominated). Decode (14.8 tok/s ≈ 2× speaking rate) is fine.
   Latency roadmap (docs/latency-roadmap.md) attacks TTFT specifically.
5. 7B@MAXN measured slightly *slower* than 25W — within thermal/scheduling
   noise; treat as equal.

## Memory recipes that made this work (needed for any on-device serving)

- Drop page cache before load AND keep dropping during it
  (`bench/prep_mem.sh` on the board): Tegra's NvMap cudaMalloc can't reclaim
  page cache, and the GGUF read refills it mid-load.
- `--no-mmap` (avoid double-occupancy: mmap'd file + GPU copy).
- `-np 1` (llama-server defaults to 4 slots → 4× compute buffers).
- One 7B server lifecycle per boot (NvMap fragmentation kills the second
  load); 3B restarts fine.
- Boot to console (`systemctl set-default multi-user.target`): the GUI holds
  ~1GB.
- `--image-max-tokens 1600` is **required** for the 3B under the full boot
  stack (hub + context engine + tailscaled + docker). Without it the ~705MB
  CUDA vision compute buffer fails to allocate. Set it via
  `VERIS_LLM_EXTRA` in the board's `.env`; `ops/orin/restart_llm.sh`
  persists whatever flags you pass it.

  **The argument REPLACES the stored flags**, so `restart_llm.sh ""` silently
  drops the cap. The resulting failure is delayed and misleading: the unit
  logs "model loaded", `/health` answers `ok`, and llama-server then SEGVs on
  the FIRST image request and restarts into the same broken state — so a
  health check says the board is fine while every description crashes it.
  Diagnose from `journalctl -u veris-llm` (look for the `cudaMalloc failed`
  line), never from `/health`. Always pass the flags you want kept:
  `restart_llm.sh "--image-max-tokens 1600"`.

## Reproduction

`bench/run_matrix.sh` (3B all modes), `bench/run_7b_remaining.sh`
(reboot-per-config), `bench/client.py` (streams with cache_prompt=false),
`bench/tegrastats_logger.py`, `bench/analyze.py`. Raw data:
`bench/results.jsonl` + `bench/tegra-*.jsonl`.
