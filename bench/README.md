# Jetson Orin Nano benchmark harness

Measures the product question: **what does each model/quant/power-mode cost in
latency, power, and memory on the actual device**, using the exact prompts and
frames from the demo pipeline.

## Topology

Dev box (this repo) drives everything; the Nano only runs `llama-server` and
`tegrastats`. USB device mode puts the Nano at `192.168.55.1`.

## One-time setup

1. Plug in the Nano (USB-C), verify: `ping 192.168.55.1`
2. Passwordless SSH (type in your terminal, needs interactive password):
   `! ssh-copy-id <user>@192.168.55.1`
3. Copy repo bits and run setup on the board:
   ```bash
   scp -r bench test_frames <user>@192.168.55.1:~/veris-bench/
   ssh <user>@192.168.55.1 'bash ~/veris-bench/bench/jetson_setup.sh'
   ```
   (Builds llama.cpp with CUDA for sm_87, downloads Qwen2.5-VL 3B/7B Q4_K_M
   GGUF + mmproj. Consider booting to console to free ~1 GB RAM — the 7B is
   tight on the shared 8 GB.)

## Per-configuration run

On the Nano (two shells or `ssh -f`):

```bash
sudo nvpmodel -m <mode>          # 0 = 15W, MAXN SUPER per nvpmodel -q; reboot if asked
sudo jetson_clocks               # lock clocks for repeatable numbers
~/llama.cpp/build/bin/llama-server -m ~/models/qwen2.5-vl-7b/*Q4_K_M*.gguf \
    --mmproj ~/models/qwen2.5-vl-7b/*mmproj*.gguf -ngl 99 --host 0.0.0.0 --port 8080 -c 4096
python3 ~/veris-bench/bench/tegrastats_logger.py --out ~/veris-bench/tegra.jsonl
```

From the dev box:

```bash
uv run python bench/client.py --base-url http://192.168.55.1:8080 \
    --tag "7B-Q4_K_M@MAXN" --runs 3
scp <user>@192.168.55.1:~/veris-bench/tegra.jsonl bench/
uv run python bench/analyze.py --results bench/results.jsonl --tegra bench/tegra.jsonl
```

## Matrix to record

| tag | model | power mode |
|---|---|---|
| 3B-Q4_K_M@15W  | Qwen2.5-VL-3B Q4_K_M | 15W |
| 3B-Q4_K_M@MAXN | Qwen2.5-VL-3B Q4_K_M | MAXN SUPER |
| 7B-Q4_K_M@15W  | Qwen2.5-VL-7B Q4_K_M | 15W |
| 7B-Q4_K_M@MAXN | Qwen2.5-VL-7B Q4_K_M | MAXN SUPER |

Every result row records TTFT, total time, tok/s, and the merged tegrastats
window gives mean/max watts, peak RAM, max temp — the numbers that decide
battery size and whether 7B is viable on-device, and the descriptions
themselves so quality can be compared against the RTX 3070 runs in
`docs/baseline-observations.md`.
