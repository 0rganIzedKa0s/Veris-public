#!/usr/bin/env bash
# One-time setup ON THE JETSON (Orin Nano 8GB, JetPack 6.x).
# Builds llama.cpp with CUDA and downloads Qwen2.5-VL GGUF quants + vision
# projectors. Run as the normal user; sudo only where marked.
set -euo pipefail

echo "== 1. System info =="
cat /etc/nv_tegra_release || true
free -h

echo "== 2. Free up RAM: boot to console (GUI eats ~1GB of the 8GB) =="
echo "   (run manually if you agree): sudo systemctl set-default multi-user.target && sudo reboot"

echo "== 3. Max power mode =="
# JetPack 6.2+ 'Super' modes: list with nvpmodel -p --verbose. MAXN SUPER is
# typically mode 2 on Orin Nano; 15W is mode 0. Benchmark BOTH.
sudo nvpmodel -q || true

echo "== 4. Build llama.cpp with CUDA =="
sudo apt-get update && sudo apt-get install -y build-essential cmake git libcurl4-openssl-dev
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
if [ ! -d "$HOME/llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
fi
cd "$HOME/llama.cpp"
# 87 = Orin (Ampere, sm_87)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --config Release -j"$(nproc)" --target llama-server

echo "== 5. Download models (GGUF + mmproj vision projector) =="
mkdir -p "$HOME/models"
cd "$HOME/models"
pip3 install -U "huggingface_hub[cli]" 2>/dev/null || python3 -m pip install -U "huggingface_hub[cli]"
hf download ggml-org/Qwen2.5-VL-3B-Instruct-GGUF --include "*Q4_K_M*" "*mmproj*" --local-dir qwen2.5-vl-3b
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF --include "*Q4_K_M*" "*mmproj*" --local-dir qwen2.5-vl-7b

echo "== 6. Done. Start a server like: =="
cat <<'EOF'
  ~/llama.cpp/build/bin/llama-server \
    -m ~/models/qwen2.5-vl-7b/*Q4_K_M*.gguf \
    --mmproj ~/models/qwen2.5-vl-7b/*mmproj*.gguf \
    -ngl 99 --host 0.0.0.0 --port 8080 -c 4096
EOF
