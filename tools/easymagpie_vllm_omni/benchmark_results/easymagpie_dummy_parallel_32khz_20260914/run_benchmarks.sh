#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="$ROOT/benchmark_results/easymagpie_dummy_parallel_32khz_20260914"
INPUT_FILE="$RESULT_DIR/benchmark_inputs.tsv"
DEPLOY_CONFIG="$ROOT/deploy/easymagpie_dummy.yaml"
MODEL="$ROOT/converted_model_roy_fullsize_32khz_parallel_dummy"
PORT=8091
SERVER_PID=""

export PYTHONPATH="$RESULT_DIR/nvml_fallback_v4:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HOME/.cache/easymp_cache/hf"
export TRITON_CACHE_DIR="$HOME/.cache/easymp_cache/triton"
export VLLM_CACHE_ROOT="$HOME/.cache/easymp_cache/vllm"
export FLASHINFER_WORKSPACE_BASE="$HOME/.cache/easymp_cache/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/easymp_cache/inductor"
export TORCH_HOME="$HOME/.cache/easymp_cache/torch"
export CUDA_CACHE_PATH="$HOME/.cache/easymp_cache/nv"
export XDG_CACHE_HOME="$HOME/.cache/easymp_cache/xdg"

stop_server() {
    if [[ -z "$SERVER_PID" ]]; then
        return
    fi
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            return
        fi
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

trap stop_server EXIT INT TERM

setsid conda run -n easymagpie-vllm --no-capture-output     env EASYMAGPIE_DEPLOY_CONFIG="$DEPLOY_CONFIG"     bash "$ROOT/scripts/run_server.sh" "$MODEL" "$PORT" >"$RESULT_DIR/parallel_server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 900); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        wait "$SERVER_PID" || true
        tail -100 "$RESULT_DIR/parallel_server.log"
        exit 1
    fi
    sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

{
    echo "started_at=$(date -Iseconds)"
    echo "git_commit=$(git -C "$ROOT/../.." rev-parse HEAD)"
    echo "model=$MODEL"
    echo "codebook_prediction_mode=parallel"
    echo "local_transformer_n_layers=0"
    echo "deploy_config=$DEPLOY_CONFIG"
    echo "requests_per_run=128"
    echo "concurrency=1,32"
    echo "repetitions=5"
    echo "max_new_tokens=128"
    echo "sample_rate=32000"
    conda run -n easymagpie-vllm --no-capture-output python -c         "import importlib.metadata as m, torch; print(f'torch={torch.__version__}'); print(f'vllm={m.version(\"vllm\")}'); print(f'vllm_omni={m.version(\"vllm-omni\")}'); print(f'gpu={torch.cuda.get_device_name(0)}')"
} >"$RESULT_DIR/metadata.txt"

for run in $(seq 1 5); do
    echo "[parallel] benchmark run $run/5"
    conda run -n easymagpie-vllm --no-capture-output         python "$ROOT/scripts/benchmark_server.py"         --text-file "$INPUT_FILE"         --num-requests 128         --concurrency 1 32         --url "http://127.0.0.1:$PORT"         --max-new-tokens 128         --sample-rate 32000         2>&1 | tee "$RESULT_DIR/parallel_run${run}.log"
done

stop_server
echo "completed_at=$(date -Iseconds)" >>"$RESULT_DIR/metadata.txt"
