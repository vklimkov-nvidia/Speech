#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="$ROOT/benchmark_results/easymagpie_dummy_backbone_codebooks_32khz_20260914"
INPUT_FILE="$RESULT_DIR/benchmark_inputs.tsv"
DEPLOY_CONFIG="$ROOT/deploy/easymagpie_dummy.yaml"
PORT=8091
SERVER_PID=""

VARIANTS=(attention_ffn mamba_ffn attention)
MODELS=(
    "$ROOT/converted_model_roy_fullsize_32khz_backbone_attn_ffn_dummy"
    "$ROOT/converted_model_roy_fullsize_32khz_backbone_mamba_ffn_dummy"
    "$ROOT/converted_model_roy_fullsize_32khz_backbone_attn_dummy"
)

NVML_FALLBACK="$ROOT/benchmark_results/easymagpie_dummy_parallel_32khz_20260914"
NVML_FALLBACK="$NVML_FALLBACK/nvml_fallback_v4"
export PYTHONPATH="$NVML_FALLBACK:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
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

start_server() {
    local variant="$1"
    local model="$2"

    setsid conda run -n easymagpie-vllm --no-capture-output \
        env EASYMAGPIE_DEPLOY_CONFIG="$DEPLOY_CONFIG" \
        bash "$ROOT/scripts/run_server.sh" "$model" "$PORT" \
        >"$RESULT_DIR/${variant}_server.log" 2>&1 &
    SERVER_PID=$!

    for _ in $(seq 1 900); do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" || true
            tail -160 "$RESULT_DIR/${variant}_server.log"
            exit 1
        fi
        sleep 2
    done
    tail -160 "$RESULT_DIR/${variant}_server.log"
    exit 1
}

run_benchmark() {
    local num_requests="$1"
    conda run -n easymagpie-vllm --no-capture-output \
        python "$ROOT/scripts/benchmark_server.py" \
        --text-file "$INPUT_FILE" \
        --num-requests "$num_requests" \
        --concurrency 1 32 \
        --url "http://127.0.0.1:$PORT" \
        --max-new-tokens 128 \
        --sample-rate 32000
}

trap stop_server EXIT INT TERM

{
    echo "started_at=$(date -Iseconds)"
    echo "git_commit=$(git -C "$ROOT/../.." rev-parse HEAD)"
    echo "git_diff_sha256=$(git -C "$ROOT/../.." diff --binary | sha256sum | cut -d' ' -f1)"
    implementation_hash=$(sha256sum \
        "$ROOT/easymagpie_vllm_omni/config.py" \
        "$ROOT/easymagpie_vllm_omni/easymagpie.py" \
        "$ROOT/easymagpie_vllm_omni/backbone_codebook.py" \
        "${MODELS[0]}/config.json" \
        "${MODELS[1]}/config.json" \
        "${MODELS[2]}/config.json" | sha256sum)
    implementation_hash="${implementation_hash%% *}"
    echo "implementation_sha256=$implementation_hash"
    echo "models=${MODELS[*]}"
    echo "variants=${VARIANTS[*]}"
    echo "prefix_pattern=MEMEM*EMEMEM*EME"
    echo "num_hidden_layers=32"
    echo "backbone_codebook_start_layer=16"
    echo "num_stacked_codebooks=16"
    echo "codebook_prediction_mode=backbone"
    echo "local_transformer_n_layers=0"
    echo "acoustic_sampling=top_k_gumbel"
    echo "acoustic_temperature=0.7"
    echo "acoustic_top_k=80"
    echo "deploy_config=$DEPLOY_CONFIG"
    echo "discarded_warmup_requests_per_level=32"
    echo "benchmark_builtin_warmup_requests=concurrency"
    echo "requests_per_run=128"
    echo "concurrency=1,32"
    echo "repetitions=5"
    echo "reported_run=4"
    echo "max_new_tokens=128"
    echo "sample_rate=32000"
    conda run -n easymagpie-vllm --no-capture-output python -c \
        "import importlib.metadata as m, torch; print(f'torch={torch.__version__}'); print(f'vllm={m.version(\"vllm\")}'); print(f'vllm_omni={m.version(\"vllm-omni\")}'); print(f'gpu={torch.cuda.get_device_name(0)}')"
} >"$RESULT_DIR/metadata.txt"

for idx in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$idx]}"
    model="${MODELS[$idx]}"
    echo "[$variant] starting server"
    start_server "$variant" "$model"

    echo "[$variant] discarded warmup"
    run_benchmark 32 >"$RESULT_DIR/${variant}_warmup.log" 2>&1

    for run in $(seq 1 5); do
        echo "[$variant] benchmark run $run/5"
        run_benchmark 128 2>&1 | tee "$RESULT_DIR/${variant}_run${run}.log"
    done

    stop_server
done

echo "completed_at=$(date -Iseconds)" >>"$RESULT_DIR/metadata.txt"
