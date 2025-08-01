#!/bin/bash
model_id=${1}
port=${2:-8000}
cores=${3:-0-31}
max_seq_len=${4:-2048}
cont_batch_size=${5:-32}
tp_size=${6:-32}
n_threads=${7:-32}

# Shift positional arguments out of the way before parsing named arguments
shift 7
set -x

# Default value for override_neuron_config
override_neuron_config="{}"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --speculative-model) draft_model_id="$2"; shift ;;
        --num-speculative-tokens) num_speculative_tokens="$2"; shift ;;
        --chat-template) chat_template="$2"; shift ;;
        --enable-chunked-prefill) enable_chunked_prefill="$2"; shift ;;
        --max-num-batched-tokens) max_num_batched_tokens="$2"; shift ;;
        --block-size) block_size="$2"; shift ;;
        --num-gpu-blocks-override) num_gpu_blocks_override="$2"; shift ;;
        --enable-prefix-caching) enable_prefix_caching="$2"; shift ;;
        --override-neuron-config) override_neuron_config="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;  # Handle unknown parameters
    esac
    shift  # Move to the next argument
done

# Build base command arguments
cmd_args=(
    --model "${model_id}"
    --tensor-parallel-size "${tp_size}"
    --max-num-seqs "${cont_batch_size}"
    --max-model-len "${max_seq_len}"
    --port "${port}"
    --device "neuron"
    --use-v2-block-manager
    --disable-log-requests
)

# Conditionally set the environment variable and add spec settings via override config
[ -n "$draft_model_id" ] && {
    echo "Setting draft model to: ${draft_model_id}"
    cmd_args+=(--speculative-max-model-len "${max_seq_len}")
    cmd_args+=(--speculative-model "${draft_model_id}")
    cmd_args+=(--num-speculative-tokens "${num_speculative_tokens}")
}

# Conditionally add chunked prefill settings via override config
[ -n "$enable_chunked_prefill" ] && {
    echo "Setting chunked prefill args"
    cmd_args+=(--enable-chunked-prefill "${enable_chunked_prefill}")
    cmd_args+=(--max-num-batched-tokens "${max_num_batched_tokens}")
    cmd_args+=(--block-size "${block_size}")
    cmd_args+=(--num-gpu-blocks-override "${num_gpu_blocks_override}")
}

# Conditionally add prefix caching related settings.
[ -n "$enable_prefix_caching" ] && {
    echo "Setting prefix caching args"
    cmd_args+=(--enable-prefix-caching )
    cmd_args+=(--block-size "${block_size}")
    cmd_args+=(--num-gpu-blocks-override "${num_gpu_blocks_override}")
}

cmd_args+=(--override-neuron-config "${override_neuron_config}")

[ -n "$chat_template" ] && cmd_args+=(--chat-template "${chat_template}")

echo "Starting VLLM Server for model: ${model_id}"

export NEURON_RT_INSPECT_ENABLE=0
export NEURON_RT_NUMERICAL_ERRORS_VERBOSITY=none
export XLA_HANDLE_SPECIAL_SCALAR=1
export UNSAFE_FP8FNCAST=1
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export VLLM_RPC_TIMEOUT=100000

# List the dependencies
echo "======================"
echo "Whl Dependencies:"
pip list | grep neuron
echo "======================"
echo "System Dependencies:"
dpkg -l | grep neuron
echo "======================"
echo "Environment Variables:"
env
echo "======================"

# Execute the command with all arguments
python3 -m vllm.entrypoints.openai.api_server "${cmd_args[@]}"