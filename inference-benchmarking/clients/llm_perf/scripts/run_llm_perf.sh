set -x

model=${1}
max_concurrent_req=${2:-1}
mean_ip_tokens=${3}
stddev_ip_tokens=${4}
mean_op_tokens=${5}
stddev_op_tokens=${6}
results_dir=${7}
n_batches=${8}
port=${9}
client_type=""
dataset=""
tokenizer=""

shift 9
while [[ $# -gt 0 ]]; do
    case $1 in
        --client-type) client_type="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --tokenizer) tokenizer="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

cd ~
source llm_perf_venv/bin/activate

# Supported client types
LLM_PERF="llm_perf"
LLM_PERF_GITHUB_PATCHED="llm_perf_github_patched"
SUPPORTED_CLIENTS=("$LLM_PERF" "$LLM_PERF_GITHUB_PATCHED")
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE="http://localhost:${port}/v1"
max_requests=$(expr ${max_concurrent_req} \* ${n_batches} )


if [ -z "${client_type}" ]; then
  client_type="$LLM_PERF"  # Default to llm_perf
fi

# Validate client type
if [[ ! " ${SUPPORTED_CLIENTS[*]} " =~ " ${client_type} " ]]; then
    echo "Error: Invalid client type '${client_type}'."
    echo "Valid options are: ${SUPPORTED_CLIENTS[*]}"
    exit 1
fi

greedy_sampling_parameters='{"top_k": 1, "top_p": 1.0, "temperature": 0.0}'
greedy_sampling_parameters=( "$greedy_sampling_parameters" )

release_sampling_parameters='{"top_k": -1, "top_p": 1.0, "temperature": 0.7, "ignore_eos": "True"}'
release_sampling_parameters=( "$release_sampling_parameters" )

# Use model as tokenizer if tokenizer is not provided
if [ -z "${tokenizer}" ]; then
    tokenizer="${model}"
fi


# Common arguments for both scenarios
common_args="--model ${model} \
       --mean-input-tokens ${mean_ip_tokens} \
       --stddev-input-tokens ${stddev_ip_tokens} \
       --mean-output-tokens ${mean_op_tokens} \
       --stddev-output-tokens ${stddev_op_tokens} \
       --max-num-completed-requests ${max_requests} \
       --timeout 1720000 \
       --num-concurrent-requests ${max_concurrent_req} \
       --results-dir ${results_dir} \
       --llm-api openai"


   
if [ "$client_type" == "$LLM_PERF_GITHUB_PATCHED" ]; then
    echo "Starting github patched token_benchmark_ray.py"
    common_args+=" --tokenizer ${tokenizer}"
    python "${LLMPERF_INSTALL_DIR}/token_benchmark_ray.py" ${common_args} --additional-sampling-params "${release_sampling_parameters[@]}"
else
     echo "Starting github token_benchmark_ray.py"
    python ./llmperf/token_benchmark_ray.py ${common_args} --additional-sampling-params "${release_sampling_parameters[@]}"
fi
