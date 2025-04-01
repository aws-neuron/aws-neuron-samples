#!/bin/bash

# Define default values
model=${1}
model_path=${2}
max_concurrent_req=${3:-1}
port=${4:-8000} 
task_name=${5:-"gsm8k_cot"}
results_dir=${6}
timeout=${7:-7200}
limit=${8:-200}
use_chat=${9:-true}

source ~/lm_eval_venv/bin/activate

echo "Running LM Eval Client for model: ${model}, model_path: ${model_path}, max_concurrent_req: ${max_concurrent_req}, port: ${port}, task_name: ${task_name}, results_dir: ${results_dir}, timeout: ${timeout}, limit: ${limit}, use_chat: ${use_chat}"

set -x 

export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE="http://localhost:${port}/v1"

# Set the endpoint based on use_chat
if [ "$use_chat" = true ] ; then
    endpoint="chat/completions"
    model_type="local-chat-completions"
    additional_args="--apply_chat_template"
    echo "Starting lm_eval with chat completions"
else
    endpoint="completions"
    model_type="local-completions"
    additional_args=""
    echo "Starting lm_eval without chat completions"
fi

# Common arguments with dynamic endpoint
common_args=(
    "--tasks ${task_name}"
    "--model_args model=${model_path},base_url=http://localhost:${port}/v1/${endpoint},tokenized_requests=False,tokenizer_backend=None,num_concurrent=${max_concurrent_req},timeout=${timeout}"
    "--log_samples"
    "--output_path ${results_dir}"
    "--limit ${limit}"
)

# Execute the command
python -m lm_eval \
    --model ${model_type} \
    ${common_args[@]} \
    ${additional_args}