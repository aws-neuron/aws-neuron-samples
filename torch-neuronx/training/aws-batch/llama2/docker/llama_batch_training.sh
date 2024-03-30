#!/usr/bin/env bash
set -euo pipefail

python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ulimit -n 65535
sysctl -w net.ipv4.ip_local_reserved_ports=41000

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export CCOM_SOCKET_IFNAME=eth0

export MASTER_PORT=41000
export NODEID=$AWS_BATCH_JOB_NODE_INDEX
export NTASKS=$AWS_BATCH_JOB_NUM_NODES

export MALLOC_ARENA_MAX=64
export XLA_USE_BF16=1
export TF_NUM_INTEROP_THREADS=8192
export PROCESSES_PER_NODE=32
export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --cache_dir=$NEURON_COMPILE_CACHE_URI"
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export NUM_NEURONCORES=32

export NEURON_RT_NUM_CORES=32
export NUM_NEURONCORES=$NEURON_RT_NUM_CORES
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES
export NEURON_RT_ROOT_COMM_ID=localhost:48620

# TP degree
TP_DEGREE=8
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=1
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=1
# global batch size
GBS=1024
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=10000
# warmup steps
WARMUP_STEPS=100
# learning rate
LR=3.0e-4
# model path
MODEL_PATH=$SCRIPT_DIR
# data path
DATA_PATH="$HOME/examples_datasets/wikicorpus_llama2_7B_tokenized_4k"
# sequence length
SEQ_LEN=4096
# pre-compilation steps
PRE_COMPILATION_STEPS_COUNT=2
# training job steps
STEPS_THIS_RUN=-1
# output directory
OUTPUT_DIR="/llama_checkpoints"
# S3 checkpoint directory
CURRENT_BATCH_JOB_ID=$(echo "$AWS_BATCH_JOB_ID" | sed 's/#.*//')
CHECKPOINT_PATH="$CHECKPOINT_SAVE_URI$CURRENT_BATCH_JOB_ID"

if [ -v AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS ]
then
	export MASTER_ADDR=$AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS
else
	export MASTER_ADDR=`ip -f inet addr show eth0 | grep -Po 'inet \K[\d.]+'`
fi

DP=$(($NEURON_RT_NUM_CORES * $NTASKS / $TP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi

DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

TRAINING_ARGS="--model_path $MODEL_PATH --data_dir $DATA_PATH --tensor_parallel_size $TP_DEGREE --batch_size $MBS \
                --max_steps $TOTAL_STEPS --warmup_steps $WARMUP_STEPS --lr $LR --grad_accum_usteps $ACC_STEPS --seq_len $SEQ_LEN --sequence_parallel_enabled \
                --selective_checkpoint_enabled --logging_interval 10 --output_dir $OUTPUT_DIR $EXTRA_ARGS"

TORCH_RUN_COMMAND="torchrun $DISTRIBUTED_ARGS tp_zero1_llama2_7b_hf_pretrain.py $TRAINING_ARGS"

set
echo "Installing all dependencies..."
python3 -m pip install -r requirements.txt

# Downloading the pre-tokenized dataset from s3
echo "Downloading tokenized dataset..."
aws s3 cp $TOKENIZED_DATASET_URI $DATA_PATH --recursive --only-show-errors

# Running Pre-Compilation
if [ "$DO_PRE_COMPILATION" = true ]; then
  echo "Starting neuron parallel compilation..."
  neuron_parallel_compile $TORCH_RUN_COMMAND --steps_this_run $PRE_COMPILATION_STEPS_COUNT
fi

# Running Training Job
echo "Starting the training job..."
$TORCH_RUN_COMMAND --steps_this_run $STEPS_THIS_RUN

# Uploading checkpoints to S3
aws s3 cp $OUTPUT_DIR $CHECKPOINT_PATH --recursive --only-show-errors
echo "Saved the checkpoints to $CHECKPOINT_PATH"