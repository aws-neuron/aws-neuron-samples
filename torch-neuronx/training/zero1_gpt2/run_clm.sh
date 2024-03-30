#!/bin/bash
set -o pipefail

sudo rmmod neuron; sudo modprobe neuron
sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620
sudo sysctl -w kernel.threads-max=10000000
ulimit -c unlimited

NUM_NEURONCORES=32
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"

LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
MALLOC_ARENA_MAX=64
echo "MALLOC_ARENA_MAX" $MALLOC_ARENA_MAX
echo "LD_PRELOAD" $LD_PRELOAD

if [ ! -z "$SLURM_NTASKS" ]; then
    # if running inside slurm, handle here
    MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    MASTER_PORT=2022
    WORLD_SIZE_JOB=$SLURM_NTASKS
    RANK_NODE=$SLURM_NODEID
    JOB_ID_TAG=job-"$SLURM_JOB_ID"
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE_JOB --node_rank $RANK_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    echo $DISTRIBUTED_ARGS
    export NEURON_RT_ROOT_COMM_ID=$MASTER_ADDR:46820
    export FI_EFA_FORK_SAFE=1
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
    echo "WORLD_SIZE_JOB=$WORLD_SIZE_JOB,  RANK_NODE=$RANK_NODE,  MASTER_ADDR_JOB=$MASTER_ADDR_JOB, NODE_LIST=$NODE_LIST"
    export TRANSFORMERS_CACHE=$HOME/hf_cache/`hostname`/hub
    export HF_DATASETS_CACHE=$HOME/hf_cache/`hostname`/datasets
fi

#Print Slurm Config
date;hostname;

export TRAINING_PRECISION=$1 #options FP32, BF16, MIXED
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1

if [[ "BF16" == $TRAINING_PRECISION ]]; then
    echo "USING BF16 ONLY"
    export XLA_USE_BF16=1
    export NEURON_CC_FLAGS="--retry_failed_compilation --distribution-strategy llm-training --model-type transformer"
elif [[ "MIXED" == $TRAINING_PRECISION ]]; then
    echo "USING MIXED PRECISION BF16 and FP32"
    export NEURON_CC_FLAGS="--retry_failed_compilation --enable-mixed-precision-accumulation --distribution-strategy llm-training --model-type transformer"
else
    echo "USING FP32 as default"
    export NEURON_CC_FLAGS="--retry_failed_compilation --distribution-strategy llm-training --model-type transformer"
fi

NEURON_CC_FLAGS+=" --cache_dir=$HOME/neuron_cache/gpt_1p5B/`hostname`"

export DISABLE_NUMERIC_CC_TOKEN=1
export NEURON_RT_HIERARCHICAL_CC=1

export NEURON_RT_EXEC_TIMEOUT=600
export TF_NUM_INTEROP_THREADS=8192

export NEURON_ENABLE_NOSEED_DROPOUT=1

GRAD_ACCUM_STEP=1
BATCH_SIZE=1
MODEL_CONFIG="config_1p5B_gpt2.json"
MODEL_SIZE=$(echo $CONFIG | grep -m 1 -Eo '[0-9MBp]+' | head -n1 | tr -d '\n')
DATASET_CONFIG=$2

if [ $GRAD_ACCUM_STEP -gt 1 ]; then
    echo "need to uncomment accelerator.py code to run"
    ./uncomment_gradaccum.sh
fi

MAX_STEPS=100000
LOG_FILE_NAME="run_log_hf_gpt2_param_"$MODEL_SIZE"_nodes"$WORLD_SIZE_JOB"_grad_accum"$GRAD_ACCUM_STEP"_bs"$BATCH_SIZE_$(date +"%m-%d-%Y")_$(date +"%H:%M:%S")
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    MAX_STEPS=10
    LOG_FILE_NAME="compile_log_hf_gpt2_param_"$MODEL_SIZE"_grad_accum"$GRAD_ACCUM_STEP"_bs"$BATCH_SIZE_$(date +"%m-%d-%Y")_$(date +"%H:%M:%S")
fi

torchrun $DISTRIBUTED_ARGS run_clm_no_trainer.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name $DATASET_CONFIG  \
    --config_name $MODEL_CONFIG \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEP \
    --max_train_steps $MAX_STEPS \
    --weight_decay 0.01 \
    --learning_rate 0.00015 \
    --lr_scheduler_type cosine \
    --use_zero1 \
    --gradient_checkpointing \
    --seed 1234 \
    --num_warmup_steps 75 \
    --use_grad_clipping \
    --validation_split_percentage 0 \
    --output_dir gpt_1p5B \
    |& tee $LOG_FILE_NAME
