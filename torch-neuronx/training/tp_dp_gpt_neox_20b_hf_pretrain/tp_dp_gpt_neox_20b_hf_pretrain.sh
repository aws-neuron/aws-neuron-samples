#!/bin/bash

#############################################
# User defined parameters and env vars

export NEURON_CC_FLAGS="--model-type=transformer --enable-experimental-O1 --enable-saturate-infinity"
export NEURON_FUSE_SOFTMAX=1

export XLA_DOWNCAST_BF16=1
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1

export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# TP degree
TP_DEGREE=32
# global batch size
GBS=256
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=10000
# warmup steps
WARMUP_STEPS=100
# learning rate
LR=0.97e-5
# data path
DATA_PATH="~/examples_datasets/wikicorpus_gpt_neox_tokenized_2k"

#############################################

export NEURON_NUM_DEVICES=32
NODE_ID=0
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $NEURON_NUM_DEVICES"
if [ ! -z "$SLURM_NTASKS" ]; then
    WORLD_SIZE=$SLURM_NTASKS
    NODE_ID=$SLURM_NODEID
    MASTER_ADDRESS=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    DISTRIBUTED_ARGS="--nproc_per_node $NEURON_NUM_DEVICES --nnodes $WORLD_SIZE --node_rank $NODE_ID --master_addr $MASTER_ADDRESS --master_port 44000"
    if [ $NODE_ID -eq 0 ]; then
        echo "WORLD_SIZE=$WORLD_SIZE"
        echo "NODE_ID=$NODE_ID"
        echo "MASTER_ADDRESS=$MASTER_ADDRESS"
        echo "DISTRIBUTED_ARGS=$DISTRIBUTED_ARGS"
    fi
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
fi

#############################################

DP=$(($NEURON_NUM_DEVICES * $WORLD_SIZE / $TP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))

if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=6
    ACC_STEPS=6
    OUTPUT_LOG=log_compile-$NODE_ID.log
else
    STEPS_THIS_RUN=-1
    OUTPUT_LOG=log_exe-$NODE_ID.log
fi

if [ $NODE_ID -eq 0 ]; then
    echo TP_DEGREE=$TP_DEGREE
    echo GBS=$GBS
    echo MBS=$MBS
    echo TOTAL_STEPS=$TOTAL_STEPS
    echo WARMUP_STEPS=$WARMUP_STEPS
    echo LR=$LR
    echo DATA_PATH=$DATA_PATH

    echo DP=$DP
    echo ACC_STEPS=$ACC_STEPS
    echo STEPS_THIS_RUN=$STEPS_THIS_RUN
    echo OUTPUT_LOG=$OUTPUT_LOG
fi

torchrun $DISTRIBUTED_ARGS \
    tp_dp_gpt_neox_20b_hf_pretrain.py \
    --data_dir $DATA_PATH \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN \
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS |& tee $OUTPUT_LOG
