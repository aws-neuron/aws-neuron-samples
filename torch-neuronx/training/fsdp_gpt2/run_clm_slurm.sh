#!/bin/bash

sudo rmmod neuron; sudo modprobe neuron
sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620
ulimit -c unlimited

MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
MASTER_PORT=2022
NUM_NEURONCORES=32

# if running inside slurm, handle here
WORLD_SIZE_JOB=$SLURM_NTASKS
RANK_NODE=$SLURM_NODEID
JOB_ID_TAG=job-"$SLURM_JOB_ID"
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE_JOB --node_rank $RANK_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS


export NEURON_RT_ROOT_COMM_ID=$MASTER_ADDR:46820
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa

# TODO : Is this needed ? enabled by default ? or disabled for debug only ?
# os.environ['NEURON_RT_STOCHASTIC_ROUNDING_SEED'] = '0'
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1

#Print Slurm Config
date;hostname;
#echo "WORLD_SIZE_JOB=$WORLD_SIZE_JOB,  RANK_NODE=$RANK_NODE,  MASTER_ADDR_JOB=$MASTER_ADDR_JOB, NODE_LIST=$NODE_LIST"


export TRAINING_PRECISION=$1 #options FP32, BF16, MIXED

if [[ "BF16" == $TRAINING_PRECISION ]]; then
    echo "USING BF16 ONLY"
    export XLA_USE_BF16=1
    # need to clean up later, as 
    # --distribution-strategy=FSDP
    export NEURON_CC_FLAGS="--retry_failed_compilation --disable-internal-data-race-checker --internal-dram-page-size=2048 --enable-experimental-spmd --internal-allreduce-buffer-size=135 --model-type transformer --tensorizer-options=\'--enable-real-yk-ccop\' --internal-max-instruction-limit=10000000 --enable-experimental-O1  --internal-layers-per-module 4 --internal-ccop-bucketing --internal-ccop-bucketing-allgather-size-in-bytes 1 --internal-ccop-bucketing-reducescatter-size-in-bytes 62481600 --internal-ccop-bucketing-allreduce-size-in-bytes 62481600 --internal-build-with-users"
elif [[ "MIXED" == $TRAINING_PRECISION ]]; then
    echo "USING MIXED PRECISION BF16 and FP32"
    export NEURON_CC_FLAGS="--distribution-strategy FSDP --enable-mixed-precision-accumulation --model-type transformer"
else
    echo "USING FP32 as default" 
    #export NEURON_CC_FLAGS="--retry_failed_compilation --disable-internal-data-race-checker --internal-dram-page-size=2048 --enable-experimental-spmd --internal-allreduce-buffer-size=135 --model-type transformer --tensorizer-options=\'--enable-real-yk-ccop\' --internal-max-instruction-limit=10000000 --enable-experimental-O1  --internal-layers-per-module 4 --internal-ccop-bucketing --internal-ccop-bucketing-allgather-size-in-bytes 1 --internal-ccop-bucketing-reducescatter-size-in-bytes 62481600 --internal-ccop-bucketing-allreduce-size-in-bytes 62481600 --internal-build-with-users"
    export NEURON_CC_FLAGS="--retry_failed_compilation --internal-dram-page-size=2048 --enable-experimental-spmd --model-type transformer --internal-max-instruction-limit=10000000 --enable-experimental-O1 --internal-build-with-users"
fi

export PARTITION_WORLD_SIZE=$NUM_NEURONCORES #TODO: remove with torchneuronx MICS
export DISABLE_NUMERIC_CC_TOKEN=1
export NEURON_RT_HIERARCHICAL_CC=1

#TODO : Change defaults for these to something more useful (eg the values in this script !)
#TODO : FAL automatically inserts --internal-dram-page-size if RT env var is set
#This needs to match compiler option --internal-dram-page-size if it is set
export NEURON_RT_ONE_TMPBUF_PAGE_SIZE_MB=2048 #Need to match
export NEURON_RT_EXEC_TIMEOUT=600
export TF_NUM_INTEROP_THREADS=8192

GRAD_ACCUM_STEP=1
BATCH_SIZE=1
MODEL_CONFIG="config_1p5B_gpt2.json"
#MODEL_CONFIG="config_125M_gpt2_small.json"
MODEL_SIZE=$(echo $CONFIG | grep -m 1 -Eo '[0-9MBp]+' | head -n1 | tr -d '\n')

if [ $GRAD_ACCUM_STEP -gt 1 ]; then
    echo "need to uncomment accelerator.py code to run"
    ./uncomment_gradaccum.sh
fi

MAX_STEPS=100000
LOG_FILE_NAME="run_log_hf_gpt2_param_"$MODEL_SIZE"_nodes"$WORLD_SIZE_JOB"_grad_accum"$GRAD_ACCUM_STEP"_bs"$BATCH_SIZE_$(date +"%m-%d-%Y")_$(date +"%H:%M:%S")
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    EPOCHS=1
    MAX_STEPS=10
    LOG_FILE_NAME="compile_log_hf_gpt2_param_"$MODEL_SIZE"_grad_accum"$GRAD_ACCUM_STEP"_bs"$BATCH_SIZE_$(date +"%m-%d-%Y")_$(date +"%H:%M:%S")
fi


torchrun $DISTRIBUTED_ARGS run_clm_no_trainer.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --config_name $MODEL_CONFIG \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEP \
    --max_train_steps $MAX_STEPS \
    --weight_decay 0.01 \
    --learning_rate 0.00015 \
    --lr_scheduler_type cosine \
    --use_fsdp \
    --gradient_checkpointing \
    --seed 1234 \
    --num_warmup_steps 75 \
    --use_grad_clipping \
    --validation_split_percentage 0 \
    --output_dir test-clm \
    |& tee $LOG_FILE_NAME
