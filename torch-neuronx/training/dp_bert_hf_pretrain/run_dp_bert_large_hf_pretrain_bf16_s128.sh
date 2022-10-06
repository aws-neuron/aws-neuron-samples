#!/usr/bin/env bash
set -o pipefail

sudo modprobe -r neuron; sudo modprobe neuron

export NEURON_RT_EXEC_TIMEOUT=600
export TF_GRPC_DEFAULT_OPTIONS=grpc.keepalive_time_ms=60000,grpc.keepalive_timeout_ms=14400000,grpc.http2.max_pings_without_data=0,grpc.http2.min_ping_interval_without_data_ms=600000

INSTANCEID=`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id`
RANK_NODE=0
WORLD_SIZE_JOB=1
MAX_STEPS=28125
BATCH_SIZE=16
GRAD_ACCUM_USTEPS=32
NUM_NEURONCORES=32
OUTPUT_DIR=output
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"

if [ ! -z "$SLURM_NTASKS" ]; then
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
    export BUCKET_CAP_MB=512
    WORLD_SIZE_JOB=$SLURM_NTASKS
    RANK_NODE=$SLURM_NODEID
    MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    MASTER_PORT=2022
    GRAD_ACCUM_USTEPS=$(($GRAD_ACCUM_USTEPS/$WORLD_SIZE_JOB))
    OUTPUT_DIR=output_${RANK_NODE}_${WORLD_SIZE_JOB}
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE_JOB --node_rank $RANK_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    echo $DISTRIBUTED_ARGS
fi

HOST=`hostname`
echo "Hostname: $HOST (instance ID: $INSTANCEID)"

steps_this_run=$MAX_STEPS
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    steps_this_run=5
fi

mkdir -p $OUTPUT_DIR
json="$OUTPUT_DIR/results.json" && rm -f $json

sudo sysctl -w net.ipv4.ip_local_reserved_ports=48620 || exit 1
XLA_USE_BF16=1 torchrun $DISTRIBUTED_ARGS dp_bert_large_hf_pretrain_hdf5.py --output_dir $OUTPUT_DIR --steps_this_run $steps_this_run --metrics_file $json --batch_size=$BATCH_SIZE --grad_accum_usteps=$GRAD_ACCUM_USTEPS |& tee log_run_ph1_bf16_${RANK_NODE}_${WORLD_SIZE_JOB}

ret_val=$?
if [ $ret_val -eq 0 ]; then
    success=1
else
    success=0
fi

exit $ret_val
