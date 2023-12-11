#!/usr/bin/env bash
set -o pipefail
ulimit -n 65535
sysctl -w net.ipv4.ip_local_reserved_ports=41000

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export CCOM_SOCKET_IFNAME=eth0

export PROCESSES_PER_NODE=32
if [ -v AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS ]
then
	export MASTER_ADDR=$AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS
else
	export MASTER_ADDR=`ip -f inet addr show eth0 | grep -Po 'inet \K[\d.]+'`
fi
export MASTER_PORT=41000
export NODEID=$AWS_BATCH_JOB_NODE_INDEX
export NTASKS=$AWS_BATCH_JOB_NUM_NODES

DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

export MALLOC_ARENA_MAX=128
export XLA_USE_BF16=1
export TF_NUM_INTEROP_THREADS=8192

set
echo "Starting the job..."
torchrun $DISTRIBUTED_ARGS allreduce.py