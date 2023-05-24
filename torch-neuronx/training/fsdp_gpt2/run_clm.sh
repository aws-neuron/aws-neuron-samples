#! /bin/bash
set -o pipefail

ulimit -c unlimited 

NUM_NEURONCORES=32
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"

#export NEURON_RT_LOG_LEVEL_NRT=DEBUG
#export NEURON_RT_LOG_LOCATION=SYSLOG
export TRAINING_PRECISION=$1 #options FP32, BF16, MIXED

# TODO : Is this needed ? enabled by default ? or disabled for debug only ?
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1


if [[ "BF16" == $TRAINING_PRECISION ]]; then
    echo "USING BF16 ONLY"
    export XLA_USE_BF16=1
    export NEURON_CC_FLAGS="--retry_failed_compilation --distribution-strategy FSDP --model-type transformer"
elif [[ "MIXED" == $TRAINING_PRECISION ]]; then
    echo "USING MIXED PRECISION BF16 and FP32"
    export NEURON_CC_FLAGS="--retry_failed_compilation --distribution-strategy FSDP --enable-mixed-precision-accumulation --model-type transformer"
else
    echo "USING FP32 as default"
    export NEURON_CC_FLAGS="--retry_failed_compilation --distribution-strategy FSDP --model-type transformer"
fi

export PARTITION_WORLD_SIZE=$NEURON_NUM_DEVICES
export DISABLE_NUMERIC_CC_TOKEN=1
export NEURON_RT_HIERARCHICAL_CC=1

export NEURON_INTERNAL_FUSE_SOFTMAX=1
export NEURON_INTERNAL_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING=1

GRAD_ACCUM_STEP=1
BATCH_SIZE=1
MODEL_CONFIG="config_1p5B_gpt2.json"
MODEL_SIZE=$(echo $MODEL_CONFIG | grep -m 1 -Eo '[0-9MBp]+' | head -n1 | tr -d '\n')
DATASET_CONFIG=$2

if [ $GRAD_ACCUM_STEP -gt 1 ]; then
	echo "need to uncomment accelerator.py code to run"
	./uncomment_gradaccum.sh
fi


MAX_STEPS=100000
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    MAX_STEPS=10
fi

# example script to run:
# neuron_parallel_compile ./run_clm.sh MIXED, MIXED can be replaced by BF16, any other string start with FP32
# or  ./run_clm.sh MIXED
torchrun $DISTRIBUTED_ARGS run_clm_no_trainer.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name $DATASET_CONFIG \
    --config_name $MODEL_CONFIG \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEP \
    --max_train_steps $MAX_STEPS \
    --weight_decay 0.01 \
    --learning_rate 0.00015 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 75 \
    --use_fsdp \
    --seed 1234 \
    --gradient_checkpointing \
    --use_grad_clipping \
    --validation_split_percentage 0 \
    --output_dir test-clm \
    |& tee "run_log_hf_gpt2_param_"$MODEL_SIZE"_grad_accum"$GRAD_ACCUM_STEP"_bs"$BATCH_SIZE &
wait %1

ret_val=$?
if [ $ret_val -eq 0 ]; then
    success=1
else
    success=0
fi

dump_to_s3_update_json_scr=../../../dump_to_s3_update_test_json.sh
if [ -e $dump_to_s3_update_json_scr ]; then
    $dump_to_s3_update_json_scr $@ --key=inference_success --value=$success || echo "Unable to update test result JSON."
else
    echo "WARNING: Script $dump_to_s3_update_json_scr not found. Not updating test result JSON."
fi

exit $ret_val
