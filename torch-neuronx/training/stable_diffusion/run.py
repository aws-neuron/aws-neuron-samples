#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import shlex
import subprocess
import time

WORLD_SIZE=32

def parse_args():
    parser = argparse.ArgumentParser(
            prog='neuron-sd-training-test-wrapper',
            description='Test wrapper for Neuron Stable Diffusion training recipe')

    parser.add_argument('--model', choices=['2.1', '1.5'], default='2.1', help='Which model to train')
    parser.add_argument('--resolution', choices=[512], default=512, type=int, help='Which resolution of model to train')
    parser.add_argument('--batch_size', type=int, default=2, help='What per-device microbatch size to use')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='How many gradient accumulation steps to do (1 for no gradient accumulation)')
    parser.add_argument('--epochs', type=int, default=6, help='How many epochs to train for')

    # For saving checkpoints
    # Save every 750 steps ~= 1 epoch (at batch2) by default
    parser.add_argument("--checkpointing_steps", type=int, default=750,
        help=(
            "Save a checkpoint of the training state every X training steps. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--max_num_checkpoints", type=int, default=None,
        help=("Max number of checkpoints to store."),
    )

    # Used to save a copy of the trained model for inference
    parser.add_argument("--save_model_epochs", type=int, default=1,
        help=(
            "Save a copy of the trained model every X epochs in a format that can be loaded using HuggingFace's from_pretrained method."
        ))
    
    # For loading checkpoints
    # TODO: add ability to specify dir with checkpoints to restore from that is different than the default
    parser.add_argument('--resume_from_checkpoint', action="store_true", help="Resume from checkpoint at resume_step.")
    parser.add_argument('--resume_checkpoint_step', type=int, default=None, help="Which cumulative training step to resume from, looking for checkpoints in the script's work directory. Leave unset to use the latest checkpoint.")

    # Environment variable parameters
    # These are set to sensible defaults so don't need to be set by the caller unless the caller wants to experiment
    # with different values
    parser.add_argument('--neuron_num_recent_models_to_keep', type=int, default=5, help="The setting for the NEURON_NUM_RECENT_MODELS_TO_KEEP env var, which specifies the max number of NEFFs to have loaded at once.")
    parser.add_argument('--malloc_arena_max', type=int, default=32, help="The setting for the MALLOC_ARENA_MAX env var, which controls how many memory arenas glibc malloc will create / be able to allocate from.")
    parser.add_argument('--neuron_rt_async_exec_max_inflight_requests', type=int, default=1, help="The setting for the NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS env var, which specifies how many requests to the RT the framework is allowed to have inflight at once.")
    parser.add_argument('--neuron_rt_stochastic_rounding_seed', type=int, default=0, help="The setting for the NEURON_RT_STOCHASTIC_ROUNDING_SEED, which controls the seed for stochastic rounding.")

    # Path to dir containing the training and inference scripts
    parser.add_argument('--training_script_path', type=str, default="./sd_training_neuron.py", help="Path to the training script (sd_training_neuron.py)")

    args = parser.parse_args()

    assert args.training_script_path is not None, "Need to pass the path of the training script via --training_script_path (path to sd_training_neuron.py)"

    # Build the test name that will get used by the ModuleTester out of the args we parsed
    test_name = f"sd_{args.model}_training-{args.resolution}-batch{args.batch_size}-AdamW-{WORLD_SIZE}w-zero1_optimizer-grad_checkpointing"
    args.test_name = test_name

    return args

if __name__ == "__main__":
    args = parse_args()

    # Set environment variables that are needed
    # Model only fits at BF16
    os.environ["XLA_DOWNCAST_BF16"] = "1"
    # Fix the stochastic rounding seed so training is reproducible
    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_SEED"] = f"{args.neuron_rt_stochastic_rounding_seed}"
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = f"{args.neuron_rt_async_exec_max_inflight_requests}"
    # Prevents OOMs on long training runs
    os.environ["MALLOC_ARENA_MAX"] = f"{args.malloc_arena_max}"
    # Reduces HBM usage by unloading NEFFs that are only used once
    os.environ["NEURON_NUM_RECENT_MODELS_TO_KEEP"] = f"{args.neuron_num_recent_models_to_keep}"
    # Can help prevent GRPC errors from the XLA runtime e.g. during long compilations
    os.environ["TF_GRPC_DEFAULT_OPTIONS"] = "grpc.keepalive_time_ms=6000000,grpc.keepalive_timeout_ms=1440000000,grpc.http2.max_pings_without_data=999,grpc.http2.min_ping_interval_without_data_ms=300000000"
    # Explicitly unset XLA debug variables in case they are set. Can reduce performance if they are.
    os.environ.pop("XLA_IR_DEBUG", None)
    os.environ.pop("XLA_HLO_DEBUG", None)

    gradient_accumulation_steps = f"--gradient_accumulation_steps {args.gradient_accumulation_steps}" if args.gradient_accumulation_steps is not None else ""
    save_model_epochs = f"--save_model_epochs {args.save_model_epochs}" if args.save_model_epochs is not None else ""
    checkpointing_steps = f"--checkpointing_steps {args.checkpointing_steps}" if args.checkpointing_steps is not None else ""
    max_num_checkpoints = f"--max_num_checkpoints {args.max_num_checkpoints}" if args.max_num_checkpoints is not None else ""
    resume_from_checkpoint = f"--resume_from_checkpoint" if args.resume_from_checkpoint else ""
    resume_checkpoint_step = f"--resume_checkpoint_step {args.resume_checkpoint_step}" if args.resume_checkpoint_step is not None else ""

    # Only need to run for 1 epoch for NPC to do its thing
    run_command = f"torchrun --nproc_per_node={WORLD_SIZE} {args.training_script_path} --model {args.model} --resolution {args.resolution} {gradient_accumulation_steps} --batch_size {args.batch_size} {save_model_epochs} {checkpointing_steps} {max_num_checkpoints} {resume_from_checkpoint} {resume_checkpoint_step}"

    # We use 10 parallel jobs because we expect up to 9 graphs: 8 without grad accum enabled, 9 with it enabled
    neuron_parallel_compile_command = "neuron_parallel_compile --num_parallel 10 " + run_command + " --epochs 1"
    neuron_parallel_compile_command = shlex.split(neuron_parallel_compile_command)

    print(f"Starting compilation with neuron_parallel_compile. Command: {neuron_parallel_compile_command}")
    start_time = time.perf_counter()
    try:
        subprocess.run(neuron_parallel_compile_command, check=True)
    except Exception as e:
        print("ERROR: neuron_parallel_compile failed! Scroll up in this log for an explanation.")
        exit(1)
    end_time = time.perf_counter()
    print(f"Done compiling graphs! Total compile time is {end_time - start_time}s")

    run_training_command = run_command + f" --epochs {args.epochs}"
    run_training_command = shlex.split(run_training_command)

    print(f"Starting training. Command: {run_training_command}")
    start_time = time.perf_counter()
    try:
        subprocess.run(run_training_command, check=True)
    except Exception as e:
        print("ERROR: training failed! Scroll up in this log for an explanation.")
        exit(1)
    end_time = time.perf_counter()
    print(f"Done training! Total time to train: {end_time - start_time}s")
