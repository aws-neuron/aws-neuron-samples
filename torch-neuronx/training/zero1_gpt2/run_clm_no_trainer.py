# HF orginal code from https://raw.githubusercontent.com/huggingface/transformers/v4.26.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import torch_neuronx
import datasets
import torch
from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import queue
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository, create_repo
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    GPT2Config,
    GPT2Model,
    GPT2LMHeadModel,
    GPTNeoConfig,
    GPTNeoModel,
    GPTNeoForCausalLM
)

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

import time
import contextlib
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

import numpy as np
import torch_xla.utils.serialization as xser
import functools
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch_xla.distributed.xla_backend
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from neuron_utils import *
from accelerate.utils.imports import is_tpu_available

# Work around `Check failed: tensor_data`` error in torch-neuronx 2.1 when using torch.utils.data.DataLoader with shuffle=True
import copy
import torch_xla.core.xla_model as xm
def mesh_reduce(tag, data, reduce_fn):
    xm.rendezvous(tag)
    xdatain = copy.deepcopy(data)
    xdatain = xdatain.to("xla")
    xdata = xm.all_gather(xdatain, pin_layout=False)
    cpu_xdata = xdata.detach().to("cpu")
    cpu_xdata_split = torch.split(cpu_xdata, xdatain.shape[0])
    xldata = [x for x in cpu_xdata_split]
    xm.mark_step()
    return reduce_fn(xldata)
xm.mesh_reduce = mesh_reduce

# we need to use the torch_xla checkpoint. Otherwise the some checkpointing patterns will be eliminated by the compiler common expression elimination
torch.utils.checkpoint.checkpoint = torch_xla.utils.checkpoint.checkpoint

try:
    from utilities.reporting import Metric, post_metrics
except ImportError:
    Metric = post_metrics = lambda *args, **kwargs: None

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

os.environ['NEURON_RT_ONE_TMPBUF_PAGE_SIZE_MB']='2048'

# Uncomment below to keep only 2 subgraphs loaded at a time
os.environ['NEURON_NUM_RECENT_MODELS_TO_KEEP'] = '3' #4 will result in OOM
if os.environ.get("XLA_DOWNCAST_BF16") == '1':
    Bf16 = torch.finfo(torch.bfloat16)
    Fp32 = torch.finfo(torch.float32)
    torch.finfo = lambda a: Fp32 if a == torch.float64 else Bf16

if os.environ.get("XLA_USE_BF16") == '1':
    Bf16 = torch.finfo(torch.bfloat16)
    torch.finfo = lambda a:Bf16

# Load a custom model config json file
def load_model(
    model_type,
    config_path,
    precision=None,
    state_dict_path=None,
    model_config=None
):
    model_config = model_config if model_config else {}

    with open(config_path) as infile:
        loaded_config = json.load(infile)

    context = contextlib.nullcontext()

    with context:
        if 'neo' in model_type:
            config = GPTNeoConfig(
                **{**loaded_config, **model_config}
            )
            model = GPTNeoForCausalLM(config)
        else:
            config = GPT2Config(
                **{**loaded_config, **model_config}
            )
            model = GPT2LMHeadModel(config)

    if xm.is_master_ordinal(local=False):
        print('==========================================================================')
        print(f'TOTAL PARMS: {count_parameters(model)}')
        print(f'embedding wte total param: {count_parameters(model.transformer.wte)}')
        print(f'embedding wpe total param: {count_parameters(model.transformer.wpe)}')
        print(f'per layer total param: {count_parameters(model.transformer.h[0])}')
        print(f'lm_head total param:{count_parameters(model.lm_head)}')
        print('==========================================================================')
    return model

def get_param_norm(
    args,
    model,
    norm_type=2.0,
    groups=None,
) -> torch.Tensor:

    norm_type = float(norm_type)
    local_norm = torch.DoubleTensor([0.0]).to('xla')
    parameters = model.parameters()
    grads_for_norm = []
    for param in parameters:
        param_norm = torch.norm(param.detach(), norm_type)
        local_norm += param_norm ** norm_type

    if args.use_fsdp:
        total_norm = model.all_reduce_op(xm.REDUCE_SUM, local_norm, groups=groups)
        total_norm = total_norm**(1.0 / norm_type)
    elif args.use_zero1:
        total_norm = xm.all_reduce(xm.REDUCE_SUM, local_norm, groups=groups, pin_layout=False)
        total_norm = total_norm**(1.0 / norm_type)
    else:
        total_norm = local_norm**(1.0 / norm_type)
    #return total_norm.cpu().item()
    return total_norm

def get_grad_norm(
    args,
    model,
    norm_type=2.0,
    groups=None,
) -> torch.Tensor:

    norm_type = float(norm_type)
    local_norm = torch.FloatTensor([float(0.0)]).to('xla')
    parameters = model.parameters()
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        if grad_not_none:
            grad = param.grad.detach()
            grad_norm = torch.norm(grad, norm_type)
            local_norm += grad_norm ** norm_type

    if args.use_fsdp:
        #Gradients are scattered, so need to add all of them together
        total_norm = model.all_reduce_op(xm.REDUCE_SUM, local_norm, groups=groups)
        total_norm = total_norm**(1.0 / norm_type)
    elif args.use_zero1:
        total_norm = xm.all_reduce(xm.REDUCE_SUM, local_norm, groups=groups, pin_layout=False)
        total_norm = total_norm**(1.0 / norm_type)
    else:
        total_norm = local_norm**(1.0 / norm_type)
    #return total_norm.cpu().item()
    return total_norm


def training_metrics_closure(logger_metrics, epoch, global_step, loss, learning_rate, tp, grad_norm=None, param_norm=None):
    loss_val = loss.detach().to('cpu').item()
    grad_norm_val = grad_norm.detach().to('cpu').item() if grad_norm else None
    param_norm_val = param_norm.detach().to('cpu').item() if param_norm else None
    if logger_metrics != None:
        logger_metrics.log(epoch, global_step, loss_val, learning_rate, tp, grad_norm_val, param_norm_val, noisy_check=True)

def build_chkpt_path(output_dir, step, rank, world_size):
    chkpt_path = os.path.join(output_dir, f"step-{step}-rank-{rank}-of-{world_size}.ckpt")
    return chkpt_path

def main():

    torch.distributed.init_process_group('xla')
    device = xm.xla_device()
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()

    print(f'rank: {rank}, world size {world_size}')

    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        if args.use_mics:
            set_seed(args.seed, device_specific=True)
            # Do not need this, since device_specific=False in set_seed above
            seed_group_size = int(os.environ.get("NEURON_MICS_PARTITION_GROUP_SIZE", 32))
            seed = args.seed + rank % seed_group_size
            np.random.seed(seed=seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if is_tpu_available():
                xm.set_rng_state(seed)
        else:
            set_seed(args.seed, device_specific=False)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.load_tokenized_dataset is not None:
        if xm.is_master_ordinal(local=False):
            print("Loading tokenized dataset from ", args.load_tokenized_dataset)
        lm_datasets = load_from_disk(args.load_tokenized_dataset)
    elif args.dataset_name is not None and args.load_tokenized_dataset is None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            if args.dataset_name == "openwebtext":
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split="train"
                )
            else:
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

	# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.block_size is not None and args.block_size > tokenizer.model_max_length:
        tokenizer.model_max_length = args.block_size
        logger.warning(
            f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Setting tokenizer.model_max_length to {args.block_size}."
        )

    if args.model_name_or_path:
        model_config = {
            "vocab_size": 50304 if args.use_zero1 else len(tokenizer),  # zero1 not support padding
            "max_length": args.block_size,
            #"fused_scaled_masked_softmax": True, #args.fused_scaled_masked_softmax,
            #"fused_gelu": args.fused_gelu,
            "gradient_checkpointing": args.gradient_checkpointing,
            "use_cache": not args.gradient_checkpointing,
        }
        if args.use_zero1:
            model_config.update({
                "n_embd": 1536,  # zero1 not support padding
                "n_head": 24,  # zero1 not support padding
            })
        model = load_model(
            model_type=args.model_name_or_path,
            config_path=args.config_name,
            model_config=model_config,
        )
        # remove model = model.to('xla') before FSDP wrapper, so that the sharding will happen in CPU and only the sharded tensor will be sent to device
        # model = model.to('xla')

        model_dtype = get_dtype(model)
        extract_graphs_only = os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None)
        if xm.is_master_ordinal(local=False) and not extract_graphs_only:
            logger_metrics = Logger(args, world_size, model_dtype)
        else:
            logger_metrics = None

        # Moved here for FSDP because once we wrap with FSDP the weights will be sharded
        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        if args.use_fsdp:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=(GPTNeoBlock,) if 'neo' in args.model_name_or_path else (GPT2Block,)
            )
            fsdp_params = dict(flatten_parameters=False,
                               shard_param_on_dim_0=True,
                               optimization_barrier_in_forward=True,
                               optimization_barrier_in_backward=True,
                               reshard_after_forward=True,  # Save memory by keep all-gathers in bwd
                               disable_reshard_on_root=False,
                               coalesce_all_gather_ops=False,
                               auto_wrap_policy=auto_wrap_policy,
                               _debug_print=True, _debug_msg=f'Worker {rank}')
            if os.environ.get('TRAINING_PRECISION', default='') == 'MIXED':
                fsdp_params['compute_dtype'] = torch.bfloat16
            if args.use_mics:
                from torch_neuronx.distributed.fsdp_mics import XlaFullyShardedDataParallelMiCS as FSDPMiCS
                model = FSDPMiCS(model, **fsdp_params)
            else:
                model = FSDP(model, **fsdp_params)
            # Need to re-assign the shared module to use the correct FSDP wrapper
            # In BERT:
            # model.cls.predictions.decoder = model.bert.embeddings.word_embeddings
            # Here counter-part, but need to verify
            # model.lm_head = model.transformer.wte
            print(model)
        elif args.use_zero1:
            if model_dtype == "torch.float32":
                model = model.to(device='xla', dtype=torch.float32)
            elif model_dtype == "torch.bfloat16":
                model = model.to(device='xla', dtype=torch.bfloat16)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))


    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    if args.load_tokenized_dataset is None:
        # Preprocessing the datasets.
        # Frst we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
    else:
        block_size = args.block_size

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    if args.load_tokenized_dataset is None:
        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    if args.save_tokenized_dataset != None and xm.is_master_ordinal(local=False):
        lm_datasets.save_to_disk(args.save_tokenized_dataset)

    train_dataset = lm_datasets["train"]

    #openwebtext dataset does not have a validation split
    if args.dataset_name == "openwebtext":
        eval_dataset = lm_datasets["train"]
    else:
        eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=(os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) == None), collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_zero1:
        optimizer = ZeroRedundancyOptimizer(
            optimizer_grouped_parameters,
            torch.optim.AdamW,
            lr=args.learning_rate,
            grad_clipping=args.use_grad_clipping,
            max_norm=args.max_grad_norm,
            pin_layout=False,
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    if xm.is_master_ordinal(local=False):
        print(args)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if xm.is_master_ordinal(local=False):
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    throughput = Throughput(args.per_device_train_batch_size, world_size, args.gradient_accumulation_steps)
    starting_epoch = 0
    resume_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", "0") == "0":
                print("Resuming from checkpoint")
                resume_step = args.resume_step if args.resume_step is not None else 0
                ckpt_path = build_chkpt_path(args.resume_from_checkpoint, resume_step, rank, world_size)
                ckpt = xser.load(ckpt_path)
                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt["optimizer"])
                starting_epoch = ckpt["epoch"]
                del ckpt
                xm.rendezvous("Checkpoint loaded")
        else:
            raise ValueError(f"Please specify a checkpoint to resume from")

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    global_step = starting_epoch * num_update_steps_per_epoch

    optimizer.zero_grad()
    xm.mark_step()

    optimizer_step_done_at_least_once=0
    running_loss = torch.zeros(1, ).to(device)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and global_step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        lr_scheduler.step()
                        global_step += 1
                    continue
            if optimizer_step_done_at_least_once < 2:
                optimizer_step_done_at_least_once+=1
                if optimizer_step_done_at_least_once==2:
                    time.sleep(1)
                    xm.rendezvous("Init Complete")

            if args.use_fsdp and args.use_mics:
                param_norm = get_param_norm(args, model, groups=model.mics_sharding_cfg.partition_groups)
            elif args.use_zero1:
                param_norm = None
            else:
                param_norm = get_param_norm(args, model)

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps

            running_loss += loss.detach()

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss.backward()

            # This version, the accelerator.accumulate is removed
            if (step + 1) % args.gradient_accumulation_steps == 0:
                #if step == args.gradient_accumulation_steps and optimizer_step_done_at_least_once > 0:

                #    optimizer_step_done_at_least_once+=1
                xm.mark_step()
                running_loss_div = running_loss / world_size
                running_loss_reduced = xm.all_reduce(xm.REDUCE_SUM, running_loss_div.detach(), groups=None)
                running_loss.zero_()

                # Record grad norm
                if args.use_fsdp and args.use_mics:
                    model.mics_gradient_sync()
                    grad_norm = get_grad_norm(args, model, groups=model.mics_sharding_cfg.partition_groups)
                elif args.use_zero1:
                    grad_norm = None
                else:
                    grad_norm = get_grad_norm(args, model)

                # gradient norm clipping
                if args.use_grad_clipping and not args.use_zero1:
                    if args.use_fsdp:
                        if args.use_mics:
                            model.clip_grad_norm_(max_norm=1, groups=model.mics_sharding_cfg.partition_groups)
                        else:
                            model.clip_grad_norm_(max_norm=args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

                optimizer.optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()
                global_step+=1

                tp = throughput.get_throughput()
                if not extract_graphs_only:
                    xm.add_step_closure(training_metrics_closure, (logger_metrics, epoch, global_step, running_loss_reduced, optimizer.param_groups[0]['lr'], tp, grad_norm, param_norm),
                        run_async=False) #no data dependency with next mark_step

                if xm.is_master_ordinal(local=False) and not extract_graphs_only:
                    additional_data = {"Step": global_step}
                    metric_data = [
                        Metric("Throughput", tp, units="seq/s", additional=additional_data),
                    ]
                    test_params = {
                        "Number of warmup steps": args.num_warmup_steps,
                        "Learning rate": args.learning_rate,
                        "Weight decay": args.weight_decay,
                        "Dataset": args.dataset_name,
                        "Tokenizer": args.tokenizer_name,
                        "Dataset config name": args.dataset_config_name,
                        "Per device train batch size": args.per_device_train_batch_size,
                        "Gradient accumulation steps": args.gradient_accumulation_steps,
                    }
                    post_metrics(metric_data, parameters=test_params, enable_cloudwatch=False)

                progress_bar.update(1)

            if isinstance(checkpointing_steps, int):
                if global_step % checkpointing_steps == 0:
                    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", "0") == "0":
                        ckpt_path = build_chkpt_path(args.output_dir, global_step, rank, world_size)
                        ckpt = {
                            "model": model.state_dict(),
                            # also save "shard_metadata" for checkpoint consolidation later via
                            # `python3 -m torch_xla.distributed.fsdp.consolidate_sharded_ckpts`
                            "shard_metadata": model.get_shard_metadata() if isinstance(model, FSDP) else None,
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                        }
                        xser.save(ckpt, ckpt_path, master_only=False)
                        print(f"Checkpoint saved to {ckpt_path}", flush=True)
                        xm.rendezvous("Checkpoint saved")

            if global_step >= args.max_train_steps:
                if xm.is_master_ordinal(local=False) and not extract_graphs_only:
                    average_throughput = round(sum(logger_metrics.throughputs)/len(logger_metrics.throughputs), 4)
                    if not os.environ.get("FI_EFA_USE_DEVICE_RDMA", None) and world_size > 32:
                       # multi-node/4-nodes throughput check
                        assert(average_throughput >= 45), "Average throughput :{} is  below derived expected derived threshold: {}".format(average_throughput, str(45))
                    else:
                        # single node throughput check
                        assert(average_throughput >= 14), "Average throughput :{} is  below derived expected derived threshold: {}".format(average_throughput, str(14))
                break

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == "__main__":
    main()
