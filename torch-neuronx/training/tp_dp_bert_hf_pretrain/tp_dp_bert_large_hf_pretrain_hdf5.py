# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

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

# This script is BERT pretraining script adapted from
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
# with XLA elements from https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py.
# Modifications done:
# - Port to Torch-XLA following Torch-XLA multi-process example
# - Add HuggingFace model/optimizer
# - Tweak arg names to be more intuitive (distringuish micro-steps from global steps)
# - Changed arg defaults
# - Added logger class to print log and also log to TensorBoard database

import os
import math
import torch
import glob
import h5py
import sys
import time
import argparse
import random
import json
import queue
import atexit
import traceback
from typing import Any, Dict, List
from datetime import datetime, timezone
from collections import deque, namedtuple
import torch_xla
import torch.nn as nn
import torch_xla.core.xla_model as xm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch.distributed as dist
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend
import numpy as np
from transformers import BertForPreTraining
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput
from transformers import (
    AdamW,
    set_seed,
)
from transformers.optimization import get_linear_schedule_with_warmup

try:
    from lamb import Lamb
except ImportError:
    print(
        "No LAMB optimizer implementation available to import, proceed with AdamW by default"
    )
import copy
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor
import inspect
import requests
import gc
from neuronx_distributed.parallel_layers import parallel_state, layers, grads, checkpointing, move_model_to_device

os.environ["NEURON_CC_FLAGS"] = (
    os.environ.get("NEURON_CC_FLAGS", "") + " --model-type=transformer"
)

# For PT autocast.
torch.cuda.is_bf16_supported = lambda: True

# Workaround for NaNs seen with transformers version >= 4.21.0
# https://github.com/aws-neuron/aws-neuron-sdk/issues/593
import transformers.modeling_utils as modeling_utils

if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16"):
    modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16

datetime_str = str(datetime.now())
results = {
    "inference_success": 1
}


Metric = namedtuple("Metric", ["name", "value", "units", "additional_data"])


class TrainingMetrics:
    def __init__(self, json_file):
        self.json_file = json_file

    def read_modify_write_file(self, data, key: str = "metrics") -> None:
        """
        data (dict of training parameters or list of metrics): Data to update in the file.
        key (str): the dictionary key under which data is to be recorded
        """
        result_dict = {}
        print(f"Writing data to the provided results file: {self.json_file}")
        if os.path.exists(self.json_file):
            with open(self.json_file) as json_file:
                result_dict = json.loads(json_file.read()) or result_dict
        print(f"Updating with {key} data: {data}")
        if result_dict:
            try:
                # handle internal named entity if present
                results = result_dict[next(iter(result_dict))]
            except Exception:
                results = result_dict
            current = results.get(key)
            if not current:
                results[key] = data
            else:
                if isinstance(current, list):
                    current.extend(data)
                elif isinstance(current, dict):
                    current.update(data)
        else:
            result_dict["results"] = {key: data}
        with open(self.json_file, "w") as json_file:
            json.dump(result_dict, json_file)

    def store_metrics(self, metrics: List[Metric]) -> None:
        """
        Writes collected metrics to the file.

        """
        data = [
            {
                "MetricName": metric.name,
                "MeasuredValue": metric.value,
                "Units": metric.units,
                "Timestamp": datetime.now(timezone.utc).isoformat(),
                "AdditionalData": metric.additional_data,
            }
            for metric in metrics
        ]
        self.update(data=data, key="metrics")

    def store_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Writes specified model and configuration parameters to the file.

        """
        self.update(data=parameters, key="parameters")

    def update(self, **kwargs: Any) -> None:
        """
        Write specified data to the output file.
        """
        self.read_modify_write_file(**kwargs)


class Throughput:
    def __init__(
        self, batch_size, world_size, grad_accum_usteps, moving_avg_window_size=10
    ):
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps
        self.moving_avg_window_size = moving_avg_window_size
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        throughput = window_size * self.seqs_per_iteration / self.window_time
        return throughput


class Logger:
    def __init__(self, args, world_size, model_dtype):
        xla = "torch_xla" in sys.modules
        self.throughputs = []
        dtype_short = model_dtype.replace("torch.", "")
        self.tb = SummaryWriter(
            os.path.join(
                args.output_dir,
                f"neuron_tblogs_{time.strftime('%m%d%y_%H%M')}"
                f"_{dtype_short}"
                f"_{args.optimizer}"
                f"_w{world_size}"
                f"_lr{args.lr}"
                f"_bs{args.batch_size}"
                f"_acc{args.grad_accum_usteps}"
                f"_warmup{args.warmup_steps}"
                f"_max{args.max_steps}"
                f"_debug{args.debug}"
                f"_bf16autocast{args.enable_pt_autocast}"
                f"_xla{xla}"
                f"_{self.get_instance_type()}",
            )
        )
        self.tb.add_text(
            "script", "```\n" + inspect.getsource(sys.modules[__name__]) + "\n```", 0
        )
        self.golden_steploss = []
        golden = "golden_steploss.txt"
        if os.path.exists(golden):
            with open(golden, "r") as f:
                self.golden_steploss = [float(i) for i in f]
            print(
                f"Read {len(self.golden_steploss)} golden step loss values from {golden}"
            )

    def get_instance_type(self):
        try:
            token = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            )
            data = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-type",
                headers={"X-aws-ec2-metadata-token": token.text},
            )
            return data.text
        except:
            return os.environ.get("HOSTNAME", "unknown")

    def log(self, epoch, step, step_loss, learning_rate, throughput, grad_norm=None):
        time_now = time.asctime()
        grad_norm_msg = f"grad-norm : {grad_norm}" if grad_norm else ""
        print(
            f"LOG {time_now} - ({epoch}, {step}) step_loss : {step_loss:.4f} "
            f"learning_rate : {learning_rate:.2e} throughput : {throughput:.2f} "
            f"{grad_norm_msg}",
            flush=True,
        )
        self.tb.add_scalar("step loss", step_loss, step)
        self.tb.add_scalar("learning rate", learning_rate, step)
        self.tb.add_scalar("throughput", throughput, step)
        if grad_norm:
            self.tb.add_scalar("grad-norm", grad_norm, step)
        self.throughputs.append(throughput)
        if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
            step_0start = step - 1
            if step_0start < len(self.golden_steploss) and step_0start >= 0:
                np.testing.assert_allclose(
                    step_loss, self.golden_steploss[step_0start], rtol=2.3e-1
                )


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        set_seed(self.seed)


def create_pretraining_dataset(
    input_file, max_pred_length, mini_batch_size, worker_init
):
    train_data = pretraining_dataset(
        input_file=input_file, max_pred_length=max_pred_length
    )
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=parallel_state.get_data_parallel_size(),
        rank=parallel_state.get_data_parallel_rank(),
    )
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=mini_batch_size,
        num_workers=0,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=True,
    )
    return train_dataloader, input_file


class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            torch.from_numpy(input[index].astype(np.int64))
            if indice < 5
            else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]

        # in torch.nn.NLLLoss, the default ignore-index is -100
        ignore_index = -100
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * ignore_index
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ]

    @property
    def sequence_length(self) -> int:
        """
        Returns the sequence length derived from the specified pre-tokenized dataset.

        """
        return len(self.inputs[0][0])


def get_model(flags):
    base_model = BertForPreTraining.from_pretrained("bert-large-uncased")
    # medium BERT size L12_A12_H768. Large BERT L24_A16_H1024 causes OOM on GPU V100
    my_config = copy.deepcopy(base_model.config)
    if flags.debug:
        my_config.num_hidden_layers = 1
        my_config.num_attention_heads = 2
        my_config.hidden_size = 16
    my_model = BertForPreTraining(my_config)
    def init_weights(weights):
        torch.nn.init.normal_(weights, mean=0.0, std=my_config.initializer_range)
    my_model.bert.embeddings.word_embeddings = layers.ParallelEmbedding(
        my_config.vocab_size,
        my_config.hidden_size,
        init_method=init_weights,
    )

    my_model.cls.predictions.decoder = layers.ColumnParallelLinear(
        my_config.hidden_size, my_config.vocab_size, bias=False)

    init_weights(my_model.cls.predictions.decoder.weight)

    class ParallelSelfAttention(BertSelfAttention):
        def __init__(self, config, position_embedding_type=None):
            super().__init__(config, position_embedding_type)
            self.query = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.key = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.value = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            init_weights(self.query.weight)
            init_weights(self.key.weight)
            init_weights(self.value.weight)
            with torch.no_grad():
                self.query.bias.zero_()
                self.key.bias.zero_()
                self.value.bias.zero_()
            self.num_attention_heads = self.num_attention_heads // parallel_state.get_tensor_model_parallel_size()
            self.all_head_size = self.all_head_size // parallel_state.get_tensor_model_parallel_size()

    class ParallelSelfOutput(BertSelfOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = layers.RowParallelLinear(config.hidden_size,
                                       config.hidden_size,
                                       input_is_parallel=True)
            init_weights(self.dense.weight)
            with torch.no_grad():
                self.dense.bias.zero_()
    
    for layer in my_model.bert.encoder.layer:
        layer.attention.self = ParallelSelfAttention(my_config)
        layer.attention.output = ParallelSelfOutput(my_config)

    return my_model


def get_dtype(model) -> str:
    """
    Reference: https://pytorch.org/xla/release/1.12/index.html#xla-tensors-and-bfloat16

    """
    if "XLA_USE_BF16" in os.environ:
        return "torch.bfloat16"
    if "XLA_DOWNCAST_BF16" in os.environ:
        if "torch.float" in str(model.dtype):
            return "torch.bfloat16"
        if "torch.double" in str(model.dtype):
            return "torch.float32"
    return str(model.dtype)    
    
def train_bert_hdf5(flags):
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=flags.tensor_parallel_size)
    rank = xm.get_ordinal()
    world_size = parallel_state.get_data_parallel_size()
    is_root = xm.is_master_ordinal(local=False)
    extract_graphs_only = os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None)
    set_seed(flags.seed)
    worker_init = WorkerInitObj(flags.seed)
    device = xm.xla_device()
    model = get_model(flags)
    move_model_to_device(model, device)
    model.train()
    model.tie_weights()
    # Additional tie needed
    # https://github.com/huggingface/transformers/blob/v4.12.0/src/transformers/models/bert/modeling_bert.py#L669
    model.cls.predictions.decoder.bias = model.cls.predictions.bias

    model_dtype = get_dtype(model)
    running_loss = torch.zeros(1, dtype=torch.double).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm"]  # gamma/beta are in LayerNorm.weight

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    assert flags.optimizer.lower() in [
        "adamw",
        "lamb",
    ], "optimizer input {} is invalid: make sure there the optimizer argument is valid: AdamW or LAMB".format(
        flags.optimizer
    )
    if flags.optimizer.lower() == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, flags.lr)
    elif flags.optimizer.lower() == "lamb":
        print(
            "Using LAMB with trust_ratio_clipping on, based on implementation from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lamb.py"
        )
        optimizer = Lamb(
            optimizer_grouped_parameters, flags.lr, trust_clip=True
        )  # default turning trust_clip on

    optimizer.zero_grad()

    if is_root:
        if not os.path.exists(flags.output_dir):
            os.makedirs(flags.output_dir, exist_ok=True)
        if not extract_graphs_only:
            logger = Logger(flags, world_size, model_dtype)
        metric_writer = TrainingMetrics(flags.metrics_file)
        throughput = Throughput(
            flags.batch_size, xm.xrt_world_size(), flags.grad_accum_usteps
        )
        print("--------TRAINING CONFIG----------")
        print(flags)
        print("--------MODEL CONFIG----------")
        print(model.config)
        print("---------------------------------")
        metric_writer.store_parameters(
            {
                "Model": model.name_or_path,
                "Model configuration": str(model.config),
                "World size": xm.xrt_world_size(),
                "Data parallel degree": world_size,
                "Batch size": flags.batch_size,
                "Total steps": flags.steps_this_run,
                "Seed": flags.seed,
                "Optimizer": str(optimizer),
                "Data type": model_dtype,
                "Gradient accumulation microsteps": flags.grad_accum_usteps,
                "Warmup steps": flags.warmup_steps,
                "Dataset": os.path.basename(os.path.normpath(flags.data_dir)),
                "Environment variables": {
                    variable: value
                    for variable, value in os.environ.items()
                    if variable.startswith("NEURON") or variable.startswith("XLA")
                },
            }
        )

    def train_loop_fn(
        model, optimizer, train_loader, epoch, global_step, training_ustep, running_loss
    ):
        max_grad_norm = 1.0

        for i, data in enumerate(train_loader):
            training_ustep += 1
            (
                input_ids,
                segment_ids,
                input_mask,
                masked_lm_labels,
                next_sentence_labels,
            ) = data
            outputs = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=masked_lm_labels,
                next_sentence_label=next_sentence_labels,
            )
            loss = outputs.loss / flags.grad_accum_usteps
            loss.backward()
            running_loss += loss.detach()

            if training_ustep % flags.grad_accum_usteps == 0:
                xm.mark_step()
                # loss averaging
                running_loss_div = running_loss / world_size
                running_loss_reduced = xm.all_reduce(
                    xm.REDUCE_SUM,
                    running_loss_div,
                    groups=parallel_state.get_data_parallel_group(as_list=True),
                )
                running_loss_reduced_detached = running_loss_reduced.detach()
                running_loss.zero_()
                # all-reduce and then clip. Order matters.
                xm.reduce_gradients(
                    optimizer, groups=parallel_state.get_data_parallel_group(as_list=True)
                )
                grads.clip_grad_norm(
                    model.parameters(), max_grad_norm
                )  # Gradient clipping is not in AdamW anymore
                optimizer.step()

                with torch.no_grad():
                    total_norm = torch.zeros(1, device=device)
                    if flags.print_grad_norm and is_root:
                        for p in model.parameters():
                            param_norm_sq = torch.square(p.grad).sum()
                            total_norm += param_norm_sq
                        total_norm = torch.sqrt(total_norm)

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                def _print_logs(running_loss_reduced_detached, total_norm):
                    if is_root and not extract_graphs_only:
                        total_norm_cpu = None
                        if flags.print_grad_norm:
                            total_norm_cpu = total_norm.cpu().item()
                        # NOTE: The running_loss is the loss of the global_step
                        logger.log(
                            epoch,
                            global_step,
                            running_loss_reduced_detached.cpu().item(),
                            optimizer.param_groups[0]["lr"],
                            throughput.get_throughput(),
                            total_norm_cpu,
                        )

                xm.add_step_closure(
                    _print_logs, (running_loss_reduced_detached, total_norm.detach())
                )
                if global_step >= flags.steps_this_run:
                    # NOTE: Prevent runtime "Call to recv failed : Broken pipe" issue
                    xm.mark_step()
                    break

        return (
            global_step,
            training_ustep,
            running_loss,
            running_loss_reduced_detached.cpu().item(),
        )

    scheduler_state_dict = None

    if flags.resume_ckpt:
        step = flags.resume_step
        state_dict = checkpointing.load(flags.output_dir, model)
        optimizer.load_state_dict(state_dict["optimizer"])
        global_step = state_dict["global_step"]
        epoch = state_dict["epoch"]
        scheduler_state_dict = state_dict["scheduler"]
    else:
        global_step = 0
        epoch = 0

    train_start = time.time()
    training_ustep = 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=flags.warmup_steps,
        num_training_steps=flags.max_steps,
        last_epoch=epoch if scheduler_state_dict else -1,
    )

    if scheduler_state_dict:
        scheduler.load_state_dict(scheduler_state_dict)

    thread_pool = ThreadPoolExecutor(1)

    assert os.path.exists(
        os.path.expanduser(flags.data_dir)
    ), "ERROR: Data directory {} doesn't exist!".format(flags.data_dir)
    while True:
        files = glob.glob(
            os.path.expanduser(flags.data_dir) + "/*_{}_*.hdf5".format("training")
        )
        files.sort()
        random.Random(epoch).shuffle(files)
        file_start_idx = 0

        num_files = len(files)
        assert (
            num_files > 0
        ), "ERROR: There are no tokenized dataset shard files in {}".format(
            flags.data_dir
        )
        assert (
            world_size <= num_files
        ), "ERROR: Please ensure there are at least {} (world_size) tokenized dataset shards in {} (currently I see only {}).".format(
            world_size, flags.data_dir, num_files
        )
        mini_batch_size = flags.batch_size
        # prep first iteration input data file
        data_file = files[(file_start_idx)]
        prev_file = data_file
        train_dataloader, _ = create_pretraining_dataset(
            data_file, flags.max_pred_len, mini_batch_size, worker_init
        )
        if flags.seq_len is not None:
            assert flags.seq_len == train_dataloader.dataset.sequence_length, (
                f"ERROR: User-specified sequence length ({flags.seq_len}) does not match "
                f"that of the pre-tokenized dataset ({train_dataloader.dataset.sequence_length})"
            )
        train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
        if is_root:
            metric_writer.store_parameters(
                {"Sequence length": train_dataloader.dataset.sequence_length}
            )

        # use DP dataloader
        for f in range(file_start_idx + 1, len(files)):
            # select data file to preload for the next iteration
            data_file = files[(f)]

            future_train_dataloader = thread_pool.submit(
                create_pretraining_dataset,
                data_file,
                flags.max_pred_len,
                mini_batch_size,
                worker_init,
            )
            xm.master_print(
                "Epoch {} file index {} begin {}".format(epoch, f, time.asctime()),
                flush=True,
            )
            print(f"Rank {rank} working on shard {prev_file}", flush=True)
            global_step, training_ustep, running_loss, final_loss = train_loop_fn(
                model,
                optimizer,
                train_device_loader,
                epoch,
                global_step,
                training_ustep,
                running_loss,
            )

            if is_root and not extract_graphs_only:
                final_time = time.time()
                time_diff = final_time - train_start
                print(
                    "Epoch {} step {} file index {} end {} loss {} perf {} seq/sec (at train microstep {} time {} from beginning time {})".format(
                        epoch,
                        global_step,
                        f,
                        time.asctime(),
                        final_loss,
                        logger.throughputs[-1],
                        training_ustep,
                        final_time,
                        train_start,
                    ),
                    flush=True,
                )
                additional_data = {
                    "Epoch": epoch,
                    "Global step": global_step,
                    "Microstep": training_ustep,
                    "File index": f,
                }
                metric_data = [
                    Metric("Loss", final_loss, "", additional_data),
                    Metric(
                        "Throughput", logger.throughputs[-1], "seq/s", additional_data
                    ),
                ]
                metric_writer.store_metrics(metric_data)

            if global_step >= flags.steps_this_run:
                if is_root and not extract_graphs_only:
                    # record aggregate & final statistics in the metrics file
                    additional_data = {
                        "Epoch": epoch,
                        "Global step": global_step,
                        "Microstep": training_ustep,
                    }
                    average_throughput = round(
                        sum(logger.throughputs) / len(logger.throughputs), 4
                    )
                    metric_data = [
                        Metric("Final loss", final_loss, "", additional_data),
                        Metric(
                            "Time to train",
                            round(time_diff / 60, 4),
                            "minutes",
                            additional_data,
                        ),
                        Metric(
                            "Average throughput",
                            average_throughput,
                            "seq/s",
                            additional_data,
                        ),
                        Metric(
                            "Peak throughput",
                            max(logger.throughputs),
                            "seq/s",
                            additional_data,
                        ),
                    ]
                    metric_writer.store_metrics(metric_data)
                state_dict = {
                    "model": model.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                checkpointing.save(state_dict, flags.output_dir)
                return
            del train_device_loader
            del train_dataloader
            gc.collect()
            train_dataloader, _ = future_train_dataloader.result(timeout=1000)
            train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
            prev_file = data_file

        epoch += 1


def _mp_fn(index, flags):
    torch.set_default_tensor_type("torch.FloatTensor")
    train_bert_hdf5(flags)
    xm.rendezvous("_mp_fn finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/examples_datasets/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128/",
        help="Pre-tokenized HDF5 dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="results.json",
        help="training metrics results file",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="choose optimizer type: (default) AdamW, LAMB",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Worker batch size.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=28125,
        help="Maximum total accumulation-steps to run.",
    )
    parser.add_argument(
        "--steps_this_run",
        type=int,
        default=128,
        help="Exit early at <value> steps and not go to max_steps. -1 to mean no early exit.",
    )
    parser.add_argument(
        "--shards_per_ckpt",
        type=int,
        default=1,
        help="Number of dataset shards before saving checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12349,
        help="Random seed. Worker seed is this value + worker rank.",
    )
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length; if specified, must match that of the pre-tokenized dataset, else derived from the dataset (via `--data_dir`)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode to help debug scripting."
    )
    parser.add_argument(
        "--max_pred_len",
        type=int,
        default=20,
        help="Maximum length of masked tokens in each input sequence.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Number of warmup accumulation-steps for learning rate .",
    )
    parser.add_argument(
        "--grad_accum_usteps",
        type=int,
        default=64,
        help="Gradient accumulation micro-steps (an accumulation-step has <value> micro-steps.",
    )
    parser.add_argument(
        "--enable_pt_autocast", action="store_true", help="Enable pytorch autocast."
    )
    parser.add_argument(
        "--print_grad_norm",
        default=False,
        action="store_true",
        help="Whether to print grad norm",
    )
    parser.add_argument(
        "--minimal_ckpt", 
        default=False, 
        action='store_true', 
        help="When specified, don't store optimizer/lr-schedule states in checkpoints."
    )
    parser.add_argument(
        "--test_checkpointing", 
        action='store_true', 
        help="When specified, validate save and load ccheckpoint"
    )
    parser.add_argument(
        "--resume_ckpt",
        action="store_true",
        help="Resume from checkpoint at resume_step."
    )
    parser.add_argument(
        "--resume_ckpt_path",
        help=
        "Checkpoint file to use rather than default. If not specified, then resume from last checkpoint or at resume_step (default file output/ckpt_<step>.pt)."
    )
    parser.add_argument(
        "--resume_step",
        default=-1,
        type=int,
        help="Step to resume training from."
    )
    parser.add_argument(
        "--tensor_parallel_size",
        default=2,
        type=int,
        help="Tensor parallel size"
    )
    
    args = parser.parse_args(sys.argv[1:])

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    if args.enable_pt_autocast:
        os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"

    # WORLD_SIZE is set by torchrun
    if os.environ.get("WORLD_SIZE"):
        dist.init_process_group("xla")
        _mp_fn(0, args)
    else:
        xmp.spawn(_mp_fn, args=(args,))