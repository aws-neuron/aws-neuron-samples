import sys
import argparse
import queue
import time
import os
import numpy as np
import requests
from torch.utils.tensorboard import SummaryWriter
import inspect
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
    GPT2LMHeadModel
)
import torch

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_mics",
        action="store_true",
        help="Use MiCS"
    )
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        help="Wrapping Model for FSDP"
    )
    parser.add_argument(
        "--use_zero1",
        action="store_true",
        help="Wrapping optimizer with ZeRO-1"
    )
    parser.add_argument(
        "--linear_layer_patch",
        action="store_true",
        help="a linear layer patch in xla"
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=1,
        help="Step to resume from. Used with resume_from_checkpoint",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="activation checkpointing"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--load_tokenized_dataset",
        type=str,
        default=None,
        help="Path to load tokenized dataset"
    )

    parser.add_argument(
        "--print_grad_norm",
        action="store_true",
        help="print grad_norm for convergence debug"
    )

    parser.add_argument(
        "--save_tokenized_dataset",
        type=str,
        default=None,
        help="Path to save tokenized dataset"
    )

    parser.add_argument(
        "--use_grad_clipping",
        action="store_true",
        help="Enable gradient norm clipping"
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (default 1.0).",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.load_tokenized_dataset is None and args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

args = parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Throughput:
    def __init__(self, batch_size, world_size, grad_accum_usteps, moving_avg_window_size=4): # our sample size is too small for 10, cut to 4
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
        xla = 'torch_xla' in sys.modules
        self.throughputs = []
        dtype_short = model_dtype.replace("torch.", "")
        self.tb = SummaryWriter(os.path.join(args.output_dir,
                                             f"neuron_GPT2_tblogs_{time.strftime('%m%d%y_%H%M')}"
                                             f"_{dtype_short}"
                                             f"_w{world_size}"
                                             f"_lr{args.learning_rate}"
                                             f"_bs{args.per_device_train_batch_size}"
                                             f"_acc{args.gradient_accumulation_steps}"
                                             f"_max{args.max_train_steps}"
                                             f"_xla{xla}"
                                             f"_fsdp{args.use_fsdp}"
                                             f"_mics{args.use_mics}"
                                             f"_zero1{args.use_zero1}"
                                             f"_{self.get_instance_type()}"))
        self.tb.add_text('script', "```\n" + inspect.getsource(sys.modules[__name__]) + "\n```", 0)
        self.golden_steploss = []
        self.pass_rate = 0
        golden="golden_steploss.txt"
        if os.path.exists(golden):
            with open(golden, "r") as f:
                self.golden_steploss = [float(i) for i in f]
            print(f"Read {len(self.golden_steploss)} golden step loss values from {golden}")

    def get_instance_type(self):
        try:
            token = requests.put("http://169.254.169.254/latest/api/token", headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"})
            data = requests.get("http://169.254.169.254/latest/meta-data/instance-type", headers={"X-aws-ec2-metadata-token": token.text})
            return data.text
        except:
            return os.environ.get("HOSTNAME", "unknown")

    def log(self, epoch, step, step_loss, learning_rate, throughput, grad_norm=None, param_norm=None, noisy_check = False, threshold = 0.99):

        time_now = time.asctime()
        grad_norm_msg = f'grad-norm : {grad_norm}' if grad_norm else ''
        param_norm_msg = f'param-norm : {param_norm}' if grad_norm else ''
        print(f'LOG {time_now} - ({epoch}, {step}) step_loss : {step_loss:.4f} '
            f'learning_rate : {learning_rate:.2e} throughput : {throughput:.2f} '
            f'{grad_norm_msg}',f'{param_norm_msg}', flush=True)
        self.tb.add_scalar('step loss', step_loss, step)
        self.tb.add_scalar('learning rate', learning_rate, step)
        self.tb.add_scalar('throughput', throughput, step)
        if grad_norm:
            self.tb.add_scalar('grad-norm', grad_norm, step)
        if param_norm:
            self.tb.add_scalar('param-norm', param_norm, step)

        self.throughputs.append(throughput)
        if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
            step_0start = step - 1
            if step_0start < len(self.golden_steploss) and step_0start >= 0:
                if not noisy_check:
                    np.testing.assert_allclose(step_loss, self.golden_steploss[step_0start], rtol=2.5e-1)
                else:
                    if np.allclose(step_loss, self.golden_steploss[step_0start], rtol=2.5e-1):
                        print('passed', flush = True)
                        self.pass_rate += 1
                    else:
                        print('failed', flush = True)
                if step_0start == len(self.golden_steploss) -1  and noisy_check:
                    print(f'noisy mode pass rate = {self.pass_rate/len(self.golden_steploss)}, threshold = {threshold}')
                    assert self.pass_rate/len(self.golden_steploss) > threshold


def get_dtype(model) -> str:
    """
    Reference: https://pytorch.org/xla/release/1.12/index.html#xla-tensors-and-bfloat16

    """
    if "TRAINING_PRECISION" in os.environ:
        if os.getenv('TRAINING_PRECISION') in ['MIXED', 'BF16']:
            return "torch.bfloat16"
        return "torch.float32"
    if "XLA_USE_BF16" in os.environ:
        return "torch.bfloat16"
    if "XLA_DOWNCAST_BF16" in os.environ:
        if "torch.float" in str(model.dtype):
            return "torch.bfloat16"
        if "torch.double" in str(model.dtype):
            return "torch.float32"
    return str(model.dtype)
