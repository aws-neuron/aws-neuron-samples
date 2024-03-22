#!/usr/bin/env python
# coding=utf-8

################################################################################
###                                                                          ###
###                                 IMPORTS                                  ###
###                                                                          ###
################################################################################

# System
import gc
import os
import shutil
import sys
import pathlib
import random
from glob import glob
from typing import Union
    
# Neuron
import torch_xla.core.xla_model as xm
import torch_neuronx
    
# General ML stuff
import torch
import torch.nn.functional as functional
from torchvision import transforms
import numpy as np

# For measuring throughput
import queue
import time

# Model
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, Adafactor
# Needed for LoRA
from diffusers.loaders import AttnProcsLayers
# LR scheduler
from diffusers.optimization import get_scheduler

# Dataset
from datasets import load_dataset
# For logging and benchmarking
from datetime import datetime
import time

from diffusers import StableDiffusionPipeline

# Command line args
import argparse

# Multicore
import torch.multiprocessing as mp
import torch.distributed as dist
import torch_xla.distributed.xla_backend
import torch_xla.distributed.parallel_loader as xpl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data.distributed import DistributedSampler

import torch_xla.debug.profiler as xp

from torch_xla.amp.syncfree.adamw import AdamW
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer



################################################################################
###                                                                          ###
###                           CONSTANTS, ENV SETUP                           ###
###                                                                          ###
################################################################################

##### Neuron compiler flags #####
# --model-type=cnn-training: To enable various CNN training-specific optimizations, including mixed tiling algorithm and spill-free attention BIR kernel matching
# --enable-saturate-infinity: Needed for correctness. We get garbage data otherwise (probably from the CLIP text encoder)
# -O1: Gets us better compile time, especially when not splitting the model at the FAL level
compiler_flags = """ --retry_failed_compilation --cache_dir="./compiler_cache" --verbose=INFO -O1 --model-type=cnn-training  --enable-saturate-infinity """

os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

# Path to where this file is located
curr_dir = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(curr_dir)
    
image_column_name = "image"
caption_column_name = "text"

LOSS_FILE_FSTRING = "LOSSES-RANK-{RANK}.txt"


################################################################################
###                                                                          ###
###                           HELPER FUNCTIONS                               ###
###                                                                          ###
################################################################################
# For measuring throughput
class Throughput:
    def __init__(self, batch_size=8, data_parallel_degree=2, grad_accum_usteps=1, moving_avg_window_size=10):
        self.inputs_per_training_step = batch_size * data_parallel_degree * grad_accum_usteps
        self.moving_avg_window_size = moving_avg_window_size
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()

    # Record a training step - to be called anytime we call optimizer.step()
    def step(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        return
    
    # Returns the throughput measured over the last moving_avg_window_size steps
    def get_throughput(self):
        throughput = self.moving_avg_window.qsize() * self.inputs_per_training_step / self.window_time
        return throughput


# Patch ZeRO Bug - need to explicitly initialize the clip_value as the dtype we want
@torch.no_grad()
def _clip_grad_norm(
    self,
    max_norm: Union[float, int],
    norm_type: Union[float, int] = 2.0,
) -> torch.Tensor:
    """
    Clip all gradients at this point in time. The norm is computed over all
    gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = self._calc_grad_norm(norm_type)

    clip_coeff = torch.tensor(
        max_norm, device=self.device) / (
            total_norm + 1e-6)
    clip_value = torch.where(clip_coeff < 1, clip_coeff,
                                torch.tensor(1., dtype=clip_coeff.dtype, device=self.device))
    for param_group in self.base_optimizer.param_groups:
        for p in param_group['params']:
            if p.grad is not None:
                p.grad.detach().mul_(clip_value)

ZeroRedundancyOptimizer._clip_grad_norm = _clip_grad_norm


# Saves a pipeline to the specified dir using HuggingFace's built-in methods, suitable for loading
# as a pretrained model in an inference script
def save_pipeline(results_dir, model_id, unet, vae, text_encoder):
    xm.master_print(f"Saving trained model to dir {results_dir}")

    if xm.is_master_ordinal():
        assert not os.path.exists(results_dir), f"Error! Can't save checkpoint to {results_dir} because it already exists."
        os.makedirs(results_dir)

    if xm.is_master_ordinal():
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(results_dir)

    xm.master_print(f"Done saving trained model to dir {results_dir}")
    return


# Saves a checkpoint of the unet and optimizer to the directory specified
# If ZeRO-1 optimizer sharding is enabled, each ordinal needs to save its own checkpoint of the optimizer
def save_checkpoint(results_dir, unet, optimizer, epoch, step, cumulative_step):
    # Save UNet state - only the master needs to save as UNet state is identical between workers
    if xm.is_master_ordinal():
        checkpoint_path = os.path.join(results_dir, f"checkpoint-unet-epoch_{epoch}-step_{step}-cumulative_train_step_{cumulative_step}.pt")
        xm.master_print(f"Saving UNet state checkpoint to {checkpoint_path}")
        data = {
            'epoch': epoch,
            'step': step,
            'cumulative_train_step': cumulative_step,
            'unet_state_dict': unet.state_dict(),
        }
        # Copied from https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/dp_bert_hf_pretrain/dp_bert_large_hf_pretrain_hdf5.py
        # Not sure if it's strictly needed
        cpu_data = xm._maybe_convert_to_cpu(data)
        torch.save(cpu_data, checkpoint_path)
        del(cpu_data)
        xm.master_print(f"Done saving UNet state checkpoint to {checkpoint_path}")

    # Save optimizer state
    # Under ZeRO optimizer sharding each worker needs to save the optimizer state
    # as each has its own unique state
    checkpoint_path = os.path.join(results_dir, f"checkpoint-optimizer-epoch_{epoch}-step_{step}-cumulative_train_step_{cumulative_step}-rank_{xm.get_ordinal()}.pt")
    xm.master_print(f"Saving optimizer state checkpoint to {checkpoint_path} (other ranks will ahve each saved their own state checkpoint)")
    data = {
        'epoch': epoch,
        'step': step,
        'cumulative_train_step': cumulative_step,
        'optimizer_state_dict': optimizer.state_dict()
    }
    cpu_data = data
    # Intentionally don't move the data to CPU here - it causes XLA to crash
    # later when loading the optimizer checkpoint once the optimizer gets run
    torch.save(cpu_data, checkpoint_path)
    del(cpu_data)
    xm.master_print(f"Done saving optimizer state checkpoint to {checkpoint_path}")

    # Make the GC collect the CPU data we deleted so the memory actually gets freed
    gc.collect()
    xm.master_print("Done saving checkpoints!")


# Loads a checkpoint of the unet and optimizer from the directory specified
# If ZeRO-1 optimizer sharding is enabled, each ordinal needs to load its own checkpoint of the optimizer
# Returns a tuple of (epoch, step, cumulative_train_step)
def load_checkpoint(results_dir, unet, optimizer, device, resume_step):
    # Put an asterisk in for globbing if the user didn't specify a resume_step
    if resume_step is None:
        resume_step = "*"
    unet_checkpoint_filenames = glob(os.path.join(results_dir, f"checkpoint-unet-epoch_*-step_*-cumulative_train_step_{resume_step}.pt"))
    optimizer_checkpoint_filenames = glob(os.path.join(results_dir, f"checkpoint-optimizer-epoch_*-step_*-cumulative_train_step_{resume_step}-rank_{xm.get_ordinal()}.pt"))

    unet_checkpoint_filenames.sort()
    optimizer_checkpoint_filenames.sort()

    # Load UNet checkpoint
    checkpoint_path = unet_checkpoint_filenames[-1]
    xm.master_print(f"Loading UNet checkpoint from path {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    unet.load_state_dict(checkpoint['unet_state_dict'], strict=True)
    ret = (checkpoint['epoch'], checkpoint['step'], checkpoint['cumulative_train_step'])
    del(checkpoint)

    # Load optimizer checkpoint
    checkpoint_path = optimizer_checkpoint_filenames[-1]
    xm.master_print(f"Loading optimizer checkpoint from path {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(optimizer, torch.nn.Module):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=True)
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    assert checkpoint['epoch'] == ret[0] and checkpoint['step'] == ret[1] and checkpoint['cumulative_train_step'] == ret[2], \
        "UNet checkpoint and optimizer checkpoint do not agree on the epoch, step, or cumulative_train_step!"
    del(checkpoint)

    gc.collect()

    xm.master_print("Done loading checkpoints!")
    return ret


# Seed various RNG sources that need to be seeded to make training deterministic
# WARNING: calls xm.rendezvous() internally
def seed_rng(device):
    LOCAL_RANK = xm.get_ordinal()
    xm.rendezvous('start-seeding-cpu')
    torch.manual_seed(9999 + LOCAL_RANK)
    random.seed(9999+ LOCAL_RANK)
    np.random.seed(9999 + LOCAL_RANK)

    xm.rendezvous('start-seeding-device')
    xm.set_rng_state(9999 + LOCAL_RANK, device=device)
    # TODO: not sure if we need to print the RNG state on CPU to force seeding to actually happen
    xm.master_print(f"xla rand state after setting RNG state {xm.get_rng_state(device=device)}\n")
    xm.rendezvous('seeding-device-done')

    xm.master_print("Done seeding CPU and device RNG!")




################################################################################
###                                                                          ###
###                           MAIN TRAINING FUNCTION                         ###
###                                                                          ###
################################################################################

def train(args):
    LOCAL_RANK = xm.get_ordinal()

    # Create all the components of our model pipeline and training loop
    xm.master_print('Building training loop components')

    device = xm.xla_device()

    t = torch.tensor([0.1]).to(device=device)
    xm.mark_step()
    xm.master_print(f"Initialized device, t={t.to(device='cpu')}")

    # Warning: calls xm.rendezvous() internally
    seed_rng(device)

    if not xm.is_master_ordinal(): xm.rendezvous('prepare')

    model_id = args.model_id
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")    
    unet.requires_grad_(True)

    xm.master_print("Enabling gradient checkpointing")
    unet.enable_gradient_checkpointing()
    
    optim_params = unet.parameters()

    # IMPORTANT: need to move unet to device before we create the optimizer for the optimizer to be training the right parameters (on-device)
    unet.train()
    unet.to(device)
    
    # Setup VAE and text encoder
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder.to(device)

    vae.requires_grad_(False)
    vae.eval()
    # Needed for vae encoder to not downcast to bf16 with XLA_DOWNCAST_BF16
    for attn in vae.encoder.mid_block.attentions:
        # Intent of this is to upcast to fp32, but actual effect under XLA_DOWNCAST_BF16 is to force to bf16.
        attn.upcast_softmax = False
    # Set to float64 so that XLA_DOWNCAST_BF16 keeps as FP32
    vae.to(device=device, dtype=torch.float64)

    # TODO: parametrize optimizer parameters
    optimizer = ZeroRedundancyOptimizer(optim_params, AdamW, pin_layout=False, lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08, capturable=True, optimizer_dtype=torch.double)

    # Download the dataset
    xm.master_print('Downloading dataset')
    # TODO: make this a parameter of the script
    dataset_name = "m1guelpf/nouns"
    dataset = load_dataset(dataset_name)
    args.dataset_name = dataset_name

    # Done anything that might trigger a download
    xm.master_print("Executing `if xm.is_master_ordinal(): xm.rendezvous('prepare')`")
    if xm.is_master_ordinal(): xm.rendezvous('prepare')

    def training_metrics_closure(epoch, global_step, loss):
        loss_val = loss.detach().to('cpu').item()
        loss_f.write(f"{LOCAL_RANK} {epoch} {global_step} {loss_val}\n")
        loss_f.flush()

    xm.rendezvous('prepare-to-load-checkpoint')

    loss_filename = f"LOSSES-RANK-{LOCAL_RANK}.txt"

    if args.resume_from_checkpoint:
        start_epoch, start_step, cumulative_train_step = load_checkpoint(args.results_dir, unet, optimizer, device, args.resume_checkpoint_step)
        loss_f = open(loss_filename, 'a')
    else:
        start_epoch = 0
        start_step = 0
        cumulative_train_step = 0

        loss_f = open(loss_filename, 'w')
        loss_f.write("RANK EPOCH STEP LOSS\n")
    
    xm.rendezvous('done-loading-checkpoint')

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer
    )

    parameters = filter(lambda p: p.requires_grad, unet.parameters())
    parameters = sum([np.prod(p.size()) * p.element_size() for p in parameters]) / (1024 ** 2)
    xm.master_print('Trainable Parameters: %.3fMB' % parameters)
    total_parameters = 0
    components = [text_encoder, vae, unet]
    for component in components:
        total_parameters += sum([np.prod(p.size()) * p.element_size() for p in component.parameters()]) / (1024 ** 2)
    xm.master_print('Total parameters: %.3fMB' % total_parameters)
    
    # Preprocess the dataset
    column_names = dataset["train"].column_names
    if image_column_name not in column_names:
        raise ValueError(
            f"Did not find '{image_column_name}' in dataset's 'column_names'"
        )
    if caption_column_name not in column_names:
        raise ValueError(
            f"Did not find '{caption_column_name}' in dataset's 'column_names'"
        )
    
    resolution = args.resolution
    training_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(resolution),
            transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column_name]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column_name}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column_name]]
        examples["pixel_values"] = [training_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples
    
    train_dataset = dataset["train"].with_transform(preprocess_train)
    args.dataset_size = len(train_dataset)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        # Set to double so that bf16 autocast keeps it as fp32
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).double()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # Create dataloaders
    world_size = xm.xrt_world_size()
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=world_size,
                                           rank=xm.get_ordinal(),
                                           shuffle=True)

    # drop_last=True needed to avoid cases of an incomplete final batch, which would result in new graphs being cut and compiled
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False if train_sampler else True, collate_fn=collate_fn, batch_size=args.batch_size, sampler=train_sampler, drop_last=True
    )

    train_device_loader = xpl.MpDeviceLoader(train_dataloader, device, device_prefetch_size=2)
    
    xm.master_print('Entering training loop')
    xm.rendezvous('training-loop-start')

    found_inf = torch.tensor(0, dtype=torch.double, device=device)
    checkpoints_saved = 0

    # Use a moving average window size of 100 so we have a large sample at
    # the end of training
    throughput_helper = Throughput(args.batch_size, world_size, args.gradient_accumulation_steps, moving_avg_window_size=100)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.perf_counter_ns()
        before_batch_load_time = time.perf_counter_ns()
        xm.master_print("####################################")
        xm.master_print(f"###### Starting epoch {epoch} ######")
        xm.master_print("####################################")
        # Add 1 to the start_step so that we don't repeat the step we saved the checkpoint after
        for step, batch in enumerate(train_device_loader, start=(start_step + 1 if epoch == start_epoch else 0)):
            after_batch_load_time = time.perf_counter_ns()

            xm.master_print(f"*** Running epoch {epoch} step {step} (cumulative step {cumulative_train_step})")
            start_time = time.perf_counter_ns()
    
            # Convert input image to latent space and add noise
            with torch.no_grad():
                vae_inputs_batched = batch['pixel_values']
                vae_inputs_unbatched = torch.split(vae_inputs_batched, 1, dim=0)

                vae_outputs = []
                # Intentionally unroll the VAE execution here. Compiler produces poor QoR for the VAE at batch > 1
                for vae_input in vae_inputs_unbatched:
                    these_latents = vae.encode(vae_input).latent_dist.sample()
                    these_latents = these_latents * 0.18215
                    these_latents = these_latents.float()  # Cast latents to bf16 (under XLA_DOWNCAST_BF16)

                    vae_outputs.append(these_latents)
                latents = torch.cat(vae_outputs, dim=0)

                del vae_inputs_batched
                del vae_inputs_unbatched
                del vae_input
                del vae_outputs
                del these_latents

            gc.collect()

            # mark_step here to separate VAE into its own graph. Results in better compiler QoR.
            xm.mark_step()

            with torch.no_grad():
                noise = torch.randn(latents.size(), dtype=latents.dtype, layout=latents.layout, device='cpu')
                noise = noise.to(device=device)
                bsz = latents.shape[0]
        
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                timesteps = timesteps.to(device=device)
        
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
                # Run text encoder on caption
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            
                target = noise

            # UNet forward pass
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            # Calculate loss
            loss = functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
    
            # Add in extra mark_steps to split the model into FWD / BWD / optimizer - helps with compiler QoR and thus
            # model fit
            # TODO: parametrize how the script splits the model
            xm.mark_step()

            # Backwards pass
            loss.backward()

            xm.mark_step()


            with torch.no_grad():
                # Optimizer
                if (cumulative_train_step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step(found_inf=found_inf)
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    xm.master_print("Finished weight update")
                    throughput_helper.step()
                else:
                    xm.master_print("Accumulating gradients")

            xm.add_step_closure(training_metrics_closure, (epoch, step, loss.detach()), run_async=True)

            xm.mark_step()

            xm.master_print(f"*** Finished epoch {epoch} step {step} (cumulative step {cumulative_train_step})")
            e2e_time = time.perf_counter_ns()
            xm.master_print(f" > E2E for epoch {epoch} step {step} took {e2e_time - before_batch_load_time} ns")

            cumulative_train_step += 1

            # Checkpoint if needed
            if args.checkpointing_steps is not None and cumulative_train_step % args.checkpointing_steps == 0 and cumulative_train_step != 0:
                xm.rendezvous('prepare-to-save-checkpoint')
                save_checkpoint(args.results_dir, unet, optimizer, epoch, step, cumulative_train_step)
                checkpoints_saved += 1
                xm.rendezvous('done-saving-checkpoint')

            before_batch_load_time = time.perf_counter_ns()

            # Only need a handful of training steps for graph extraction. Cut it off so we don't take forever when
            # using a large dataset.
            if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) and cumulative_train_step > 5:
                break

        if args.save_model_epochs is not None and epoch % args.save_model_epochs == 0 and not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
            save_pipeline(args.results_dir + f"-EPOCH_{epoch}", args.model_id, unet, vae, text_encoder)
        
        end_epoch_time = time.perf_counter_ns()
        xm.master_print(f" Entire epoch {epoch} took {(end_epoch_time - start_epoch_time) / (10 ** 9)} s")
        xm.master_print(f" Given {step + 1} many steps, e2e per iteration is {(end_epoch_time - start_epoch_time) / (step + 1) / (10 ** 6)} ms")
        xm.master_print(f"!!! Finished epoch {epoch}")

        # Only need a handful of training steps for graph extraction. Cut it off so we don't take forever when
        # using a large dataset.
        if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) and cumulative_train_step > 5:
            break

    # Save the trained model for use in inference
    xm.rendezvous('finish-training')
    if xm.is_master_ordinal() and not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        save_pipeline(os.path.join(args.results_dir, "stable_diffusion_trained_model_neuron"), args.model_id, unet, vae, text_encoder)

    loss_f.close()

    xm.master_print(f"!!! Finished all epochs")
     
    # However, I may need to block here to await the async? How to do that???
    xm.wait_device_ops()

    xm.master_print(f"Average throughput over final 100 training steps was {throughput_helper.get_throughput()} images/s")

    xm.rendezvous('done')
    xm.master_print(f"!!! All done!")

    return




################################################################################
###                                                                          ###
###                             ARG PARSING, MAIN                            ###
###                                                                          ###
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='Neuron SD training script',
                    description='Stable Diffusion training script for Neuron Trn1')
    parser.add_argument('--model', choices=['2.1', '1.5'], help='Which model to train')
    parser.add_argument('--resolution', choices=[512, 768], type=int, help='Which resolution of model to train')
    parser.add_argument('--batch_size', type=int, help='What per-device microbatch size to use')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='How many gradient accumulation steps to do (1 for no gradient accumulation)')
    parser.add_argument('--epochs', type=int, default=2000, help='How many epochs to train for')

    # Arguments for checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=None,
        help=(
            "Save a checkpoint of the training state every X training steps. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--max_num_checkpoints", type=int, default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument("--save_model_epochs", type=int, default=None,
        help=(
            "Save a copy of the trained model every X epochs in a format that can be loaded using HuggingFace's from_pretrained method."
        ))

    # TODO: add ability to specify dir with checkpoints to restore from that is different than the default
    parser.add_argument('--resume_from_checkpoint', action="store_true", help="Resume from checkpoint at resume_step.")
    parser.add_argument('--resume_checkpoint_step', type=int, default=None, help="Which cumulative training step to resume from, looking for checkpoints in the script's work directory. Leave unset to use the latest checkpoint.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    env_world_size = os.environ.get("WORLD_SIZE")

    args = parse_args()

    # Lookup model name by model, resolution
    model_id_lookup = {
        "2.1": {
            512: "stabilityai/stable-diffusion-2-1-base",
        },
        "1.5": {
            512: "runwayml/stable-diffusion-v1-5"
        }
    }

    assert args.model in model_id_lookup.keys() and \
        args.resolution in model_id_lookup[args.model].keys(), \
        f"Error: model {args.model} at resolution {args.resolution} is not yet supported!"

    model_id = model_id_lookup[args.model][args.resolution]
    args.model_id = model_id

    test_name = f"sd_{args.model}_training-{args.resolution}-batch{args.batch_size}-AdamW-{env_world_size}w-zero1_optimizer-grad_checkpointing"

    # Directory to save artifacts to, like checkpoints
    results_dir = os.path.join(curr_dir, test_name + '_results')
    os.makedirs(results_dir, exist_ok=True)
    args.results_dir = results_dir

    dist.init_process_group('xla')
    world_size = xm.xrt_world_size()

    args.world_size = world_size

    assert int(world_size) == int(env_world_size), f"Error: world_size {world_size} does not match env_world_size {env_world_size}"

    xm.master_print(f"Starting Stable Diffusion training script on Neuron, training model {model_id} with the following configuration:")
    for k, v in vars(args).items():
        xm.master_print(f"{k}: {v}")
    xm.master_print(f"World size is {world_size}")
    xm.master_print("")
    xm.master_print(f"## Neuron RT flags ##")
    xm.master_print(f"NEURON_RT_STOCHASTIC_ROUNDING_SEED: {os.getenv('NEURON_RT_STOCHASTIC_ROUNDING_SEED', None)}")
    xm.master_print(f"NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS: {os.getenv('NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS', None)}")
    xm.master_print("")
    xm.master_print(f"## XLA flags ##")
    xm.master_print(f"XLA_IR_DEBUG: {os.getenv('XLA_IR_DEBUG', None)}")
    xm.master_print(f"XLA_HLO_DEBUG: {os.getenv('XLA_HLO_DEBUG', None)}")
    xm.master_print(f"XLA_DOWNCAST_BF16: {os.getenv('XLA_DOWNCAST_BF16', None)}")
    
    xm.rendezvous("Entering training function")

    train(args)

    xm.rendezvous("Done training")
