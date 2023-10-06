import os
import sys
import argparse
from datetime import datetime
import math
import queue
import time
import inspect
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import (
  AutoModelForImageClassification,
  AutoConfig
)
import timm

SUPPORTED_PLATFORMS = ['torchvision', 'transformers', 'timm']

class WarmupAndExponentialDecayScheduler(_LRScheduler):
  """Update the learning rate of wrapped optimizer based on epoch and step.

  Args:
    optimizer: Instance of torch.optim.Optimizer. Learning rate will be changed.
    num_steps_per_epoch: int, the number of steps required to finish 1 epoch.
    divide_every_n_epochs: After this number of epochs, learning rate will be
      divided by the `divisor` param.
    divisor: The learning rate will be divided by this amount when epoch %
      divide_every_n_epochs == 0 (epoch 0 is excluded).
    num_warmup_epochs: Float. Learning rate will ramp up from 0 to max learning
      rate over this many epochs. Note that partial epochs are allowed, e.g. 0.5
      epochs.
    min_delta_to_update_lr: If the new learning rate does not differ much from
      the learning rate of the previous step, don't bother updating the
      optimizer's learning rate.
  """

  def __init__(self,
               optimizer,
               num_steps_per_epoch,
               divide_every_n_epochs=20,
               divisor=5,
               num_warmup_epochs=0.9,
               min_delta_to_update_lr=1e-6):
    self._num_steps_per_epoch = num_steps_per_epoch
    self._divide_every_n_epochs = divide_every_n_epochs
    self._divisor = divisor
    self._num_warmup_epochs = num_warmup_epochs
    self._min_delta_to_update_lr = min_delta_to_update_lr
    self._previous_lr = -1
    self._max_lr = optimizer.param_groups[0]['lr']
    super(WarmupAndExponentialDecayScheduler, self).__init__(optimizer)

  def _epoch(self):
    return self._step_count // self._num_steps_per_epoch

  def _is_warmup_epoch(self):
    return self._epoch() < math.ceil(self._num_warmup_epochs)

  def get_lr(self):
    epoch = self._epoch()
    lr = 0.0

    if self._is_warmup_epoch():
      # Ramp up learning rate from 0.0 to self._max_lr using a linear slope.
      num_warmup_steps = self._num_warmup_epochs * self._num_steps_per_epoch
      lr = min(self._max_lr,
               self._max_lr * ((self._step_count + 1.0) / num_warmup_steps))
    else:
      # Normal epoch. Use an exponential decay determined by init params.
      lr = self._max_lr / (
          self._divisor**(epoch // self._divide_every_n_epochs))

    # _LRScheduler expects a list of learning rates like this.
    return [lr for _ in self.base_lrs]

  def step(self, epoch=None):
    current_lr = self.get_lr()[0]

    # Outside of warmup epochs, we use the same learning rate for every step
    # in an epoch. Don't bother updating learning rate if it hasn't changed.
    if abs(current_lr - self._previous_lr) >= self._min_delta_to_update_lr:
      super(WarmupAndExponentialDecayScheduler, self).step()
      self._previous_lr = current_lr
    else:
      self._step_count += 1  # This normally happens in super().step().


class Throughput:
  def __init__(self, batch_size, world_size, log_steps, moving_avg_window_size=10):
    self.seqs_per_iteration = batch_size * world_size
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
  def __init__(self, args, world_size):
        xla = 'torch_xla' in sys.modules
        self.train_throughputs = []
        self.test_throughputs = []
        self.summary_writer = SummaryWriter(os.path.join(args.logdir,
                                             f"neuron_tblogs_{time.strftime('%m%d%y_%H%M')}"
                                             f"_w{world_size}"
                                             f"_lr{args.lr}"
                                             f"_bs{args.batch_size}"
                                             f"_bf16autocast{args.enable_pt_autocast}"
                                             f"_xla{xla}"))
        self.summary_writer.add_text('script', "```\n" + inspect.getsource(sys.modules[__name__]) + "\n```", 0)

  def print_training_update(self,
                          device,
                          step,
                          lr,
                          loss,
                          throughput,
                          epoch=None,
                          summary_writer=None):
    """Prints the training metrics at a given step.

    Args:
      device (torch.device): The device where these statistics came from.
      step_num (int): Current step number.
      loss (float): Current loss.
      throughput (float): The examples/sec throughput for the current batch.
      epoch (int, optional): The epoch number.
      summary_writer (SummaryWriter, optional): If provided, this method will
        write some of the provided statistics to Tensorboard.
    """
    update_data = [
        'Training', 'Device={}'.format(str(device)),
        'Epoch={}'.format(epoch) if epoch is not None else None,
        'Step={}'.format(step), 'Learning_Rate={}'.format(lr),
        'Loss={:.5f}'.format(loss), 'Throughput={:.5f}'.format(throughput),
        'Time={}'.format(datetime.now())
    ]
    print('|', ' '.join(item for item in update_data if item), flush=True)
    self.write_to_summary(
        summary_writer,
        dict_to_write={
            'Throughput': throughput,
        })

  def print_test_update(self, device, throughput, accuracy, epoch=None, step=None):
    """Prints single-core test metrics.

    Args:
      device: Instance of `torch.device`.
      accuracy: Float.
    """
    update_data = [
        'Test', 'Device={}'.format(str(device)),
        'Step={}'.format(step) if step is not None else None,
        'Epoch={}'.format(epoch) if epoch is not None else None,
        'Throughput={:.5f}'.format(throughput),
        'Accuracy={:.2f}'.format(accuracy) if accuracy is not None else None,
        'Time={}'.format(datetime.now())
    ]
    print('|', ' '.join(item for item in update_data if item), flush=True)

  def write_to_summary(self,
                      global_step=None,
                      dict_to_write={}):
    """Writes scalars to a Tensorboard SummaryWriter.

    Optionally writes XLA perf metrics.

    Args:
      global_step (int, optional): The global step value for these data points.
        If None, global_step will not be set for this datapoint.
      dict_to_write (dict, optional): Dict where key is the scalar name and value
        is the scalar value to be written to Tensorboard.
    """
    if self.summary_writer is None:
      return
    for k, v in dict_to_write.items():
      self.summary_writer.add_scalar(k, v, global_step)


def build_train_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='resnet50', help="Image classification model.")
  parser.add_argument('--platform', default='torchvision', choices=SUPPORTED_PLATFORMS, help="The Platform where the model is from (torchvision/transformers/timm).")
  parser.add_argument('--pretrained', action='store_true', help="Use model from Pre-trained.")
  parser.add_argument('--data_dir', type=str, default="/home/ubuntu/examples_datasets/imagenet", help="Image classification dataset directory.")
  parser.add_argument('--logdir', type=str, default="log_training", help="Training log directory.")
  parser.add_argument('--batch_size', type=int, default=8, help="Batch size per core used in training.")
  parser.add_argument('--num_epochs', type=int, default=2, help="Number of training epochs.")
  parser.add_argument('--num_workers', type=int, default=0, help="Number of worker used in data loader.")
  parser.add_argument('--log_steps', type=int, default=20, help="Number of steps between each other log message.")
  parser.add_argument('--max_steps', type=int, default=28125, help="Number of max training steps.")
  parser.add_argument('--expected_average_throughput', type=int, default=0, help="Expected average training throughput (seq/s).")
  parser.add_argument('--image_dim', type=int, default=224, help="Image dimension after transformation.")
  parser.add_argument('--test_batch_size', type=int, default=8, help="Batch size per core used in testing.")
  parser.add_argument('--lr', type=float, default=0.00005, help="Learning rate used in training.")
  parser.add_argument('--lr_scheduler_type', type=str, default=None, choices=["WarmupAndExponentialDecayScheduler",])
  parser.add_argument('--lr_scheduler_divide_every_n_epochs', type=int, default=20)
  parser.add_argument('--lr_scheduler_divisor', type=int, default=5)
  parser.add_argument('--momentum', type=float, default=0.9, help="Momentum used in SGD optimizer")
  parser.add_argument('--target_accuracy', type=float, default=0, help="Target accuracy (%).")
  parser.add_argument('--drop_last', action='store_true', help="Enable deop_last in data loader.")
  parser.add_argument('--fake_data', action='store_true', help="Use fake (random) data for training and testing.")
  parser.add_argument('--fake_train_dataset_length', type=int, default=50000, help="Length of fake training dataset.")
  parser.add_argument('--fake_test_dataset_length', type=int, default=1000, help="Length of fake testing dataset.")
  parser.add_argument('--metrics_debug', action='store_true', help="Print debug metrics at the end of each epoch.")
  parser.add_argument('--enable_pt_autocast', action='store_true', help="Enable Auto-cast to BF16 in GPU.")
  parser.add_argument('--do_eval', action='store_true', help="Evaluate the model with eval dataset after training.")

  return parser

def get_model(platform, model, pretrained):
  if platform == "torchvision":
    default_model_property = {
        'model_fn': getattr(torchvision.models, model)
    }
    model_properties = {
        'inception_v3': {
            'model_fn': lambda: torchvision.models.inception_v3(aux_logits=False)
        },
    }
    return model_properties.get(model, default_model_property)['model_fn'](pretrained=pretrained)
  elif platform == "transformers":
    if pretrained:
      return AutoModelForImageClassification.from_pretrained(model)
    else:
      config = AutoConfig.from_pretrained(model)
      return AutoModelForImageClassification.from_config(config)
  elif platform == "timm":
    return timm.create_model(model, pretrained=pretrained)
  else:
    raise ValueError('Unsupported Platform.')

def get_data_transforms(img_dim):
  resize_dim = max(img_dim, 256)
  normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
  # Matches Torchvision's eval transforms except Torchvision uses size
  # 256 resize for all models both here and in the train loader. Their
  # version crashes during training on 299x299 images, e.g. inception.
  test_transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.CenterCrop(img_dim),
            transforms.ToTensor(),
            normalize,
        ])

  return train_transform, test_transform

def create_data_loaders(train_dataset,
                        test_dataset,
                        rank,
                        world_size,
                        train_batch_size,
                        test_batch_size,
                        num_workers,
                        drop_last=False):
  train_sampler, test_sampler = None, None
  if world_size > 1:
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True)
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False)

  train_loader = DataLoader(
      train_dataset,
      batch_size=train_batch_size,
      shuffle=False if train_sampler else True,
      sampler=train_sampler,
      drop_last=drop_last,
      num_workers=num_workers,
      pin_memory=True)
  test_loader = DataLoader(
      test_dataset,
      batch_size=test_batch_size,
      shuffle=False,
      sampler=test_sampler,
      drop_last=drop_last,
      num_workers=num_workers,
      pin_memory=True)

  return train_loader, test_loader
