import os
import sys
import time
from datetime import datetime
import numpy as np
import traceback
import torch
import torch.distributed as dist
import torch_xla.distributed.xla_backend
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

if not '..' in sys.path: sys.path.append('..')
import common.vision_utils as vision_utils

def train():
  print('==> Preparing data..')
  is_root = xm.is_master_ordinal(local=False)
  train_transform, test_transform = vision_utils.get_data_transforms(FLAGS.image_dim)
  if FLAGS.fake_data:
    train_dataset_len = FLAGS.fake_train_dataset_length
    test_dataset_len = FLAGS.fake_test_dataset_length
    train_dataset = datasets.FakeData(
      size=train_dataset_len,
      image_size=(3, FLAGS.image_dim, FLAGS.image_dim),
      num_classes=1000,
      transform=train_transform)
    test_dataset = datasets.FakeData(size=test_dataset_len,
      image_size=(3, FLAGS.image_dim, FLAGS.image_dim),
      num_classes=1000,
      transform=test_transform)
  elif FLAGS.data_dir:
    train_dataset = datasets.ImageFolder(
        os.path.join(FLAGS.data_dir, 'train'),
        train_transform)
    train_dataset_len = len(train_dataset.imgs)
    test_dataset = datasets.ImageFolder(
        os.path.join(FLAGS.data_dir, 'test'),
        test_transform)
  else:
    # use cifar10 by default
    train_transform, test_transform = vision_utils.get_data_transforms(FLAGS.image_dim)
    train_dataset = datasets.CIFAR10(
        os.path.join("./"), train=True, transform=train_transform, download=True
    )
    test_dataset = datasets.CIFAR10(
        os.path.join("./"), train=False, transform=test_transform, download=True
    )

  train_loader, test_loader = vision_utils.create_data_loaders(
    train_dataset,
    test_dataset,
    xm.get_ordinal(),
    xm.xrt_world_size(),
    FLAGS.batch_size,
    FLAGS.test_batch_size,
    FLAGS.num_workers,
    FLAGS.drop_last)

  torch.manual_seed(42)

  device = xm.xla_device()
  model = vision_utils.get_model(FLAGS.platform, FLAGS.model, FLAGS.pretrained).to(device)
  writer = None
  if xm.is_master_ordinal():
    logger = vision_utils.Logger(FLAGS, xm.xrt_world_size())
  optimizer = optim.SGD(
      model.parameters(),
      lr=FLAGS.lr,
      momentum=FLAGS.momentum,
      weight_decay=1e-4)
  lr_scheduler = None
  if FLAGS.lr_scheduler_type == "WarmupAndExponentialDecayScheduler":
    num_training_steps_per_epoch = train_dataset_len // (
        FLAGS.batch_size * xm.xrt_world_size())
    lr_scheduler = vision_utils.WarmupAndExponentialDecayScheduler(
        optimizer,
        num_training_steps_per_epoch,
        divide_every_n_epochs=getattr(FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
        divisor=getattr(FLAGS, 'lr_scheduler_divisor', None))
  loss_fn = nn.CrossEntropyLoss()

  if is_root:
    throughput = vision_utils.Throughput(FLAGS.batch_size, xm.xrt_world_size(), FLAGS.log_steps)
    print('--------TRAINING CONFIG----------')
    print(FLAGS)
    print('---------------------------------')
    train_start = time.time()

  def train_loop_fn(loader, epoch, global_step):
    model.train()
    for step, (data, target) in enumerate(loader):
      optimizer.zero_grad()
      output = model(data)
      logits = output if isinstance(output, torch.Tensor) else output.logits
      loss = loss_fn(logits, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      if lr_scheduler:
        lr_scheduler.step()
      global_step += 1
      if is_root:
        step_throughput = throughput.get_throughput()
        logger.train_throughputs.append(step_throughput)
        if step % FLAGS.log_steps == 0:
          logger.print_training_update(
            device,
            step,
            FLAGS.lr if not lr_scheduler else lr_scheduler.optimizer.param_groups[0]['lr'],
            loss.item(),
            step_throughput,
            epoch,
            writer)
      if global_step >= FLAGS.max_steps:
        xm.mark_step()
        break
    return global_step, loss

  def test_loop_fn(loader, epoch):
    total_samples, correct = 0, 0
    model.eval()
    with torch.no_grad():
      for step, (data, target) in enumerate(loader):
        output = model(data)
        logits = output if isinstance(output, torch.Tensor) else output.logits
        pred = logits.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
        total_samples += data.size()[0]
        if is_root:
          step_throughput = throughput.get_throughput()
          logger.test_throughputs.append(step_throughput)
          if step % FLAGS.log_steps == 0:
            logger.print_test_update(device, step_throughput, None, epoch, step)
    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  accuracy, max_accuracy = 0.0, 0.0
  global_step = 0
  for epoch in range(1, FLAGS.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, datetime.now()))
    global_step, loss = train_loop_fn(train_device_loader, epoch, global_step)
    xm.master_print('Epoch {} train end {}'.format(epoch, datetime.now()))
    if FLAGS.metrics_debug:
      xm.master_print(met.metrics_report())
    if is_root:
      average_train_throughput = round(sum(logger.train_throughputs)/len(logger.train_throughputs), 4)
      xm.master_print('Average train throughput: {:.4f}'.format(average_train_throughput))
      xm.master_print('Max train throughput: {:.4f}'.format(max(logger.train_throughputs)))
    if global_step >= FLAGS.max_steps:
      break
  
  if is_root:
    time_to_train = time.time() - train_start

  if FLAGS.do_eval:
    if is_root:
      throughput = vision_utils.Throughput(FLAGS.batch_size, xm.xrt_world_size(), FLAGS.log_steps)
    accuracy = test_loop_fn(test_device_loader, epoch)
    xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
        epoch, datetime.now(), accuracy))
    max_accuracy = max(accuracy, max_accuracy)
    if is_root:
      logger.write_to_summary(
          epoch,
          dict_to_write={'Accuracy/test': accuracy})
      average_test_throughput = round(sum(logger.test_throughputs)/len(logger.test_throughputs), 4)
      xm.master_print('Average test throughput: {:.4f}'.format(average_test_throughput))
      xm.master_print('Max test throughput: {:.4f}'.format(max(logger.test_throughputs)))
      xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))

def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  train()
  xm.rendezvous("_mp_fn finished")

if __name__ == '__main__':
  parser = vision_utils.build_train_parser()
  args = parser.parse_args(sys.argv[1:])
  
  if os.environ.get("WORLD_SIZE"):
    dist.init_process_group('xla')
    _mp_fn(0, args)
  else:
    xmp.spawn(_mp_fn, args=(args,))
