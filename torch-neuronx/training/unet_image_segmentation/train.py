import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import torch_xla.distributed.xla_backend
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
from torchvision import transforms as T
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

if not '..' in sys.path: sys.path.append('..')
import common.vision_utils as vision_utils
from model import UNet

class CarvanaDataset(Dataset):
    def __init__(self, img_ids, data_dir="./"):
        self.img_ids = img_ids
        self.data_dir = data_dir

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.data_dir + "/train/" + img_id + ".jpg"
        mask_path = self.data_dir + "/train_masks/" + img_id + "_mask.gif"
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_transforms = T.Compose([ T.Resize((FLAGS.image_dim, FLAGS.image_dim)), T.ToTensor(), normalize ])
        mask_transforms = T.Compose([ T.Resize((FLAGS.image_dim, FLAGS.image_dim)), T.ToTensor() ])
        image = image_transforms(image)
        mask = mask_transforms(mask)
        return image, mask

def calc_accuracy(output, target):
    #dice_score
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    score = (2. * (output * target).sum()) / (output + target).sum()
    return torch.mean(score).item()

def train():
    print('==> Preparing data..')
    is_root = xm.is_master_ordinal(local=False)
    images = glob.glob(FLAGS.data_dir + "train/*")
    images = sorted(images)
    images = [s.split("/")[-1].split(".")[0] for s in images]
    train_images, test_images = train_test_split(images, test_size=0.1, random_state=42)
    train_dataset = CarvanaDataset(train_images, FLAGS.data_dir)
    test_dataset = CarvanaDataset(test_images, FLAGS.data_dir)

    if is_root:
        print(f"train_dataset : {len(train_dataset)}, test_dataset : {len(test_dataset)}")
        img, mask = train_dataset[0]
        print(f"Image shape : {np.shape(img)}, mask shape : {np.shape(mask)}")

    train_loader, test_loader = vision_utils.create_data_loaders(
        train_dataset,
        test_dataset,
        xr.global_ordinal(),
        xr.world_size(),
        FLAGS.batch_size,
        FLAGS.test_batch_size,
        FLAGS.num_workers,
        FLAGS.drop_last)

    torch.manual_seed(42)

    device = xm.xla_device()
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    writer = None
    if xm.is_master_ordinal():
        logger = vision_utils.Logger(FLAGS, xr.world_size())
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)
    loss_fn = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    if is_root:
        throughput = vision_utils.Throughput(FLAGS.batch_size, xr.world_size(), FLAGS.log_steps)
        train_start = time.time()

    def train_loop_fn(loader, epoch, global_step):
        model.train()
        for step, (data, target) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)
            global_step += 1
            if is_root:
                step_throughput = throughput.get_throughput()
                logger.train_throughputs.append(step_throughput)
                if step % FLAGS.log_steps == 0:
                    logger.print_training_update(
                    device,
                    step,
                    FLAGS.lr,
                    loss.item(),
                    step_throughput,
                    epoch,
                    writer)
            if global_step >= FLAGS.max_steps:
                xm.mark_step()
                break
        return global_step, loss

    def test_loop_fn(loader, epoch):
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for step, (data, target) in enumerate(loader):
                output = model(data)
                # dice score
                accuracy += calc_accuracy(output, target)
                if is_root:
                    step_throughput = throughput.get_throughput()
                    logger.test_throughputs.append(step_throughput)
                    if step % FLAGS.log_steps == 0:
                        logger.print_test_update(device, step_throughput, None, epoch, step)
        accuracy = accuracy / max(len(loader), 1)
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
        if is_root:
            average_train_throughput = round(sum(logger.train_throughputs)/len(logger.train_throughputs), 4)
            xm.master_print('Average train throughput: {:.4f}'.format(average_train_throughput))
            xm.master_print('Max train throughput: {:.4f}'.format(max(logger.train_throughputs)))
        if global_step >= FLAGS.max_steps:
            break

    if is_root:
        time_to_train = time.time() - train_start
        xm.master_print("TrainLoss: {:.4f}".format(loss.item()))
        xm.master_print("TrainRuntime {} minutes".format(round(time_to_train/60, 4)))

    if FLAGS.do_eval:
        if is_root:
            throughput = vision_utils.Throughput(FLAGS.batch_size, xr.world_size(), FLAGS.log_steps)
        accuracy = test_loop_fn(test_device_loader, epoch)
        xm.master_print('Epoch {} test end {}, Accuracy(Dice_score)={:.2f}'.format(
            epoch, datetime.now(), accuracy))
        max_accuracy = max(accuracy, max_accuracy)
        if is_root:
            logger.write_to_summary(
                epoch,
                dict_to_write={'Accuracy/test': accuracy})
            average_test_throughput = round(sum(logger.test_throughputs)/len(logger.test_throughputs), 4)
            xm.master_print('Average test throughput: {:.4f}'.format(average_test_throughput))
            xm.master_print('Max test throughput: {:.4f}'.format(max(logger.test_throughputs)))
            xm.master_print('Accuracy(Dice_score): {:.2f}'.format(max_accuracy))


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
