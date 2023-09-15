from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import DataLoader, Dataset
import torch_xla.distributed.parallel_loader as xpl
from transformers import Trainer, TrainingArguments


@dataclass
class TrnTrainingArguments(TrainingArguments):
    loader_prefetch_size: Optional[int] = field(
        default=8,
        metadata={"help": "The max capacity of the queue used by the thread which is reading samples from the loader."},
    )
    device_prefetch_size: Optional[int] = field(
        default=4,
        metadata={"help": "The max size of the per-device queues, where the worker threads deposit tensors which have already been sent to devices."},
    )
    host_to_device_transfer_threads: Optional[int] = field(
        default=1,
        metadata={"help": "The number of threads that work in parallel to transfer data from loader queue to device queue."},
    )
    @property
    def _no_sync_in_gradient_accumulation(self):
        return False


class TrnTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        train_loader = super().get_train_dataloader()
        kwargs = {
            "loader_prefetch_size": self.args.loader_prefetch_size,
            "device_prefetch_size": self.args.device_prefetch_size,
            "host_to_device_transfer_threads": self.args.host_to_device_transfer_threads
        }
        if isinstance(train_loader, xpl.MpDeviceLoader):
            train_loader._parallel_loader_kwargs = kwargs
        return train_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_loader = super().get_eval_dataloader(eval_dataset)
        kwargs = {
            "loader_prefetch_size": self.args.loader_prefetch_size,
            "device_prefetch_size": self.args.device_prefetch_size,
            "host_to_device_transfer_threads": self.args.host_to_device_transfer_threads
        }
        if isinstance(eval_loader, xpl.MpDeviceLoader):
            eval_loader._parallel_loader_kwargs = kwargs
        return eval_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        test_loader = super().get_eval_dataloader(test_dataset)
        kwargs = {
            "loader_prefetch_size": self.args.loader_prefetch_size,
            "device_prefetch_size": self.args.device_prefetch_size,
            "host_to_device_transfer_threads": self.args.host_to_device_transfer_threads
        }
        if isinstance(test_loader, xpl.MpDeviceLoader):
            test_loader._parallel_loader_kwargs = kwargs
        return test_loader