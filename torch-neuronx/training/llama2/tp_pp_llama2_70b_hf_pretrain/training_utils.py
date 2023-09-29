from transformers import default_data_collator
from torch.utils.data.dataloader import DataLoader
import datasets
from torch.utils.data import DistributedSampler
from transformers import set_seed
from lr import CosineAnnealing

def get_learning_rate_scheduler(optimizer, args, last_epoch=-1):
    lr_scheduler = CosineAnnealing(optimizer, max_steps=args.max_steps, min_lr=args.min_lr, warmup_steps=args.warmup_steps, constant_steps=args.constant_steps, last_epoch=last_epoch)
    return lr_scheduler

def get_param_groups_by_weight_decay(model):
    """Get param groups."""
    if hasattr(model, "local_named_parameters"):
        # Zero1 use the first param in opt to decide the device
        param_optimizer = list(model.local_named_parameters())
    else:
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
    return optimizer_grouped_parameters

def create_llama_pretraining_dataset(
    data_dir, mini_batch_size, seed, dp_size, dp_rank,
):
    #Workaround because python functions are not picklable
    class WorkerInitObj(object):
        def __init__(self, seed):
            self.seed = seed

        def __call__(self, id):
            set_seed(self.seed)
    worker_init = WorkerInitObj(seed)
    train_data = datasets.load_from_disk(data_dir)
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=False,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        train_data,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=mini_batch_size,
        num_workers=0,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=True,
    )
    return train_dataloader