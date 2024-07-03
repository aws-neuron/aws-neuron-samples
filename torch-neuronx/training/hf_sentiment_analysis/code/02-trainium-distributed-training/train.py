import csv
from datasets import Dataset, DatasetDict
import logging
import os
import pandas as pd
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(0)

model_name = "bert-base-cased"
## define xla as device for using AWS Trainium Neuron Cores
device = "xla"

torch.distributed.init_process_group(device)

# Get the global number of workes.
world_size = xm.xrt_world_size()
logger.info("Workers: {}".format(world_size))

batch_size = 8
num_epochs = 6

logger.info("Device: {}".format(device))

## tokenize_and_encode
# params:
#   data: DatasetDict
# This method returns a dictionary of input_ids, token_type_ids, attention_mask
def tokenize_and_encode(data):
    results = tokenizer(data["text"], padding="max_length", truncation=True)
    return results

if __name__ == '__main__':
    path = os.path.abspath("data")
    csv_path = path + "/train.csv"

    train = pd.read_csv(
        csv_path,
        sep=',',
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        encoding='utf-8'
    )

    train_dataset = Dataset.from_dict(train)

    hg_dataset = DatasetDict({"train": train_dataset})

    ## Loading Hugging Face AutoTokenizer for the defined model
    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)

    ds_encoded = hg_dataset.map(tokenize_and_encode, batched=True, remove_columns=["text"])

    ds_encoded.set_format("torch")

    ## Create a subsed of data sampler, for parallelizing the training across multiple cores
    if world_size > 1:
        train_sampler = DistributedSampler(
            ds_encoded["train"],
            num_replicas=world_size,
            rank=xm.get_ordinal(),
            shuffle=True,
        )

    ## Creating a DataLoader object for iterating over it during the training epochs
    train_dl = DataLoader(
        ds_encoded["train"],
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True)

    ## Loading a subset of the data in the different Neuron Cores provided as input
    train_device_loader = pl.MpDeviceLoader(train_dl, device)

    ## Loading Hugging Face pre-trained model for sequence classification for the defined model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, force_download=True).to(device)

    current_timestamp = strftime("%Y-%m-%d-%H-%M", gmtime())

    optimizer = AdamW(model.parameters(), lr=1.45e-4 * world_size)

    num_training_steps = num_epochs * len(train_dl)
    progress_bar = tqdm(range(num_training_steps))

    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    ## Start model training and defining the training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_device_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            ## xm.optimizer_step is performing the sum of all the gradients updates done in the different Cores
            xm.optimizer_step(optimizer)
            progress_bar.update(1)

        logger.info("Epoch {}, rank {}, Loss {:0.4f}".format(epoch, xm.get_ordinal(), loss.detach().to("cpu")))

    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    ## Using XLA for saving model after training for being sure only one copy of the model is saved
    os.makedirs("./../../models/checkpoints/{}".format(current_timestamp), exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    xm.save(checkpoint, "./../../models/checkpoints/{}/checkpoint.pt".format(current_timestamp))
