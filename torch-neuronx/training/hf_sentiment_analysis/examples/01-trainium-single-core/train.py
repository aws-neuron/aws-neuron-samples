import csv
from datasets import Dataset, DatasetDict
import logging
import os
import pandas as pd
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
import torch_xla.core.xla_model as xm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "xla"

batch_size = 8
num_epochs = 6

logger.info("Device: {}".format(device))

def tokenize_and_encode(examples):
    results = tokenizer(examples["text"], padding="max_length", truncation=True)
    return results

if __name__ == '__main__':

    train = pd.read_csv(
        "./../../data/train.csv",
        sep=',',
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        encoding='utf-8',
        error_bad_lines=False
    )

    train_dataset = Dataset.from_dict(train)

    hg_dataset = DatasetDict({"train": train_dataset})

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    ds_encoded = hg_dataset.map(tokenize_and_encode, batched=True, remove_columns=["text"])

    ds_encoded.set_format("torch")

    train_dl = DataLoader(ds_encoded["train"], shuffle=True, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    model.to(device)

    current_timestamp = strftime("%Y-%m-%d-%H-%M", gmtime())

    optimizer = AdamW(model.parameters(), lr=1.45e-4)

    num_training_steps = num_epochs * len(train_dl)
    progress_bar = tqdm(range(num_training_steps))
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            xm.mark_step()
            optimizer.zero_grad()
            progress_bar.update(1)

        logger.info("Epoch {}, rank {}, Loss {:0.4f}".format(epoch, xm.get_ordinal(), loss.detach().to("cpu")))

    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    os.makedirs("./../../models/checkpoints/{}".format(current_timestamp), exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    xm.save(checkpoint, "./../../models/checkpoints/{}/checkpoint.pt".format(current_timestamp))
