import csv
from datasets import Dataset, DatasetDict
import logging
import pandas as pd
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "bert-base-cased"
device = "cuda" if torch.cuda.is_available() else "cpu"

num_epochs = 6
batch_size = 8

logger.info("Device: {}".format(device))

## tokenize_and_encode
# params:
#   data: DatasetDict
# This method returns a dictionary of input_ids, token_type_ids, attention_mask
def tokenize_and_encode(data):
    results = tokenizer(data["text"], padding="max_length", truncation=True)
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

    ## Loading Hugging Face AutoTokenizer for the defined model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds_encoded = hg_dataset.map(tokenize_and_encode, batched=True, remove_columns=["text"])

    ds_encoded.set_format("torch")

    ## Creating a DataLoader object for iterating over it during the training epochs
    train_dl = DataLoader(ds_encoded["train"], shuffle=True, batch_size=batch_size)

    ## Loading Hugging Face pre-trained model for sequence classification for the defined model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

    current_timestamp = strftime("%Y-%m-%d-%H-%M", gmtime())

    optimizer = AdamW(model.parameters(), lr=1.45e-4)

    num_training_steps = num_epochs * len(train_dl)
    progress_bar = tqdm(range(num_training_steps))
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    ## Start model training and defining the training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    torch.save(model, "./../../models/checkpoints/{}/checkpoint.pt".format(current_timestamp))