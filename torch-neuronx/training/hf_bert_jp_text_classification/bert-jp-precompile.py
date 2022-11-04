from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import torch, torch_xla.core.xla_model as xm

device = xm.xla_device()

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

train_dataset = load_from_disk("./train/").with_format("torch")
train_dataset = train_dataset.select(range(64))

training_args = TrainingArguments(
    num_train_epochs = 2,
    learning_rate = 5e-5,
    per_device_train_batch_size = 8,
    output_dir = "./results",
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
)

train_result = trainer.train()
