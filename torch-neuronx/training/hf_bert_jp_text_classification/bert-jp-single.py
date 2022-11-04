from transformers import BertForSequenceClassification, BertJapaneseTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import torch, torch_xla.core.xla_model as xm

device = xm.xla_device()

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

train_dataset = load_from_disk("./train/").with_format("torch")
eval_dataset = load_from_disk("./test/").with_format("torch")

training_args = TrainingArguments(
    num_train_epochs = 10,
    learning_rate = 5e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    output_dir = "./results",
)

    #logging_strategy = "epoch",

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    tokenizer = tokenizer,
)

train_result = trainer.train()
print(train_result)

eval_result = trainer.evaluate()
print(eval_result)

trainer.save_model("./results")
