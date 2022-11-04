from transformers import BertJapaneseTokenizer
from datasets import load_dataset

# Prepare dataset
dataset = load_dataset("amazon_reviews_multi", "ja")
dataset = dataset.remove_columns(["review_id", "product_id", "reviewer_id", "review_title", "language", "product_category"])
dataset = dataset.filter(lambda dataset: dataset["stars"] != 3)
dataset = dataset.map(lambda dataset: {"labels": int(dataset["stars"] > 3)}, remove_columns=["stars"])

# Tokenization
MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["review_body"], padding="max_length", max_length=128, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["review_body"])

train_dataset = tokenized_datasets["train"].shuffle().select(range(4000))
eval_dataset = tokenized_datasets["test"].shuffle().select(range(256))

# Save dataset
train_dataset.save_to_disk("./train/")
eval_dataset.save_to_disk("./test/")

# Print sample
index = 150000
print(dataset["train"][index])
print('Tokenize:', tokenizer.tokenize(dataset["train"]['review_body'][index]))
print('Encode:', tokenizer.encode(dataset["train"]['review_body'][index]))
