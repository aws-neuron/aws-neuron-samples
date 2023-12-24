#!/usr/bin/env bash
set -eu

# installing the requirements
python3 -m pip install transformers regex datasets sentencepiece protobuf==3.20.*

# downloading and tokenizing the dataset
cd ./data
python3 get_dataset.py

# pushing the tokenized dataset to predefined S3 location
aws s3 cp ~/examples_datasets/wikicorpus_llama2_7B_tokenized_4k/ $TOKENIZED_DATASET_URI --recursive --only-show-errors