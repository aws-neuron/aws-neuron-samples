#!/usr/bin/env bash

set -Exeuo

HF_CACHE_PKG=huggingface_cache_GPT2_1p5B_0513.tgz
mkdir -p ~/.cache
pushd ~/.cache
aws s3 cp --no-progress s3://kaena-nn-models/train/pt/$HF_CACHE_PKG .
tar -xzf $HF_CACHE_PKG
rm $HF_CACHE_PKG
popd


