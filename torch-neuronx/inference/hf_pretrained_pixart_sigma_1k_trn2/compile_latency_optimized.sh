#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH

echo "compiling text encoder"
python neuron_pixart_sigma/compile_text_encoder.py \
--compiled_models_dir "compile_workdir_latency_optimized"

echo "compiling transformer"
python neuron_pixart_sigma/compile_transformer_latency_optimized.py \
--compiled_models_dir "compile_workdir_latency_optimized"

echo "compiling decoder"
python neuron_pixart_sigma/compile_decoder.py \
--compiled_models_dir "compile_workdir_latency_optimized"