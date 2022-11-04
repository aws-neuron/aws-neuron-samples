#!/bin/bash

rm -rf /tmp/parallel_compile_workdir/
rm -rf /var/tmp/neuron-compile-cache/

pip list | grep "neuron\|torch"
yum list installed | grep neuron
pip list | grep transformers


echo "### Prepare dataset"
python3 datasets_setup.py


echo "### single worker precompile"
sudo rmmod neuron; sudo modprobe neuron
time XLA_USE_BF16=1 neuron_parallel_compile python3 bert-jp-precompile.py

echo "### single worker 1st training"
sudo rmmod neuron; sudo modprobe neuron
time XLA_USE_BF16=1 python3 bert-jp-single.py

echo "### single worker 2nd training"
sudo rmmod neuron; sudo modprobe neuron
time XLA_USE_BF16=1 python3 bert-jp-single.py


echo "### Run Inference"
python3 bert-jp-inference.py


echo "### dual worker precompile"
sudo rmmod neuron; sudo modprobe neuron
time XLA_USE_BF16=1 neuron_parallel_compile torchrun --nproc_per_node=2 bert-jp-dual.py

echo "### dual worker 1st dual training"
sudo rmmod neuron; sudo modprobe neuron
time XLA_USE_BF16=1 torchrun --nproc_per_node=2 bert-jp-dual.py

echo "### dual worker 2nd dual training"
sudo rmmod neuron; sudo modprobe neuron
time XLA_USE_BF16=1 torchrun --nproc_per_node=2 bert-jp-dual.py
