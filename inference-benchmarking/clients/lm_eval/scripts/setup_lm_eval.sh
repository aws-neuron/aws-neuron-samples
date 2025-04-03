#!/bin/bash
# Install LMEvaL
cd ~
python3 -m venv ~/lm_eval_venv
source lm_eval_venv/bin/activate
pip install -U pip
pip install lm_eval[api]==0.4.7