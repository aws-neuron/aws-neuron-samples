#!/bin/bash

set -x

# Supported client types (matching SupportedClients class)
LLM_PERF="llm_perf"
LLM_PERF_GITHUB_PATCHED="llm_perf_github_patched"

SUPPORTED_CLIENTS=("$LLM_PERF" "$LLM_PERF_GITHUB_PATCHED")

if [ $# -gt 1 ]; then
    echo "Usage: $0 [client_type]"
    echo "Error: Too many arguments provided"
    exit 1
fi

if [ $# -eq 1 ]; then
    client_type=$1
fi

# Default to LLM_PERF if client_type is not set
if [ -z "$client_type" ]; then
    client_type="$LLM_PERF_GITHUB_PATCHED"
fi

# Validate client type
if [[ ! " ${SUPPORTED_CLIENTS[*]} " =~ " ${client_type} " ]]; then
    echo "Error: Invalid client type '${client_type}'."
    echo "Valid options are: ${SUPPORTED_CLIENTS[*]}"
    exit 1
fi

cd ~
python3 -m venv ~/llm_perf_venv
source llm_perf_venv/bin/activate
pip install -U pip

# Use LLMPERF_INSTALL_DIR if set, otherwise use script directory
INSTALL_DIR="${LLMPERF_INSTALL_DIR:-$(dirname "$0")}"
cd "$INSTALL_DIR" || exit 1

# Uninstall any existing llmperf installation
for package in llmperf amzn-llm-perf; do
    if pip show "$package" >/dev/null 2>&1; then
        echo "Uninstalling existing $package installation..."
        pip uninstall -y "$package"
    else
        echo "No existing $package installation found."
    fi
done

if [ "$client_type" == "$LLM_PERF_GITHUB_PATCHED" ]; then
    # Exit if directory already exists
    if [ -d ~/llmperfGithubPatched ]; then
        rm -rf ~/llmperfGithubPatched
    fi
    # Clone the repository directly to the desired folder name
    git clone https://github.com/ray-project/llmperf.git ~/llmperfGithubPatched || { echo "Failed to clone repository"; exit 1; }
    cd ~/llmperfGithubPatched || { echo "Failed to change to llmperfGithubPatched directory"; exit 1; }
    wget https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_downloads/d406a6505a2e3ab07bb3cd5b2f8f5e04/neuron_perf.patch || { echo "Failed to download patch"; exit 1; }
    # Apply the patch
    git apply neuron_perf.patch || { echo "Failed to apply patch"; exit 1; }
    echo "Successfully patched llmperf repository in llmperfGithubPatched folder"
else
    git clone https://github.com/ray-project/llmperf.git
    cd llmperf || exit
fi

pip install -e .
pip install fastapi==0.112.4
pip install "pydantic>2.10"
pip install pyarrow==20.0.0