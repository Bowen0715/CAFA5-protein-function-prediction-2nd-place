#!/bin/bash

pwd

mamba --version

mamba create -p /root/autodl-tmp/cafa6/pytorch-env python=3.10 -y
mamba activate /root/autodl-tmp/cafa6/pytorch-env
mamba install -y "pytorch>=2.6" pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip3 install torch --index-url https://download.pytorch.org/whl/cu126
mamba install mkl=2024.0 -c conda-forge -y
mamba install -c conda-forge cupy -y
pip install biopython joblib tqdm pandas pyyaml pyarrow numba scikit-learn numpy scipy fair-esm obonet pyvis transformers torchmetrics torchsummary sentencepiece psutil


pip install requests
