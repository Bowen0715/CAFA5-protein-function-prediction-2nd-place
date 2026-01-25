#!/bin/bash
# pwd
mamba --version

# error datashader
# mamba create -p /root/autodl-tmp/cafa6/rapids-env -c rapidsai -c conda-forge -c nvidia  \
#     rapids=23.02 python=3.8 cuda-version=11.2 -y

mamba create -p /root/autodl-tmp/cafa6/rapids-env \
  -c rapidsai -c conda-forge -c nvidia \
  python=3.8 cuda-version=11.2 -y
mamba activate /root/autodl-tmp/cafa6/rapids-env

mamba install -y cudf=23.02 cuml=23.02 cugraph=23.02 -c rapidsai -c conda-forge -c nvidia

which python
pip uninstall cupy numba -y # I reinstall default rapids cupy and numba via pypi due to the problems of my environment
# It is not actually needed in general case
# pip install tqdm cupy-cuda112==10.6 numba==0.56.4 py-boost==0.4.3
pip install tqdm py-boost
