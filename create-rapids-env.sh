#!/bin/bash
# pwd
mamba --version

mamba create -p /root/autodl-tmp/cafa6/rapids-env/rapids-env -c rapidsai -c conda-forge -c nvidia  \
    python=3.8 cudf=23.02 cuml=23.02 cugraph=23.02 cupy -y

mamba activate /root/autodl-tmp/cafa6/rapids-env/rapids-env
which python

pip install tqdm py-boost

# 
pip uninstall cupy numba -y # I reinstall default rapids cupy and numba via pypi due to the problems of my environment
# It is not actually needed in general case
pip install tqdm cupy-cuda112==10.6 numba==0.56.4 py-boost==0.4.3 