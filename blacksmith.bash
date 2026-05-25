#!/bin/bash

git clone git@github.com:tenstorrent/tt-blacksmith.git || true

cd tt-blacksmith
uv venv --python=3.12 --clear

source env/activate --xla


export LD_LIBRARY_PATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH
# NEEDED FOR
# Traceback (most recent call last):
#   File "/home/delaunap/workspace/tenstorrent_tests/tt-blacksmith/blacksmith/experiments/torch/mnist/train.py", line 11, in <module>
#     import torch_xla
#   File "/home/delaunap/workspace/tenstorrent_tests/tt-blacksmith/env/xla_env/lib/python3.12/site-packages/torch_xla/__init__.py", line 15, in <module>
#     import _XLAC_cuda_functions
# ImportError: libpython3.12.so.1.0: cannot open shared object file: No such file or directory

n=$(ls ../blacksmith_*.out 2>/dev/null | wc -l)
n=$((n + 1))


export TT_VISIBLE_DEVICES=0
export PJRT_DEVICE=TT
export XLA_STABLEHLO_COMPILE=1

sudo /home/delaunap/.tenstorrent-venv/bin/tt-smi -r

python blacksmith/experiments/torch/mnist/train.py > ../blacksmith_${n}.out 2> ../blacksmith_${n}.err

