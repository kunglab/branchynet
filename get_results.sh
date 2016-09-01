#!/bin/sh
./download_data.sh

mkdir _models
mkdir _figs

python experiment_lenet_mnist.py
python experiment_alex_cifar10.py
python experiment_resnet_cifar10.py
