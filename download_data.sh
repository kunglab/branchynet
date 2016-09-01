#!/bin/sh
mkdir -p datasets/data/pcifar10
cd datasets/data/pcifar10
# or download from here https://drive.google.com/file/d/0Byyuc5LmNmJPWUc5dVdUSms3U1E/view?usp=sharing
wget 'https://dl.dropboxusercontent.com/u/4542002/branchynet/data.npz' -O data.npz
