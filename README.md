# BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks

Deep neural networks are state of the art methods for many learning tasks due to their ability to extract increasingly better features at each network layer. However, the improved performance of additional layers in a deep network comes at the cost of added latency and energy usage in feedforward inference. As networks continue to get deeper and larger, these costs become more prohibitive for real-time and energy-sensitive applications. To address this issue, we present BranchyNet, a novel deep network architecture that is augmented with side branches. The architecture allows prediction results for a large portion of test samples to exit the network early via these branches when samples can already be inferred with high confidence. BranchyNet exploits the observation that features learned at an early layer of a network may often be sufficient for the classification of many data points. For more difficult samples, which are expected to be infrequent, BranchyNet will use further or all network layers to provide the best likelihood of correct prediction. We study the BranchyNet architecture using several well-known networks (LeNet, AlexNet, ResNet) and datasets (MNIST, CIFAR10) and show that it can both improve accuracy and significantly reduce the inference time of the network.

This repository containing the code to reproduce result found in "BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks" paper.

If you use this codebase, please cite:

    @article{teerapittayanonbranchynet,
      title={BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks},
      author={Teerapittayanon, Surat and McDanel, Bradley and Kung, HT}
    }
    
### Requirements
* A machine with a decent GPU
* Python 2.7

### Python Dependencies
* chainer==1.17.0
* matplotlib
* dill
* scikit-image
* scipy
* scikit-learn

### Quickstart

1. Download data.npz from https://drive.google.com/file/d/0Byyuc5LmNmJPWUc5dVdUSms3U1E/view?usp=sharing and put it at datasets/data/pcifar10/data.npz
2. Run script
```
./get_results.sh
```
Or take a look at the ipython notebooks.

### Run with Docker

1. Download and install Docker https://www.docker.com/products/docker-desktop
2. Run
```
./docker/build.sh
./docker/shell.sh
```

3. After you are inside the container's shell, run

```
./get_results.sh
```

For GPU support, you may need to customize Dockerfile to support CUDA. See https://docs.chainer.org/en/v1.17.0/install.html for more information.
