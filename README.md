## FlowNet2 (TensorFlow)

This repo contains FlowNet2[1] for TensorFlow. It includes FlowNetC, S, CS, CSS, CSS-ft-sd, SD, and 2.

### Installation
```
pip install enum
pip install pypng
pip install matplotlib
pip install image
pip install scipy
pip install numpy
pip install tensorflow
```

Linux:
`sudo apt-get install python-tk`

You must have CUDA installed:
`make all`

### Download weights
To download the weights for all models (4.4GB), run the `download.sh` script in the `checkpoints` directory. All test scripts rely on these checkpoints to work properly.

### Sources
[1] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, T. Brox
FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks,
IEEE Conference in Computer Vision and Pattern Recognition (CVPR), 2017.
