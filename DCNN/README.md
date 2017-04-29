# Dynamic Convolutional Neural Network
An MXNet implementation of paper [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188v1)
This paper describe a convolutional architecture dubbed the Dynamic Convolutional Neural Network (DCNN) which can adopt for the semantic modelling of sentences
The network contains 1-D Wide Convolution, Dynamic k-Max Pooling, Folding layers

## Using it
First download the dataset use `download.sh`, also you can use your own dataset with modifying `DataIter.py`
To run it, just `python dcnn_train.py`
I was able to achieve a best test accuracy of 85-86%

## Paper/Matlab implementation issues
There is some discrepancy between the paper and Matlab code provided. Therefore, it was difficult to rely on the Matlab code for details not provided in the paper. For example: (1) different number of layers and filters. (2) the L2 regularization is not specified in the paper but is very detailed in the code (different values for different matrices). It would be hard to guess those values

## Dynamic k-Max Pooling
Currently we use CustomOp to implement the k-max pooling layer which is executed on the CPU
however it can be implemented by symbol function after [[Topk and arange](https://github.com/dmlc/mxnet/pull/4047) is merged

## References
- [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188v1)
- [Dynamic CNN in theano](https://github.com/FredericGodin/DynamicCNN)
