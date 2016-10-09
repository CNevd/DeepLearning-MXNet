import os, sys
import mxnet as mx
import numpy as np
import argparse
import train_model
import logging
logging.basicConfig(level=logging.DEBUG)
from myDataIter import get_iterator, myDataIter

parser = argparse.ArgumentParser(description='Deep Click Prediction Model')
parser.add_argument('--gpus', type=int, default=-1,
	      help='which gpu card to use, -1 means using cpu')
parser.add_argument('--num-epochs', type=int, default=50,
	      help='the maximal number of training epochs')
parser.add_argument('--load-epoch', type=int,
              help="load the model on an epoch using the model-prefix")
parser.add_argument('--lr', type=float, default=.1,
	      help='the initial learning rate')
parser.add_argument('--batch-size', type=int, default=2000,
	      help='the batch size')
parser.add_argument('--model-prefix', type=str,
	      help='the prefix of the model to load/save')
parser.add_argument('--save-model-prefix', type=str,
              help='the prefix of the model to save')
parser.add_argument('--kv-store', type=str, default='local',
	      help='the kvstore type')
parser.add_argument('--active-type', type=str, default='relu',
	      help='active type, option: sigmoid, tanh, relu')
args = parser.parse_args()
logging.info('arguments %s', args)



def get_symbol(input_dim, embbed_size):
    data = mx.symbol.Variable('data')
    embed_weight=mx.sym.Variable("embed_weight")
    emb = mx.symbol.Embedding(data=data, weight = embed_weight, input_dim = input_dim, output_dim = embbed_size)
    flatten = mx.symbol.Flatten(emb, name = "emb_flatten")
    fc1  = mx.symbol.FullyConnected(data = flatten, name='fc1', num_hidden = 300)
    bn1 = mx.symbol.BatchNorm(data=fc1, name="bn1")
    act1 = mx.symbol.Activation(data = fc1, name='act1', act_type=args.active_type)
    dp1 = mx.symbol.Dropout(data = act1, p=0.05)
    fc2  = mx.symbol.FullyConnected(data = dp1, name = 'fc2', num_hidden = 100)
    bn2 = mx.symbol.BatchNorm(data=fc2, name="bn2")
    act2 = mx.symbol.Activation(data = fc2, name='act2', act_type=args.active_type)
    dp2 = mx.symbol.Dropout(data = act2, p=0.05)
    fc3  = mx.symbol.FullyConnected(data = dp2, name='fc3', num_hidden = 1)
    dcpm  = mx.symbol.LogisticRegressionOutput(data = fc3, name = 'softmax')
    return dcpm


if __name__ == '__main__':
    net_dcpm = get_symbol(133465, 11)
    (train_iter,val_iter) = get_iterator(args.batch_size)
    train = myDataIter('./data/train_fm', args.batch_size)
    # train
    train_model.fit(args, net_dcpm, train, val_iter) 
