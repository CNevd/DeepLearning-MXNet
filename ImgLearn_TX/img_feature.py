#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import mxnet as mx
import argparse
import logging
from lr_scheduler_Lan import Lan_Scheduler
logging.basicConfig()


def get_symbol(dropout):
  # data
  data = mx.symbol.Variable('data')
  # layer1
  conv1 = mx.symbol.Convolution(data=data, num_filter=5, kernel=(5,5), stride=(1,1), pad=(2,2), name='conv1')
  act1 = mx.symbol.Activation(data=conv1, act_type='relu', name='relu1')
  pool1 = mx.symbol.Pooling(data=act1, kernel=(3,3), stride=(2,2), pad=(1,1), name='pool1', pool_type='max')
  lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
  # layer2
  conv2 = mx.symbol.Convolution(data=lrn1, num_filter=5, kernel=(5,5), stride=(1,1), pad=(2,2), name='conv2')
  act2 = mx.symbol.Activation(data=conv2, act_type='relu', name='relu2')
  pool2 = mx.symbol.Pooling(data=act2, kernel=(3,3), stride=(2,2), pad=(1,1), name='pool2', pool_type='max')
  lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
  # layer3
  conv3 = mx.symbol.Convolution(data=lrn2, num_filter=5, kernel=(5,5), stride=(1,1), pad=(2,2), name='conv3')
  act3 = mx.symbol.Activation(data=conv3, act_type='relu', name='relu3')
  pool3 = mx.symbol.Pooling(data=act3, kernel=(2,2), stride=(1,1), name='pool3', pool_type='max')
  lrn3 = mx.symbol.LRN(data=pool3, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
  # layer4
  conv4 = mx.symbol.Convolution(data=lrn3, num_filter=5, kernel=(5,5), stride=(1,1), pad=(2,2), name='conv4')
  act4 = mx.symbol.Activation(data=conv4, act_type='relu', name='relu4')
  pool4 = mx.symbol.Pooling(data=act4, kernel=(2,2), stride=(1,1), name='pool4', pool_type='max')
  lrn4 = mx.symbol.LRN(data=pool4, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
  # layer5
  fc5 = mx.symbol.FullyConnected(data=lrn4, num_hidden=512, name='fc5')
  dp5 = mx.symbol.Dropout(data=fc5, p=dropout, name="dp5")
  #layer6
  fc6 = mx.symbol.FullyConnected(data=dp5, num_hidden=256, name='fc6')
  dp6 = mx.symbol.Dropout(data=fc6, p=dropout, name="dp6")
  # layer7
  fc7 = mx.symbol.FullyConnected(data=fc6, num_hidden=1000, name='fc7')
  # output
  symbol = mx.symbol.SoftmaxOutput(data=fc7, name='softmax')
  return symbol

def get_iterator(args, kv, data_shape):
    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "train.rec",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "test.rec",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)


def train_img(args, ctx):
    # setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    kv = mx.kvstore.create(args.kv_store)
    print "Loading data..."
    data_shape = (3, 28, 28)
    train_iter, val_iter = get_iterator(args, kv, data_shape)
    # load symbol
    ext_net = get_symbol(0.5)

    # initialization
    arg_params = {}
    arg_names = ext_net.list_arguments()
    arg_shape, out_shape, aux_shape = ext_net.infer_shape(data = (args.batch_size,3,28, 28))
    arg_shape_dict = dict(zip(arg_names, arg_shape))
  
    lr_scheduler = Lan_Scheduler(500)
    optimizer_params = {'lr_scheduler': lr_scheduler, 'learning_rate': 0.01}
    mod = mx.module.module.Module(ext_net, logger=logger, context=ctx,
                                work_load_list=args.work_load_list)

    batch_end_callback = mx.callback.Speedometer(args.batch_size, frequent=args.frequent)
    eval_metric = mx.metric.CompositeEvalMetric()
    metric_acc = mx.metric.Accuracy()
    metric_ce = mx.metric.CrossEntropy()
    eval_metric.add(metric_acc)
    eval_metric.add(metric_ce)
  
    # start train
    print "start training..." 
    mod.fit(train_iter, val_iter, eval_metric=eval_metric, batch_end_callback=batch_end_callback,
            kvstore=args.kv_store,
            optimizer='NAG', optimizer_params=optimizer_params,
            initializer=mx.initializer.Uniform(0.01), arg_params=arg_params,
            allow_missing=True,
            begin_epoch=args.begin_epoch, num_epoch=args.num_epoch, validation_metric='acc')
  
    print "Train done for epoch: %s"%args.num_epoch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a image feature extraction Neural Network  ')
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'img_feat'), type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='the input data directory',
                        default='data/', type=str)
    parser.add_argument('--gpus', help='GPU device to train with, eg: 0,1,2',
                        default=None, type=str)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', help='num epoch of training',
                        default=100, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=50, type=int)
    parser.add_argument('--kv_store', dest='kv_store', help='the kv-store type',
                        default='local', type=str)
    parser.add_argument('--work_load_list', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size',
                        default=256, type=int)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    ctx = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_img(args,ctx)

