#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import numpy as np
import mxnet as mx
import DataIter
import argparse
import logging
logging.basicConfig()
import debug_ipy


class k_max_pool(mx.operator.CustomOp):
  def __init__(self, k):
    super(k_max_pool, self).__init__()
    self.k = int(k)
  def forward(self, is_train, req, in_data, out_data, aux):
    x = in_data[0].asnumpy()
    assert(4 == len(x.shape))
    ind = np.argsort(x, axis = 2)
    sorted_ind = np.sort(ind[:,:,-(self.k):,:], axis = 2)
    dim0, dim1, dim2, dim3 = sorted_ind.shape
    self.indices_dim0 = np.arange(dim0).repeat(dim1 * dim2 * dim3)
    self.indices_dim1 = np.transpose(np.arange(dim1).repeat(dim2 * dim3).reshape((dim1*dim2*dim3, 1)).repeat(dim0, axis=1)).flatten()
    self.indices_dim2 = sorted_ind.flatten() 
    self.indices_dim3 = np.transpose(np.arange(dim3).repeat(dim2).reshape((dim2*dim3, 1)).repeat(dim0 * dim1, axis = 1)).flatten()
    y = x[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3].reshape(sorted_ind.shape)
    self.assign(out_data[0], req[0], mx.nd.array(y))

  def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
    x = out_grad[0].asnumpy()
    y = in_data[0].asnumpy()
    assert(4 == len(x.shape))
    assert(4 == len(y.shape))
    y[:,:,:,:] = 0
    y[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3] \
      = x.reshape([x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3],])
    self.assign(in_grad[0], req[0], mx.nd.array(y))
    
@mx.operator.register("k_max_pool")
class k_max_poolProp(mx.operator.CustomOpProp):
  def __init__(self, k):
    self.k = int(k)
    super(k_max_poolProp, self).__init__(True)
  def list_argument(self):
    return ['data']
  def list_outputs(self):
    return ['output']
  def infer_shape(self, in_shape):
    data_shape = in_shape[0]
    assert(len(data_shape) == 4)
    out_shape = (data_shape[0], data_shape[1], self.k, data_shape[3])
    return [data_shape], [out_shape]

  def create_operator(self, ctx, shapes, dtypes):
    return k_max_pool(self.k)


def fold(x, shape):
  long_rows = mx.sym.Reshape(data=x, shape=(int(shape[0]), int(shape[1]), -1, 2))
  sumed = mx.sym.sum(long_rows, axis=3, keepdims=True)
  fold_out = mx.sym.Reshape(data=sumed, shape=(int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])/2))
  return fold_out



def get_dcnn(sentence_size, embed_size, batch_size, vocab_size,
       dropout = 0.5,
       ktop = 4,
       filter_widths=[7,5],
       fiters=[6,14],
       conv_wds=[0.000015,0.0000015],
       pool_widths=[10,5]):
  data = mx.sym.Variable('data')

  # embedding layer
  embed_layer = mx.sym.Embedding(data=data, input_dim=vocab_size, output_dim=embed_size, name='embed', attr={'wd_mult':'0.00005'})
  embed_out = mx.sym.Reshape(data=embed_layer, shape=(batch_size, 1, sentence_size, embed_size))

  layers = [embed_out]

  # ConvFoldingPoolLayer
  nl = float(len(fiters))
  for i in xrange(len(fiters)):
    # row wise wide conv
    conv_outi = mx.sym.Convolution(data=layers[-1], name="conv%s" % i, kernel=(filter_widths[i], 1), num_filter=fiters[i], pad=(filter_widths[i]-1,0), attr={'wd_mult':str(conv_wds[i])})

    _, out_shape, _ = conv_outi.infer_shape(data = (batch_size, sentence_size))
    assert(1 == len(out_shape))
    fold_outi = fold(conv_outi, out_shape[0])

    # get ki for axis=2
    ki = ktop if i == nl-1 else max(ktop, int(np.ceil((nl-i-1) / nl * float(out_shape[0][2]))))

    pool_outi = mx.symbol.Custom(data=fold_outi, name='k_max_pool%s' % i, op_type='k_max_pool', k=ki)
    #pool_outi = mx.sym.Pooling(data=fold_outi, pool_type='max', kernel=(pool_widths[i], 1), stride=(2,1))
    act_outi = mx.sym.Activation(data=pool_outi, act_type='tanh', name="act%s" % i)
    layers.append(act_outi)

  if dropout > 0.0:
    dp_out = mx.sym.Dropout(data=layers[-1], p=dropout, name="dp")
  else:
    dp_out = layers[-1]

  fc = mx.symbol.FullyConnected(data=dp_out, num_hidden=2, name='fc', attr={'wd_mult':'0.00005'})
  dcnn  = mx.symbol.SoftmaxOutput(data = fc, name = 'softmax')
  group = mx.symbol.Group([dcnn, data,embed_layer, embed_out, conv_outi,fold_outi, pool_outi])
  return dcnn

def train_dcnn(args, ctx):

  # setup logging
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  print "Loading data..."
  (train_iter, val_iter, sentence_size, vocab_size) = DataIter.load_data('./data/twitter.pkl', args.batch_size)
  #(train_iter, sentence_size, vocab_size) = DataIter.read_and_sort_matlab_data('./data/binarySentiment/train.txt', './data/binarySentiment/train_lbl.txt', args.batch_size)
  #(val_iter, _, _) = DataIter.read_and_sort_matlab_data('./data/binarySentiment/test.txt', './data/binarySentiment/test_lbl.txt', args.batch_size)
  data_names = [k[0] for k in train_iter.provide_data]
  label_names = [k[0] for k in train_iter.provide_label]

  # load symbol
  dcnn = get_dcnn(sentence_size, args.embed_size, args.batch_size, vocab_size)

  # initialization
  arg_params = {}
  aux_params = {}
  arg_names = dcnn.list_arguments()
  arg_shape, out_shape, aux_shape = dcnn.infer_shape(data = (args.batch_size, sentence_size))
  arg_shape_dict = dict(zip(arg_names, arg_shape))

  arg_params['embed_weight'] = mx.random.normal(0, 0.05, shape=arg_shape_dict['embed_weight'])
  arg_params['conv0_weight'] = mx.random.uniform(0, 0.01, shape=arg_shape_dict['conv0_weight'])
  arg_params['conv0_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv0_bias'], ctx=ctx)
  arg_params['conv1_weight'] = mx.random.uniform(0, 0.01, shape=arg_shape_dict['conv1_weight'])
  arg_params['conv1_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv1_bias'], ctx=ctx)
  arg_params['fc_weight'] = mx.random.uniform(0, 0.01, shape=arg_shape_dict['fc_weight'])
  arg_params['fc_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc_bias'], ctx=ctx)

  optimizer_params = {'wd': 0.0005,
                      'learning_rate': 0.1,
                      'rescale_grad': (1.0 / args.batch_size)}
  mod = mx.module.module.Module(dcnn, data_names=data_names, label_names=label_names,
                         logger=logger, context=ctx, work_load_list=args.work_load_list)

  batch_end_callback = mx.callback.Speedometer(args.batch_size, frequent=args.frequent)
  eval_metric = mx.metric.CompositeEvalMetric()
  metric_acc = mx.metric.create('acc')
  metric_ce = mx.metric.create('ce')
  eval_metric.add(metric_acc)
  eval_metric.add(metric_ce)

  # start train
  print "start training..." 
  mod.fit(train_iter, val_iter, eval_metric=eval_metric, batch_end_callback=batch_end_callback,
          kvstore=args.kv_store,
          optimizer='adagrad', optimizer_params=optimizer_params,
          initializer=mx.initializer.Uniform(0.01), arg_params=arg_params,
          allow_missing=True,
          begin_epoch=args.begin_epoch, num_epoch=args.num_epoch, validation_metric='acc')
  
  print "Train done for epoch: %s"%args.num_epoch
  # Save model

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Dynamic Convolutional Neural Network  ')
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'dcnn'), type=str)
    parser.add_argument('--gpus', help='GPU device to train with',
                        default=None, type=str)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', help='num epoch of training',
                        default=500, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=500, type=int)
    parser.add_argument('--kv_store', dest='kv_store', help='the kv-store type',
                        default='local', type=str)
    parser.add_argument('--work_load_list', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size',
                        default=4, type=int)
    parser.add_argument('--embed_size', dest='embed_size', help='embedding size of one word, must be even',
                        default=48, type=int)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
  args = parse_args()
  ctx = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
  train_dcnn(args,ctx)

