import sys,os
import numpy as np
import mxnet as mx
from DataIter import DataIter
import logging
logging.basicConfig(level=logging.DEBUG)


# Parameters
USR_NUM = 2953334
DOC_DIM = 200
OUT_DIM = 128
batch_size = 1024
num_hidden = 512


class Cosine(mx.metric.EvalMetric):
  def __init__(self):
    super(Cosine, self).__init__('loss')

  def update(self, label, preds):
    for pred in preds:
      self.sum_metric += pred.asnumpy().mean()
      self.num_inst += 1

def get_dssm():
  doc_pos = mx.sym.Variable('doc_pos')
  doc_neg = mx.sym.Variable('doc_neg')
  data_usr = mx.sym.Variable("data_usr")

  # shared weights
  w1 = mx.sym.Variable('fc1_doc_weight')
  w2 = mx.sym.Variable('fc2_doc_weight')
  w3 = mx.sym.Variable('fc3_doc_weight')
  b1 = mx.sym.Variable('fc1_doc_bias')
  b2 = mx.sym.Variable('fc2_doc_bias')
  b3 = mx.sym.Variable('fc3_doc_bias')

  def cosine(usr, doc):
    dot = usr * doc
    dot = mx.sym.sum_axis(dot, axis=1)
    return dot

  def doc_mlp(data):
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=num_hidden, name='fc1', weight=w1, bias=b1)
    fc1 = mx.sym.Activation(data=fc1, act_type='relu')
    fc2 = mx.sym.FullyConnected(data=fc1, num_hidden=num_hidden, name='fc2', weight=w2, bias=b2)
    fc2 = mx.sym.Activation(data=fc2, act_type='relu')
    fc3 = mx.sym.FullyConnected(data=fc2, num_hidden=OUT_DIM, name='fc3', weight=w3, bias=b3)
    fc3 = mx.sym.Activation(data=fc3, act_type='relu')
    fc3 = mx.sym.L2Normalization(data=fc3)
    return fc3

  # usr net
  usr = mx.sym.Embedding(data=data_usr, input_dim=USR_NUM, output_dim=OUT_DIM, name='usr_embbed')
  usr = mx.sym.Flatten(data=usr)
  usr = mx.sym.L2Normalization(data=usr)
  # doc net
  mlp_pos = doc_mlp(doc_pos)
  mlp_neg = doc_mlp(doc_neg)

  cosine_pos = cosine(usr, mlp_pos)
  cosine_neg = cosine(usr, mlp_neg)
  exp = mx.sym.exp(data=(cosine_neg - cosine_pos))
  pred = mx.sym.log1p(data=exp)
  out = mx.sym.MAERegressionOutput(data=pred, name='mae')
  return out

def train(ctx):
  # Setup logging
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  # Load data
  train_iter = DataIter('./dssm_data/dssm/0', batch_size, DOC_DIM)
  data_names = [k[0] for k in train_iter.provide_data]
  label_names = [k[0] for k in train_iter.provide_label]

  # Set symbol
  dssm = get_dssm()
  arg_names = dssm.list_arguments()
  arg_shape, out_shape, aux_shape = dssm.infer_shape(data_usr = (batch_size,),
                                                     doc_pos  = (batch_size, DOC_DIM),
                                                     doc_neg  = (batch_size, DOC_DIM))
  arg_shape_dict = dict(zip(arg_names, arg_shape))
  print "DSSM: ", arg_shape_dict

  # Module
  mod = mx.module.Module(dssm, data_names=data_names, label_names=label_names, logger=logger, context=ctx)

  opt_params = {'wd': 0.01,
                'learning_rate': 0.01}
  batch_end_callback = mx.callback.Speedometer(batch_size, frequent=1)
  epoch_end_callback = mx.callback.do_checkpoint('dssm', period=1)
  kv = 'device'

  #mx.profiler.profiler_set_config(mode='all', filename='profile_output.json')
  #mx.profiler.profiler_set_state('run')
  def asum_stat(x):
      return x
  #monitor = mx.mon.Monitor(1, stat_func=asum_stat, pattern=".*")
  monitor = None

  mod.fit(train_iter, eval_metric=Cosine(),
          batch_end_callback=batch_end_callback,
          epoch_end_callback=epoch_end_callback,
          kvstore=kv,
          optimizer='adadelta', optimizer_params=opt_params,
          initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
          num_epoch=1,
          monitor=monitor)

  #mx.profiler.profiler_set_state('stop')


if __name__ == '__main__':
  gpus = "0" # change to None to use cpu
  ctx = mx.cpu() if gpus is None else [mx.gpu(int(i)) for i in gpus.split(',')]
  train(ctx)
