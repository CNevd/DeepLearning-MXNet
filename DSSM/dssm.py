import sys,os
import time
import numpy as np
import mxnet as mx
from DataIter import DataIter
import logging
logging.basicConfig(level=logging.DEBUG)


# Parameters
USR_NUM = 2953334
DOC_DIM = 200
OUT_DIM = 128
batch_size = 2048
num_hidden = 512
num_epoch = 1


class Cosine(mx.metric.EvalMetric):
  def __init__(self):
    super(Cosine, self).__init__('loss')

  def update(self, label, preds):
    for pred in preds:
      self.sum_metric += pred.asnumpy().mean()
      self.num_inst += 1

class Acc(mx.metric.EvalMetric):
  def __init__(self):
    super(Acc, self).__init__('acc')

  def update(self, label, preds):
    for pred in preds:
      pred_np = pred.asnumpy()
      self.sum_metric += (pred_np<0.693147181).sum()
      self.num_inst += len(pred_np)

def get_dssm():
  doc_pos = mx.sym.Variable('doc_pos')
  doc_neg = mx.sym.Variable('doc_neg')
  data_usr = mx.sym.Variable("data_usr", stype='csr')

  #with mx.AttrScope(ctx_group="cpu"):
  w_usr = mx.sym.Variable('usr_weight', stype='row_sparse', shape=(USR_NUM, OUT_DIM))
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
  #with mx.AttrScope(ctx_group="cpu"):
  usr1 = mx.sym.dot(data_usr, w_usr)
  usr = mx.sym.L2Normalization(data=usr1)
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
  train_iter = DataIter('./dssm_data/dssm/0', batch_size, USR_NUM, DOC_DIM)
  data_names = [k[0] for k in train_iter.provide_data]
  label_names = [k[0] for k in train_iter.provide_label]

  # Set symbol
  dssm = get_dssm()
  arg_names = dssm.list_arguments()
  arg_shape, out_shape, aux_shape = dssm.infer_shape(data_usr = (batch_size, USR_NUM),
                                                     doc_pos  = (batch_size, DOC_DIM),
                                                     doc_neg  = (batch_size, DOC_DIM))
  arg_shape_dict = dict(zip(arg_names, arg_shape))
  print "DSSM: ", arg_shape_dict

  # Module
  kv = mx.kvstore.create('local')
  mod = mx.mod.Module(symbol=dssm, data_names=data_names, label_names=label_names, logger=logger, context=ctx)
  mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
  mod.init_params(initializer=mx.init.Uniform(scale=.05))
  #mod.init_params(mx.init.Xavier(factor_type="in", magnitude=2.34))
  sgd = mx.optimizer.Adam(learning_rate=0.01, rescale_grad=1.0/batch_size)
  mod.init_optimizer(optimizer=sgd, kvstore=kv)

  # Metric and Callback
  metric = mx.metric.CompositeEvalMetric()
  metric.add(Cosine())
  metric.add(Acc())
  batch_end_callback = mx.callback.Speedometer(batch_size, frequent=50)
  epoch_end_callback = mx.callback.do_checkpoint('dssm', period=1)

  # Train
  logging.debug('start training ...')
  for epoch in range(num_epoch):
    tic = time.time()
    metric.reset()
    nbatch = 0
    data_iter = iter(train_iter)
    end_of_batch = False
    next_batch = next(data_iter)
    while not end_of_batch:
      batch = next_batch
      # Get row ids for devices
      if (len(ctx) > 1):
        mx.module.executor_group._load_data(batch, 
                                            mod._exec_group.data_arrays,
                                            mod._exec_group.data_layouts)
        data_num = len(mod._exec_group.data_arrays[0])
        row_ids = [mod._exec_group.data_arrays[0][i][1].indices
                     for i in range(data_num)]
      else:
        row_ids = [batch.data[0].indices]

      # pull sparse weight
      index = mod._exec_group.param_names.index('usr_weight')
      kv.row_sparse_pull('usr_weight', mod._exec_group.param_arrays[index],
                          priority=-index, row_ids=row_ids)
      mod.forward_backward(batch)
      # update parameters
      mod.update()
      try:
          # pre fetch next batch
          next_batch = next(data_iter)
          mod.prepare(next_batch)
      except StopIteration:
          end_of_batch = True
      # accumulate metric
      mod.update_metric(metric, batch.label)
      batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                eval_metric=metric,
                                                locals=locals())
      batch_end_callback(batch_end_params)
      nbatch += 1

    # one epoch of training is finished
    for name, val in metric.get_name_value():
      logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
      toc = time.time()
      logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

    # sync aux params across devices
    arg_params, aux_params = mod.get_params()
    mod.set_params(arg_params, aux_params)
    epoch_end_callback(epoch, mod.symbol, arg_params, aux_params)

    train_iter.reset()



if __name__ == '__main__':
  gpus = '0' # change to None to use cpu
  ctx = mx.cpu() if gpus is None else [mx.gpu(int(i)) for i in gpus.split(',')]
  train(ctx)
