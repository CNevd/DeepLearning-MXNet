import os,sys
import math
import mxnet as mx
import numpy as np
import logging
logging.basicConfig()
from sklearn.preprocessing import StandardScaler

# Parameters
learning_rate = 0.05
batch_size = 100
num_epoch = 15
dropoutRate = 0.05
display_step = 1

# Network Parameters
n_hidden_1 = 784 # 1st layer number of features
n_hidden_2 = 784 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


def selu(data):
  scale = 1.0507009873554804934193349852946
  alpha = 1.6732632423543772848170429916717
  data1 = mx.sym.LeakyReLU(data=data, act_type = "leaky", slope = alpha)
  condition = data>=0
  return scale * mx.sym.where(condition=condition, x=data, y=data1)

def dropout_selu(data, shape, rate):
  alpha= -1.7580993408473766
  fixedPointMean=0.0
  fixedPointVar=1.0
  keep_prob = 1.0 - rate
  if not 0 < keep_prob <= 1:
    raise ValueError("keep_prob must be a scalar tensor or a float in the "
                      "range (0, 1], got %g" % keep_prob)
  if (1 == keep_prob): return data
  random_tensor = mx.sym.random_uniform(shape = shape) + keep_prob
  binary_tensor = mx.sym.floor(data = random_tensor)
  binary_tensor = mx.sym.BlockGrad(data = binary_tensor)
  ret = data * binary_tensor + alpha * (1-binary_tensor)
  a = math.sqrt(fixedPointVar / (keep_prob *((1 - keep_prob) * math.pow(alpha - fixedPointMean, 2) + fixedPointVar)))
  b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
  ret = a * ret + b
  return ret

def get_sym(is_train=True):
  data = mx.sym.Variable('data')

  layer_1 = mx.sym.FullyConnected(data=data, num_hidden=n_hidden_1, name='fc1')
  layer_1 = selu(layer_1)
  if (is_train):
    _, out_shape1, _ = layer_1.infer_shape(data = (batch_size, n_input))
    layer_1 = dropout_selu(layer_1, out_shape1[0], dropoutRate)
  layer_2 = mx.sym.FullyConnected(data=layer_1, num_hidden=n_hidden_2, name='fc2')
  layer_2 = selu(layer_2)
  if (is_train):
    _, out_shape2, _ = layer_2.infer_shape(data = (batch_size, n_input))
    layer_2 = dropout_selu(layer_2, out_shape2[0], dropoutRate)
  layer_out = mx.sym.FullyConnected(data=layer_2, num_hidden=n_classes, name='fc3')
  out = mx.sym.SoftmaxOutput(data = layer_out, name = 'softmax')
  return out

def get_scaler(train_iter):
  train_iter.reset()
  x = None
  for batch in train_iter:
    if x is None:
      x = batch.data[0]
    else:
      x = mx.nd.concat(x, batch.data[0], dim=0)
  return StandardScaler().fit(x.asnumpy())

def train(ctx):
  # setup logging
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  kv = mx.kvstore.create("device")
  print "Loading data..."
  basedir = os.path.dirname(__file__)
  train_dataiter = mx.io.MNISTIter(
          image=os.path.join(basedir, "mnist_data", "train-images-idx3-ubyte"),
          label=os.path.join(basedir, "mnist_data", "train-labels-idx1-ubyte"),
          data_shape=(784,),
          batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
  val_dataiter = mx.io.MNISTIter(
          image=os.path.join(basedir, "mnist_data", "t10k-images-idx3-ubyte"),
          label=os.path.join(basedir, "mnist_data", "t10k-labels-idx1-ubyte"),
          data_shape=(784,),
          batch_size=512, shuffle=True, flat=True, silent=False)

  # Normalize Data to mean = 0, stdev = 1
  Scaler = get_scaler(train_dataiter)

  # get symbol
  train_net = get_sym()
  pred_net = get_sym(False)

  # initialization
  arg_params = {}
  arg_params['fc1_weight'] = mx.nd.random_normal(scale=np.sqrt(1.0/n_input), shape=(n_hidden_1, n_input))
  arg_params['fc2_weight'] = mx.nd.random_normal(scale=np.sqrt(1.0/n_hidden_1), shape=(n_hidden_2, n_hidden_1))
  arg_params['fc3_weight'] = mx.nd.random_normal(scale=np.sqrt(1.0/n_hidden_2), shape=(n_classes, n_hidden_2))
  arg_params['fc1_bias'] = mx.nd.random_normal(scale=1e-10, shape=(n_hidden_1,))
  arg_params['fc2_bias'] = mx.nd.random_normal(scale=1e-10, shape=(n_hidden_2,))
  arg_params['fc3_bias'] = mx.nd.random_normal(scale=1e-10, shape=(n_classes,))


  mod = mx.mod.Module(train_net, logger=logger, context=ctx)
  mod.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)
  mod.set_params(arg_params, None)
  mod.init_optimizer(kvstore=kv, optimizer='sgd', optimizer_params={'learning_rate':learning_rate})
  metric = mx.metric.create(['ce', 'acc'])

  mod_pred = mx.mod.Module(pred_net, logger=logger, context=ctx)
  mod_pred.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label, shared_module=mod)

  for i_epoch in range(num_epoch):
    train_dataiter.reset()
    for i_iter, batch in enumerate(train_dataiter):
      batch.data[0] = mx.nd.array(Scaler.transform(batch.data[0].asnumpy()))
      mod.forward(batch)
      mod.backward()
      mod.update()
      mod.update_metric(metric, batch.label)

    if (i_epoch % display_step == 0):
      metric.reset()
      mod_pred.reshape(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)
      mod_pred.forward(batch, is_train=False)
      mod_pred.update_metric(metric, batch.label)
      for name, val in metric.get_name_value():
        print "[Epoch:%d] train-%s: %f"  % (i_epoch, name, val)

      metric.reset()
      mod_pred.reshape(data_shapes=val_dataiter.provide_data, label_shapes=val_dataiter.provide_label)
      val_batch = val_dataiter.next()
      val_batch.data[0] = mx.nd.array(Scaler.transform(val_batch.data[0].asnumpy()))
      mod_pred.forward(val_batch, is_train=False)
      mod_pred.update_metric(metric, val_batch.label)
      val_dataiter.reset()
      for name, val in metric.get_name_value():
        print "[Epoch:%d] val-%s %f"  % (i_epoch, name, val)


if __name__ == '__main__':
   gpus = "0"
   ctx = mx.cpu() if gpus is None else [mx.gpu(int(i)) for i in gpus.split(',')]
   train(ctx)

