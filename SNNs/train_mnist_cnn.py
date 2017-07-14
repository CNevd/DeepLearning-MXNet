import os,sys
import math
import mxnet as mx
import numpy as np
import logging
logging.basicConfig()
from sklearn.preprocessing import StandardScaler


# Parameters
learning_rate = 0.025
batch_size = 128
training_iters = 50
dropoutRate_SNN = 0.05
dropoutRate_ReLU = 0.5
display_step = 1

# Network Parameters
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

def conv_net_SNN(is_train=True):
  data = mx.sym.Variable('data')
  
  data = mx.sym.reshape(data=data, shape=(-1, 1, 28, 28))
  conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=32, name='conv1')
  conv1 = mx.sym.Pooling(data=conv1, kernel=(2, 2), pool_type='max', stride=(2,2))
  conv2 = mx.sym.Convolution(data=conv1, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=64, name='conv2')
  conv2 = mx.sym.Pooling(data=conv2, kernel=(2, 2), pool_type='max', stride=(2,2))
  conv2 = mx.sym.flatten(data=conv2)
  fc = mx.sym.FullyConnected(data=conv2, num_hidden=1024, name='fc1')
  fc = selu(fc)
  if (is_train):
    fc = dropout_selu(fc, (batch_size, 1024), dropoutRate_SNN)
  layer_out = mx.sym.FullyConnected(data=fc, num_hidden=n_classes, name='fc2')
  out = mx.sym.SoftmaxOutput(data=layer_out, name='softmax')
  return out

def conv_net_ReLU():
  data = mx.sym.Variable('data')

  data = mx.sym.reshape(data=data, shape=(-1, 1, 28, 28))
  conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=32, name='conv1_relu')
  conv1 = mx.sym.Pooling(data=conv1, kernel=(2, 2), pool_type='max', stride=(2,2))
  conv2 = mx.sym.Convolution(data=conv1, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=64, name='conv2_relu')
  conv2 = mx.sym.Pooling(data=conv2, kernel=(2, 2), pool_type='max', stride=(2,2))
  conv2 = mx.sym.flatten(data=conv2)
  fc = mx.sym.FullyConnected(data=conv2, num_hidden=1024, name='fc1_relu')
  fc = mx.sym.Activation(data=fc, act_type='relu')
  fc = mx.sym.Dropout(data=fc, p=dropoutRate_ReLU)
  layer_out = mx.sym.FullyConnected(data=fc, num_hidden=n_classes, name='fc2_relu')
  out = mx.sym.SoftmaxOutput(data=layer_out, name='softmax')
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

  print "Loading data..."
  basedir = os.path.dirname(__file__)
  train_dataiter = mx.io.MNISTIter(
          image=os.path.join(basedir, "mnist_data", "train-images-idx3-ubyte"),
          label=os.path.join(basedir, "mnist_data", "train-labels-idx1-ubyte"),
          data_shape=(784,),
          batch_size=batch_size, shuffle=True, flat=True, silent=False)
  val_dataiter = mx.io.MNISTIter(
          image=os.path.join(basedir, "mnist_data", "t10k-images-idx3-ubyte"),
          label=os.path.join(basedir, "mnist_data", "t10k-labels-idx1-ubyte"),
          data_shape=(784,),
          batch_size=batch_size, shuffle=True, flat=True, silent=False)

  
  # Normalize Data to mean = 0, stdev = 1
  Scaler = get_scaler(train_dataiter)

  # get symbol
  train_ReLU = conv_net_ReLU()
  train_SNN = conv_net_SNN()
  pred_SNN = conv_net_SNN(False)

  # initialization for SNN
  arg_names = train_SNN.list_arguments()
  arg_shape, out_shape, aux_shape = train_SNN.infer_shape(data = (batch_size, 784))
  arg_shape_dict1 = dict(zip(arg_names, arg_shape))
  print "SNN: ",arg_shape_dict1

  arg_params1 = {}
  arg_params1['conv1_weight'] = mx.nd.random_normal(scale=np.sqrt(1.0/25), shape=arg_shape_dict1['conv1_weight'])
  arg_params1['conv2_weight'] = mx.nd.random_normal(scale=np.sqrt(1.0/(25*32)), shape=arg_shape_dict1['conv2_weight'])
  arg_params1['fc1_weight'] = mx.nd.random_normal(scale=np.sqrt(1.0/(7*7*64)), shape=arg_shape_dict1['fc1_weight'])
  arg_params1['fc2_weight'] = mx.nd.random_normal(scale=np.sqrt(1.0/1024), shape=arg_shape_dict1['fc2_weight'])
  arg_params1['conv1_bias'] =  mx.nd.random_normal(scale=1e-10, shape=arg_shape_dict1['conv1_bias'])
  arg_params1['conv2_bias'] =  mx.nd.random_normal(scale=1e-10, shape=arg_shape_dict1['conv2_bias'])
  arg_params1['fc1_bias'] = mx.nd.random_normal(scale=1e-10, shape=arg_shape_dict1['fc1_bias'])
  arg_params1['fc2_bias'] = mx.nd.random_normal(scale=1e-10, shape=arg_shape_dict1['fc2_bias'])

  # initialization for ReLU
  arg_names = train_ReLU.list_arguments()
  arg_shape, out_shape, aux_shape = train_ReLU.infer_shape(data = (batch_size, 784))
  arg_shape_dict2 = dict(zip(arg_names, arg_shape))
  print "ReLU: ",arg_shape_dict2

  arg_params2 = {}
  arg_params2['conv1_relu_weight'] = mx.nd.random_normal(scale=np.sqrt(2.0/25), shape=arg_shape_dict2['conv1_relu_weight'])
  arg_params2['conv2_relu_weight'] = mx.nd.random_normal(scale=np.sqrt(2.0/(25*32)), shape=arg_shape_dict2['conv2_relu_weight'])
  arg_params2['fc1_relu_weight'] = mx.nd.random_normal(scale=np.sqrt(2.0/(7*7*64)), shape=arg_shape_dict2['fc1_relu_weight'])
  arg_params2['fc2_relu_weight'] = mx.nd.random_normal(scale=np.sqrt(2.0/1024), shape=arg_shape_dict2['fc2_relu_weight'])
  arg_params2['conv1_relu_bias'] =  mx.nd.random_normal(scale=1e-6, shape=arg_shape_dict2['conv1_relu_bias'])
  arg_params2['conv2_relu_bias'] =  mx.nd.random_normal(scale=1e-6, shape=arg_shape_dict2['conv2_relu_bias'])
  arg_params2['fc1_relu_bias'] = mx.nd.random_normal(scale=1e-6, shape=arg_shape_dict2['fc1_relu_bias'])
  arg_params2['fc2_relu_bias'] = mx.nd.random_normal(scale=1e-6, shape=arg_shape_dict2['fc2_relu_bias'])

  # create kvstore and metric
  kv = None
  metric_SELU = mx.metric.create(['ce', 'acc'])
  metric_ReLU = mx.metric.create(['ce', 'acc'])

  # module for snn
  mod = mx.mod.Module(train_SNN, logger=logger, context=ctx)
  mod.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)
  mod.set_params(arg_params1, None)
  mod.init_optimizer(kvstore=kv, optimizer='sgd', optimizer_params={'learning_rate':learning_rate})

  mod_pred = mx.mod.Module(pred_SNN, logger=logger, context=ctx)
  mod_pred.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label, shared_module=mod)

  # module for relu
  mod_ReLU = mx.mod.Module(train_ReLU, logger=logger, context=ctx)
  mod_ReLU.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)
  mod_ReLU.set_params(arg_params2, None)
  mod_ReLU.init_optimizer(kvstore=kv, optimizer='sgd', optimizer_params={'learning_rate':learning_rate})

  step = 0
  train_dataiter.reset()
  while(step < training_iters):
    batch = train_dataiter.next()

    # ReLU
    mod_ReLU.forward(batch)
    mod_ReLU.backward()
    mod_ReLU.update()
    mod_ReLU.update_metric(metric_ReLU, batch.label)
    if (step % display_step == 0):
      # ReLU
      metric_ReLU.reset()
      mod_ReLU.forward(batch, is_train=False)
      mod_ReLU.update_metric(metric_ReLU, batch.label)
      for name, val in metric_ReLU.get_name_value():
        print "[Step:%d] ReLU train-%s: %f"  % (step, name, val)

    # SELU
    batch.data[0] = mx.nd.array(Scaler.transform(batch.data[0].asnumpy()))
    mod.forward(batch)
    mod.backward()
    mod.update()
    mod.update_metric(metric_SELU, batch.label)
    if (step % display_step == 0):
      metric_SELU.reset()
      mod_pred.forward(batch, is_train=False)
      mod_pred.update_metric(metric_SELU, batch.label)
      for name, val in metric_SELU.get_name_value():
        print "[Step:%d] SELU train-%s: %f"  % (step, name, val)

    step += 1

  val_dataiter.reset()
  metric_SELU.reset()
  metric_ReLU.reset()
  for val_batch in val_dataiter:
    # ReLU
    mod_ReLU.forward(val_batch, is_train=False)
    mod_ReLU.update_metric(metric_ReLU, val_batch.label)
    # SELU
    val_batch.data[0] = mx.nd.array(Scaler.transform(val_batch.data[0].asnumpy()))
    mod_pred.forward(val_batch, is_train=False)
    mod_pred.update_metric(metric_SELU, val_batch.label)

  for name, val in metric_ReLU.get_name_value():
      print "ReLU val-%s: %f"  % (name, val)
  for name, val in metric_SELU.get_name_value():
      print "SELU val-%s: %f"  % (name, val)


if __name__ == '__main__':
   gpus = "0" # change to None to use cpu
   ctx = mx.cpu() if gpus is None else [mx.gpu(int(i)) for i in gpus.split(',')]
   train(ctx)

