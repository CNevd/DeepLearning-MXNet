import sys
import linecache
import numpy as np
import mxnet as mx

def get_data(file_path):
  data_in = np.loadtxt(file_path)
  label = mx.nd.array(data_in[:,0])
  data = mx.nd.array(data_in[:,1:])
  return data,label

def get_iterator(batchSize):
  train_data,train_label = get_data('./data/train_fm')
  val_data,val_label = get_data('./data/test_fm')
  train_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batchSize, shuffle=False, last_batch_handle='roll_over')
  val_iter = mx.io.NDArrayIter(data=val_data, label=val_label, batch_size=11150)
  return train_iter,val_iter


class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class myDataIter(mx.io.DataIter):
    def __init__(self, fname, batch_size):
        super(myDataIter, self).__init__()
        self.batch_size = batch_size
        self.fname = fname
        self.index_start = 1
        self.provide_data = [('data', (batch_size, 16))]
        self.provide_label = [('softmax_label', (batch_size, ))]

   
    def __iter__(self):
        while(True):
            bdata = []
            blabel = []
            if (not linecache.getline(self.fname, self.index_start + self.batch_size)):
                return
            for i in range(self.index_start, self.index_start + self.batch_size):
                line = linecache.getline(self.fname, i)
                line_label, line_data =  line.strip().split('\t',1)
                blabel.append(line_label)
                bdata.append(np.array(line_data.split('\t')))
            data_all = [mx.nd.array(bdata)]
            label_all = [mx.nd.array(blabel)]
            data_names = ['data']
            label_names = ['softmax_label']
            self.index_start += self.batch_size
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        self.index_start = 1
