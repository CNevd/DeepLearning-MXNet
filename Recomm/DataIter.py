import sys
import linecache
import mxnet as mx

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

class DataIter(mx.io.DataIter):
    def __init__(self, fname, batch_size):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.fname = fname
        self.index_start = 1
        self.provide_data = [('user', (batch_size, )), ('item', (batch_size, ))]
        self.provide_label = [('rate', (batch_size, ))]

    def __iter__(self):
        while(True):
            buser = []
            bitem = []
            brate = []
            if (not linecache.getline(self.fname, self.index_start + self.batch_size)):
                return
            for i in range(self.index_start, self.index_start + self.batch_size):
                line = linecache.getline(self.fname, i)
                lines = line.strip().split('::')
                if(len(lines) != 4):
                    continue
                line_user, line_item, line_rate, _ = lines
                buser.append(line_user)
                bitem.append(line_item)
                brate.append(line_rate)
            data_all = [mx.nd.array(buser), mx.nd.array(bitem)]
            label_all = [mx.nd.array(brate)]
            data_names = ['user', 'item']
            label_names = ['rate']
            self.index_start += self.batch_size
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        self.index_start = 1
