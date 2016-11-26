import os, sys
import numpy
import mxnet as mx


# load matlab data
def read_and_sort_matlab_data(x_file, y_file, batchSize, padding_value=15448):
    '''
    load the Stanford sentiment dataset
    '''
    sorted_dict = {}
    x_data = []
    i=0
    file = open(x_file,"r")
    for line in file:
        words = line.split(",")
        result = []
        length=None
        for word in words:
            word_i = int(word)
            if word_i == padding_value and length==None:
                length = len(result)
            result.append(word_i)
        x_data.append(result)

        if length==None:
            length=len(result)

        if length in sorted_dict:
            sorted_dict[length].append(i)
        else:
            sorted_dict[length]=[i]
        i+=1

    file.close()

    file = open(y_file,"r")
    y_data = []
    for line in file:
        words = line.split(",")
        y_data.append(int(words[0])-1)
    file.close()

    new_train_list = []
    new_label_list = []
    lengths = []
    for length, indexes in sorted_dict.items():
        for index in indexes:
            new_train_list.append(x_data[index])
            new_label_list.append(y_data[index])
            lengths.append(length)
    data_np = numpy.asarray(new_train_list,dtype=numpy.int32)
    label_np = numpy.asarray(new_label_list,dtype=numpy.int32)

    return data_np, label_np, lengths


class Batch(object):
    def __init__(self, data_names, data,
                 label_names, label,
                 bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DataIter(mx.io.DataIter):
    def __init__(self, x_file, y_file, batch_size,
                 data_name='data', label_name='softmax_label'):
        super(DataIter, self).__init__()
        self.data_name = data_name
        self.label_name = label_name
        self.batch_size = batch_size
        self.data, self.label, self.lengths = read_and_sort_matlab_data(x_file, y_file, batch_size, padding_value=15448)
        self.n_batches = len(self.lengths) / self.batch_size
        self.permutation = numpy.random.permutation(self.n_batches)
        self.default_bucket_key = self.lengths[-1]
        self.provide_data = [(data_name, (batch_size, self.default_bucket_key))]
        self.provide_label = [(label_name, (batch_size, ))]

    def __iter__(self):
        data_names = [self.data_name]
        label_names = [self.label_name]

        for index in self.permutation:
            seq_len = self.lengths[(index + 1) * self.batch_size - 1]
            bdata = self.data[index * self.batch_size : (index+1) * self.batch_size, 0 : seq_len]
            blabel = self.label[index * self.batch_size : (index + 1) * self.batch_size]
            data_all = [mx.nd.array(bdata)]
            label_all = [mx.nd.array(blabel)]
            data_names = [self.data_name]
            label_names = [self.label_name]
            data_batch = Batch(data_names, data_all, label_names, label_all, seq_len)
            yield data_batch

    def reset(self):
        self.permutation = numpy.random.permutation(self.n_batches)
