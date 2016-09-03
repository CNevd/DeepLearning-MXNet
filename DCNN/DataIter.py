import os, sys
import numpy
try:
    import cPickle as pickle
except ImportError:
    import pickle
import mxnet as mx

def load_data(pkl_path, batchSize):
    """
    load the sentiment dataset, either Stanford or Twitter

    Return:
    - train
    - dev
    - test
    - word2index
    - index2word
    - pretrained embedding
    """
    datasets = pickle.load(open(pkl_path, 'r'))
    train_data, train_label = datasets[0]
    dev_data, dev_label = datasets[1]
    test_data, test_label = datasets[2]
    word2index = datasets[3]
    index2word = datasets[4]
    pretrained_embeddings = datasets[5]
    sentence_size = len(train_data[0])
    vocab_size = len(word2index)
    train_iter = mx.io.NDArrayIter(data=mx.nd.array(train_data),
                                   label=mx.nd.array(train_label),
                                   batch_size=batchSize,
                                   shuffle=True,
                                   last_batch_handle='roll_over')
    val_iter = mx.io.NDArrayIter(data=mx.nd.array(test_data),
                                 label=mx.nd.array(test_label),
                                 batch_size=batchSize)
    return train_iter, val_iter, sentence_size, vocab_size

# load matlab data
def read_and_sort_matlab_data(x_file, y_file, batchSize, padding_value=15448):
    '''
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
    sentence_size = len(data_np[0])
    data_iter = mx.io.NDArrayIter(data=mx.nd.array(data_np),
                                   label=mx.nd.array(label_np),
                                   batch_size=batchSize,
                                   shuffle=True,
                                   last_batch_handle='roll_over')
    return data_iter, sentence_size, padding_value



#if __name__ == '__main__':
#  load_data('./data/twitter.pkl', 1)
#  read_and_sort_matlab_data('./data/binarySentiment/train.txt', './data/binarySentiment/train_lbl.txt', 1)

