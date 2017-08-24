import sys,os
import numpy as np
import mxnet as mx


class DataIter(mx.io.DataIter):
    def __init__(self, file_path, batch_size, usr_num, doc_dim):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.usr_num = usr_num
        self.file_path = file_path
        self.files = os.listdir(self.file_path)
        self.index = 0
        self.provide_data = [('data_usr',(self.batch_size, self.usr_num)),
                             ('doc_pos',(self.batch_size, doc_dim)),
                             ('doc_neg',(self.batch_size, doc_dim))]
        self.label = [mx.nd.zeros(shape=(self.batch_size,))]
        self.provide_label = [('mae_label',(self.batch_size,))]
        # Load news vector
        self.news_vec = np.load('./dssm_data/dssm/s_titledata.npy')

    def __iter__(self):
        for f in self.files:
            fname = os.path.join(self.file_path, f)
            data = np.load(fname)
            self.index_start = 0
            size = data.shape[0]
            idx = np.arange(size)
            np.random.shuffle(idx)
            while(self.index_start + self.batch_size < size):
                index_end = self.index_start + self.batch_size
                uid = data[self.index_start : index_end, 0].astype(int)
                usr_data = mx.nd.csr_matrix(np.ones(self.batch_size), np.arange(self.batch_size+1), uid, (self.batch_size, self.usr_num))
                doc_pos = self.news_vec[data[self.index_start : index_end, 2]]
                doc_neg = self.news_vec[data[self.index_start : index_end, 3]]
                data_all = [usr_data, mx.nd.array(doc_pos), mx.nd.array(doc_neg)]
                data_names = ['data_usr', 'doc_pos', 'doc_neg']
                self.index_start = index_end
                data_batch = mx.io.DataBatch(data_all, self.label,
                                             provide_data=self.provide_data,
                                             provide_label=self.provide_label)
                yield data_batch

    def reset(self):
        pass
