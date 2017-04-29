import os, sys
import mxnet as mx
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from DataIter import get_iterator, DataIter
from sklearn.metrics import roc_auc_score

# global num of categorical features
NUM_FEATURE = 133465

def AUC(label, pred):
    """ Custom evaluation metric on AUC.
    """
    auc = roc_auc_score(label, pred)
    return auc

def get_dcpm(input_dim, embbed_size, need_embedding=True):
    data = mx.symbol.Variable('data')
    if need_embedding:
        embed_weight=mx.symbol.Variable("embed_weight")
        emb = mx.symbol.Embedding(data=data, weight = embed_weight, input_dim = input_dim, output_dim = embbed_size)
        flatten = mx.symbol.Flatten(emb, name = "emb_flatten")
    else:
        flatten = data
    fc1  = mx.symbol.FullyConnected(data = flatten, name='fc1', num_hidden = 300)
    #bn1 = mx.symbol.BatchNorm(data=fc1, name="bn1")
    act1 = mx.symbol.Activation(data = fc1, name='act1', act_type='tanh')
    dp1 = mx.symbol.Dropout(data = act1, p=0.05)
    fc2  = mx.symbol.FullyConnected(data = dp1, name = 'fc2', num_hidden = 100)
    bn2 = mx.symbol.BatchNorm(data=fc2, name="bn2")
    act2 = mx.symbol.Activation(data = bn2, name='act2', act_type='tanh')
    dp2 = mx.symbol.Dropout(data = act2, p=0.05)
    fc3  = mx.symbol.FullyConnected(data = dp2, name='fc3', num_hidden = 1)
    dcpm  = mx.symbol.LogisticRegressionOutput(data = fc3, name = 'softmax')
    group = mx.symbol.Group([dcpm, flatten])
    return dcpm

def train_dcpm(args, ctx):
    # setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    kv = mx.kvstore.create(args.kv_store)
    # load data
    print "Loading data"
    train_iter = DataIter(args.data_dir + '/train_fm', args.batch_size)
    val_iter = DataIter(args.data_dir + '/test_fm', args.batch_size)
    data_names = [k[0] for k in train_iter.provide_data]
    label_names = [k[0] for k in train_iter.provide_label]

    # load symbol
    dcpm = get_dcpm(NUM_FEATURE, args.embed_size, args.need_embedding)

    # initialization
    arg_params = {}
    arg_names = dcpm.list_arguments()
    arg_shape, out_shape, aux_shape = dcpm.infer_shape(data = (args.batch_size, 16))
    arg_shape_dict = dict(zip(arg_names, arg_shape))
    print out_shape

    optimizer_params = {'wd': 0.01,
                        'learning_rate': 0.1}
    mod = mx.module.module.Module(dcpm, data_names=data_names, label_names=label_names,
                           logger=logger, context=ctx, work_load_list=args.work_load_list)

    batch_end_callback = mx.callback.Speedometer(args.batch_size, frequent=args.frequent)
    model_prefix = args.prefix + "-%d" % (kv.rank)
    epoch_end_callback = mx.callback.do_checkpoint(model_prefix, period=10)
    eval_metric = mx.metric.CompositeEvalMetric()
    metric_rmse = mx.metric.create('rmse')
    metric_auc = mx.metric.np(AUC)
    eval_metric.add(metric_rmse)
    eval_metric.add(metric_auc)

    # start train
    print "start training..."
    mod.fit(train_iter, val_iter, eval_metric=eval_metric,
            epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback,
            kvstore=kv,
            optimizer='adam', optimizer_params=optimizer_params,
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            arg_params=arg_params,
            allow_missing=True,
            begin_epoch=args.begin_epoch, num_epoch=args.num_epoch, validation_metric=eval_metric)

    print "Train done for epoch: %s"%args.num_epoch

def predict_dcpm(args, ctx):
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Deep Click Prediction Model')
    parser.add_argument('--data_dir', dest='data_dir', help='data path',
                        default="./data", type=str)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'dcpm'), type=str)
    parser.add_argument('--gpus', help='GPU device to train with',
                        default=None, type=str)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', help='num epoch of training',
                        default=50, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=5, type=int)
    parser.add_argument('--kv_store', dest='kv_store', help='the kv-store type',
                        default='local', type=str)
    parser.add_argument('--work_load_list', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size',
                        default=2000, type=int)
    parser.add_argument('--need_embedding', dest='need_embedding', help='wether use embedding for features',
                        default=True, type=bool)
    parser.add_argument('--embed_size', dest='embed_size', help='embedding size of one feature, must be even',
                        default=11, type=int)
    args = parser.parse_args()
    logging.info('arguments %s', args)
    return args

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    ctx = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_dcpm(args, ctx)
    #predict_dcpm(args, ctx)
