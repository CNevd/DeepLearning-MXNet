import os, sys
import mxnet as mx
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from DataIter import DataIter

USER_NUM = 6040
ITEM_NUM = 3952

def inference_svd(batch_size, user_num, item_num, dim=5):
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    rate = mx.symbol.Variable('rate')

    bias_user = mx.symbol.Embedding(data = user, input_dim = user_num, output_dim = 1, name = 'embed_bias_user')
    bias_user = mx.sym.Flatten(data = bias_user)
    bias_item = mx.symbol.Embedding(data = item, input_dim = item_num, output_dim = 1, name = 'embed_bias_item')
    bias_item = mx.sym.Flatten(data = bias_item)
    embd_user = mx.symbol.Embedding(data = user, input_dim = user_num, output_dim = dim, name = 'embed_user')
    embd_item = mx.symbol.Embedding(data = item, input_dim = item_num, output_dim = dim, name = 'embed_item')
    dot = mx.sym.Flatten(data = (embd_user*embd_item))
    infer = mx.symbol.sum(data = dot, axis = 1, keepdims = True)
    infer = infer + bias_user + bias_item
    svd = mx.symbol.LinearRegressionOutput(data = infer, label = rate)
    group = mx.symbol.Group([infer, rate, dot, bias_user, svd])
    return svd

def train_svd(args, ctx):
    # setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    kv = mx.kvstore.create(args.kv_store)
    # load data
    print "Loading data"
    train_iter = DataIter(args.data_dir + '/ratings_train.dat', args.batch_size)
    val_iter = DataIter(args.data_dir + '/ratings_test.dat', args.batch_size)
    data_names = [k[0] for k in train_iter.provide_data]
    label_names = [k[0] for k in train_iter.provide_label]

    # load symbol
    svd = inference_svd(args.batch_size, USER_NUM, ITEM_NUM, dim = args.dim)

    # initialization
    arg_names = svd.list_arguments()
    input_shapes = {'user' : (args.batch_size,1),
                    'item' : (args.batch_size,1),
                    'rate' : (args.batch_size,1)}
    arg_shape, out_shape, aux_shape = svd.infer_shape(**input_shapes)
    arg_shape_dict = dict(zip(arg_names, arg_shape))
    arg_params = {}
    #arg_params = {'embed_user_weight' : mx.random.normal(0, 0.02, shape = arg_shape_dict['embed_user_weight']),
    #              'embed_item_weight' : mx.random.normal(0, 0.02, shape = arg_shape_dict['embed_item_weight'])}

    optimizer_params = {'wd': 0.05,
                        'learning_rate': 0.001}
    mod = mx.module.module.Module(svd, data_names=data_names, label_names=label_names,
                           logger=logger, context=ctx, work_load_list=args.work_load_list)

    batch_end_callback = mx.callback.Speedometer(args.batch_size, frequent=args.frequent)
    model_prefix = args.prefix + "-%d" % (kv.rank)
    epoch_end_callback = mx.callback.do_checkpoint(model_prefix, period=30)
    eval_metric = mx.metric.CompositeEvalMetric()
    metric_rmse = mx.metric.create('rmse')
    eval_metric.add(metric_rmse)

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
            begin_epoch=args.begin_epoch, num_epoch=args.num_epoch,
            validation_metric=eval_metric)

    print "Train done for epoch: %s"%args.num_epoch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a SVD Model')
    parser.add_argument('--data_dir', dest='data_dir', help='data path',
                        default="./data/movielens/ml-1m", type=str)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'dcpm'), type=str)
    parser.add_argument('--gpus', help='GPU device to train with',
                        default=None, type=str)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', help='num epoch of training',
                        default=100, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=100, type=int)
    parser.add_argument('--kv_store', dest='kv_store', help='the kv-store type',
                        default='local', type=str)
    parser.add_argument('--work_load_list', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size',
                        default=1000, type=int)
    parser.add_argument('--dim', dest='dim', help='embedding size of user and item',
                        default=15, type=int)
    args = parser.parse_args()
    logging.info('arguments %s', args)
    return args

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    ctx = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_svd(args, ctx)

