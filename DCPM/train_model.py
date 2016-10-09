import mxnet as mx
import logging
import os
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

def AUC(label, pred):
    """ Custom evaluation metric on AUC.
    """
    auc = roc_auc_score(label, pred)
    return auc

def RMSE(label, pred):
    """ Custom evaluation metric on rmse.
    """
    rmse = math.sqrt(mean_squared_error(label, pred))
    return rmse

def fit(args, network, train, val, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
 
    # load model
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # train
    devs = mx.gpu(args.gpus) if args.gpus >= 0 else mx.cpu()

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient
    
    arg_names = network.list_arguments()
    print arg_names

    # disable kvstore for single device
    if 'local' in kv.type:
        kv = None

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        #arg_params         = arg_params,
        momentum           = 0.9,
        wd                 = 0.,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

    eval_metrics = ['rmse']

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))

    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = mx.metric.np(AUC),
        kvstore            = kv,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)
