import os

import tensorflow as tf

from mlel.ml_tensorflow.load_imagenet import load_imagenet
from mlel.ml_tensorflow.load_models import prepare_model, load_preprocessing, prepare_optimizer


def init_evaluation(args, split):
    tf.random.set_seed(args.seed)
    custom_trained = os.path.isdir(args.eval_model)

    # open strategy scope for using all GPUs
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        if not custom_trained or args.eval_preprocessing == 'builtin':
            preproc_f = load_preprocessing('builtin', args.model, args)
        elif args.eval_preprocessing == 'like-train':
            preproc_f = load_preprocessing(args.preprocessing, args.model, args)
        dataset, _ = load_imagenet(args.data_path, None, split, preproc_f, args.batch_size, args.n_batches)
        
        if not custom_trained:
            model, _ = prepare_model(args.model, None, weights='pretrained')
        else:
            # TODO check if using default (optimizer = None) makes a difference!
            # currently, this loads the optimizer from the training directory
            # optimizer = prepare_optimizer(args.model, args.opt.lower(), args.opt_decy, args.lr, args.momentum, args.weight_decay, ds_info, args.epochs)
            model, _ = prepare_model(args.model, None, weights=args.eval_model)
    
    model.save_weights(os.path.join(args.output_dir, 'eval_weights.hdf5'))
    model_info = {
        'params': model.count_params(),
        'fsize': os.path.getsize(os.path.join(args.output_dir, 'eval_weights.hdf5'))
    }
    eval_func = lambda: model.evaluate(dataset, return_dict=True)
    return eval_func, model_info
    

def finalize_evaluation(results):
    return {key.replace('categorical_', '').replace('top_k', 'top_5') : val for key, val in results.items()}
