import os

import tensorflow as tf
import onnxruntime as rt
import tf2onnx
from onnx_opcounter import calculate_params
import numpy as np
import tqdm

from mlel.ml_tensorflow.load_imagenet import load_imagenet
from mlel.ml_tensorflow.load_models import prepare_model, load_preprocessing, MODEL_CUSTOM_INPUT, calculate_flops

def _iterating_inference(model_path, output_names, ds):
    top1m = tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=1, name="sparse_top_k_categorical_accuracy", dtype=None
    )
    top5m = tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=5, name="sparse_top_k_categorical_accuracy", dtype=None
    )
    lossm = tf.keras.losses.SparseCategoricalCrossentropy()
    acc1 = np.zeros(len(ds))
    acc5 = np.zeros(len(ds))
    loss = np.zeros(len(ds))

    inf_model = rt.InferenceSession(model_path)
    for i, (x, y) in tqdm.tqdm(enumerate(ds.as_numpy_iterator()), total=len(ds)):
        result = inf_model.run(output_names, {'input': x})[0]
        top1m.update_state(y, result)
        top5m.update_state(y, result)
        acc1[i] = top1m.result().numpy()
        acc5[i] = top5m.result().numpy()
        loss[i] = lossm(y, result).numpy()
    return {"loss": np.mean(loss), "accuracy": np.mean(acc1), "top_5_accuracy": np.mean(acc5)}

def init_inference(args, split):
    tf.random.set_seed(args.seed)
    custom_trained = os.path.isdir(args.infer_model)

    # open strategy scope for using all GPUs
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
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
            model, _ = prepare_model(args.model, None, weights=args.infer_model)

    output_path = os.path.join(args.output_dir, 'model.onnx')
    crop_size = MODEL_CUSTOM_INPUT.get(args.model, (224, 224))
    spec = (tf.TensorSpec((args.batch_size, crop_size[0], crop_size[1], 3), tf.float32, name='input'), )
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    
    model_info = {
        'params': model.count_params(),
        'fsize': os.path.getsize(output_path),
        'flops': int(calculate_params(model_proto))
    }

    eval_func = lambda: _iterating_inference(output_path, output_names, dataset)
    return eval_func, model_info
    

def finalize_inference(results):
    return results
