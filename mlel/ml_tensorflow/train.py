import os
from datetime import datetime

import tensorflow as tf

from mlel.ml_tensorflow.load_imagenet import load_imagenet
from mlel.ml_tensorflow.load_models import prepare_model, prepare_optimizer, prepare_lr_scheduling, load_preprocessing


class TimestampOnEpochEnd(tf.keras.callbacks.Callback):

    def __init__(self, path):
        super(TimestampOnEpochEnd, self).__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        timestamp = (datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
        if 'timestamp' not in self.model.history.history:
            self.model.history.history['timestamp'] = []
        self.model.history.history['timestamp'].append(timestamp)


def init_training(args):
    tf.random.set_seed(args.seed)

    # open strategy scope for using all GPUs
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        preproc_f = load_preprocessing(args.preprocessing, args.model, args)
        ds_train, ds_train_info = load_imagenet(args.data_path, None, 'train', preproc_f, args.batch_size, args.n_batches)
        ds_valid, _ = load_imagenet(args.data_path, None, 'validation', preproc_f, args.batch_size, args.n_batches)
        optimizer = prepare_optimizer(args.model, args.opt.lower(), args.opt_decy, args.lr, args.momentum, args.weight_decay, ds_train_info, args.epochs)

        if args.resume:
            model, mfile = prepare_model(args.model, optimizer, weights=args.resume)
            initial_epoch = int(mfile[11:14])
        else:
            model, _ = prepare_model(args.model, optimizer)
            initial_epoch = 0

    # create callbacks
    callbacks = [TimestampOnEpochEnd(os.path.join(args.output_dir, "epoch_timestamps.csv"))]
    for i in [10, 5, 2, 1]:
        if args.epochs % i == 0:
            save_freq = ds_train_info.steps_per_epoch * i # checkpoints every i epochs
            break
    callbacks.append([tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output_dir, 'checkpoint_{epoch:03d}.hdf5'), save_weights_only=True, save_freq=save_freq)])
    lr_callback = prepare_lr_scheduling(args.model, args.lr_scheduler, args.lr_gamma, args.lr_step_size, args.lr)
    if lr_callback is not None:
        callbacks.append(lr_callback)
    if args.early_delta > 0:
        print(f'Will early stop after {args.early_patience} epochs with less than {args.early_delta} validation accuracy improvement!')
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=args.early_delta, patience=args.early_patience, restore_best_weights=True))


    return lambda: model.fit(ds_train, epochs=args.epochs, callbacks=callbacks, initial_epoch=initial_epoch, validation_data=ds_valid)


def finalize_training(train_res, results, args):
    final_epoch = len(train_res.history['loss'])
    train_res.model.save_weights(os.path.join(args.output_dir, f'checkpoint_{final_epoch:03d}_final.hdf5'))

    history = {key.replace('categorical_', '').replace('top_k', 'top_5') : val for key, val in train_res.history.items()}
    results.update({
        'history': history,
        'model': {
            'params': train_res.model.count_params(),
            'fsize': os.path.getsize(os.path.join(args.output_dir, f'checkpoint_{final_epoch:03d}_final.hdf5'))
        }
    })
    return results