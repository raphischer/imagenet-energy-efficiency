import math
import os
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds


NUM_SAMPLES = {
    'train': 1281167,
    'validation': 50000
}


def load_imagenet(data_dir, write_dir=None, split='train', map_f=None, batch_size=1, n_batches=-1, variant='imagenet2012'):
    assert(variant in ['imagenet2012_subset', 'imagenet2012'])
    assert(split in ['train', 'validation'])
    if write_dir is None:
        write_dir = os.path.join(data_dir, split)
        data_dir = os.path.join(data_dir, 'raw')
        assert(os.path.isdir(data_dir))
        assert(os.path.isdir(write_dir))
    # Construct a tf.data.Dataset
    download_config = tfds.download.DownloadConfig(
        extract_dir=os.path.join(write_dir, 'extracted'),
        manual_dir=data_dir
    )
    download_and_prepare_kwargs = {
        'download_dir': os.path.join(write_dir, 'downloaded'),
        'download_config': download_config,
    }
    ds, info = tfds.load(variant,
                   data_dir=os.path.join(write_dir, 'data'),         
                   split=split,
                   shuffle_files=False,
                   download=True,
                   as_supervised=True,
                   with_info=True,
                   download_and_prepare_kwargs=download_and_prepare_kwargs
    )

    if map_f is not None:
        ds = ds.map(map_f)

    info.split = split
    info.batch_size = batch_size
    info.steps_per_epoch = n_batches if n_batches > 0 else math.ceil(info.splits[split].num_examples / batch_size)

    # batch the data
    if not batch_size is None:
        n_gpus = max(len(tf.config.list_physical_devices('GPU')), 1) # if no GPU is available, use given batch size
        ds = ds.batch(batch_size * n_gpus)
        if n_batches <= 0:
            info.steps_per_epoch = math.ceil(info.splits[split].num_examples / (batch_size * n_gpus))
    if n_batches > 0:
        info.steps_per_epoch 
        ds = ds.take(n_batches)

    return ds, info


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Load Imagenet for Tensorflow")

    parser.add_argument("--data-raw", default="/raid/imagenet/", type=str, help="directory with downloaded 'ILSVRC2012_img_train.tar' and 'ILSVRC2012_img_val.tar'")
    parser.add_argument("--split", default="train", type=str, choices=['train', 'validation'], help="data split to use")

    args = parser.parse_args()

    ds = load_imagenet(args.data_raw, None, args.split)
    print(ds)
