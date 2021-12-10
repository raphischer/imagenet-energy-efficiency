import tensorflow as tf
import tensorflow_datasets as tfds

import os
import argparse


def load_imagenet(data_dir, write_dir=None, split='train', map_f=None, batch_size=None, variant='imagenet2012'):
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
    ds = tfds.load(variant,
                   data_dir=os.path.join(write_dir, 'data'),         
                   split=split, 
                   shuffle_files=False, 
                   download=True, 
                   as_supervised=True,
                   download_and_prepare_kwargs=download_and_prepare_kwargs
    )

    # preprocess the images for mobilenet
    if map_f is not None:
        ds = ds.map(map_f)

    if not batch_size is None:
        ds = ds.batch(batch_size)

    return ds


def resize_with_crop(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 224, 224)
    i = tf.keras.applications.mobilenet_v2.preprocess_input(i)
    return (i, label)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Load Imagenet for Tensorflow")

    parser.add_argument("--data-raw", default="/raid/imagenet/raw/", type=str, help="directory with downloaded 'ILSVRC2012_img_train.tar' and 'ILSVRC2012_img_val.tar'")
    parser.add_argument("--data-out", default="/raid/imagenet/validation/", type=str, help="directory where Tensorflow dataset is created")
    parser.add_argument("--split", default="train", type=str, choices=['train', 'validation'], help="data split to use")

    args = parser.parse_args()

    ds = load_imagenet(args.data_raw, args.data_out, args.split)
    print(ds)
