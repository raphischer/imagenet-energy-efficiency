import math
import os
import argparse
import inspect

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from autoaugment import ImageNetPolicy


PRETRAINED_PREPR = {n.replace('_', ''): e.preprocess_input for n, e in tf.keras.applications.__dict__.items() if inspect.ismodule(e) and hasattr(e, 'preprocess_input')}
PRETRAINED_PREPR['resnet101'] = PRETRAINED_PREPR['resnet']
PRETRAINED_PREPR['resnet152'] = PRETRAINED_PREPR['resnet']
PRETRAINED_PREPR['mobilenetv3small'] = PRETRAINED_PREPR['mobilenetv3']
PRETRAINED_PREPR['mobilenetv3large'] = PRETRAINED_PREPR['mobilenetv3']


def resize_with_crop(image, label, model_name):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 224, 224)
    i = PRETRAINED_PREPR[model_name](i)
    return (i, label)


def resize_with_crop_and_normalize(image, label, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 224, 224)
    i = (i - tf.convert_to_tensor(mean)) / tf.convert_to_tensor(std)
    return i, label


def tf_random_resized_crop(img, crop_size, interpolation):
    ratio = np.random.uniform(0.75, 1.3333333333333333)
    w = tf.cast(tf.shape(img)[0], float)
    h = tf.cast(tf.shape(img)[1], float)
    d = tf.cast(tf.shape(img)[2], tf.int32)
    area = tf.random.uniform(shape=(), minval=tf.cast(0.08 * w * h, tf.int32), maxval=tf.cast(w*h, tf.int32), dtype=tf.int32)
    c_w = tf.sqrt(tf.cast(area, float) * ratio)
    c_w = tf.math.minimum(c_w, w)
    c_h = tf.cast(area, float) / c_w
    c_h = tf.math.minimum(c_h, h)
    crop = tf.image.random_crop(img, (tf.cast(c_w, tf.int32), tf.cast(c_h, tf.int32), d))
    return tf.image.resize(crop, [crop_size, crop_size], interpolation)


def tf_random_horizontal_flip(img, hflip_prob):
    if np.random.uniform() < hflip_prob:
        return tf.image.flip_up_down(img)
    return img


def tf_autoaugment(img_in):
    img = tf.keras.preprocessing.image.array_to_img(img_in, scale=False)
    policy = ImageNetPolicy()
    img = policy(img)
    return tf.convert_to_tensor(tf.keras.preprocessing.image.img_to_array(img))


def preprocessing_preset(img, label, crop_size, interpolation='bilinear', auto_augment_policy=None, random_erase_prob=0.0, hflip_prob=0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # inspired by https://github.com/pytorch/vision/blob/main/references/classification/presets.py
    img = tf_random_resized_crop(img, crop_size, interpolation)
    # transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0:
        img = tf_random_horizontal_flip(img, hflip_prob)
        # trans.append(transforms.RandomHorizontalFlip(hflip_prob))
    if auto_augment_policy is not None:
        if not auto_augment_policy == 'imagenet': # only auto_augment_policy = 'imagenet' is used!
            raise NotImplementedError(f'AutoAugment policy {auto_augment_policy} not implemented!')
        img = tf.py_function(func=tf_autoaugment, inp=[img], Tout=tf.float32)
        # if auto_augment_policy == "ra":
        #     trans.append(autoaugment.RandAugment(interpolation=interpolation))
        # elif auto_augment_policy == "ta_wide":
        #     trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
        # else:
        #     aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
        #     trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
    # this does not work for normalization
    # img = tf.keras.layers.experimental.preprocessing.Normalization(mean=mean, variance=np.square(std))(img)
    img = (img - tf.convert_to_tensor(mean)) / tf.convert_to_tensor(std)
            # transforms.PILToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            # transforms.Normalize(mean=mean, std=std),
    if random_erase_prob > 0:
        img = random_erasing(img, random_erase_prob, sh=0.33)
        # trans.append(transforms.RandomErasing(p=random_erase_prob))

    return img, label


def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    if not img.dtype == tf.float32:
        img = tf.cast(img, tf.float32)

    def erase(e_img):

        img_width    = tf.shape(e_img)[1]
        img_height   = tf.shape(e_img)[0]
        img_channels = tf.shape(e_img)[2]

        area = tf.cast(img_height * img_width, float)

        target_area = tf.random.uniform([], minval=sl, maxval=sh) * area
        aspect_ratio = tf.random.uniform([], minval=r1, maxval=1/r1)

        w = tf.cast(tf.sqrt(tf.cast(target_area, float) * aspect_ratio), tf.int32)
        w = tf.math.minimum(w, img_width)
        h = tf.cast(tf.cast(target_area, tf.int32) / w, tf.int32)
        h = tf.math.minimum(h, img_height)

        x1 = tf.cond(img_height == h, lambda:0, lambda:tf.random.uniform([], minval=0, maxval=img_height - h, dtype=tf.int32))
        y1 = tf.cond(img_width  == w, lambda:0, lambda:tf.random.uniform([], minval=0, maxval=img_width - w, dtype=tf.int32))
        
        part1 = tf.slice(e_img, [0,0,0], [x1,img_width,img_channels]) # first row
        part2 = tf.slice(e_img, [x1,0,0], [h,y1,img_channels]) # second row 1

        # other methods not used in https://github.com/pytorch/vision/tree/main/references/classification
        part3 = tf.random.uniform((h,w,img_channels), dtype=tf.float32) # second row 2
        part3 -= part3 # this hack is needed bc tf.zeros uses not supported numpy code internally
        
        part4 = tf.slice(e_img, [x1,y1+w,0], [h,img_width-y1-w,img_channels]) # second row 3
        part5 = tf.slice(e_img, [x1+h,0,0], [img_height-x1-h,img_width,img_channels]) # third row

        middle_row = tf.concat([part2,part3,part4], axis=1)
        return tf.concat([part1,middle_row,part5], axis=0)

    return tf.cond(tf.random.uniform([]) > probability, lambda: tf.identity(img), lambda: erase(img))


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


        # dataset = torchvision.datasets.ImageFolder(
        #     traindir,
        #     presets.ClassificationPresetTrain(
        #         crop_size=train_crop_size,
        #         interpolation=interpolation,
        #         auto_augment_policy=auto_augment_policy,
        #         random_erase_prob=random_erase_prob,
        #     ),
        # )

    # preprocess the images for mobilenet

    x, y = next(iter(ds))
    # x2 = tf_random_resized_crop(x, 224, 'bilinear')
    # x3 = tf_random_horizontal_flip(x, .3)
    # x4 = tf.keras.preprocessing.image.img_to_array(ImageNetPolicy()(tf.keras.preprocessing.image.array_to_img(x)))
    # x5 = (tf.cast(x, float) - tf.convert_to_tensor(mean)) / tf.convert_to_tensor(std)
    # x6 = random_erasing(x, .3, sh=0.33, method='black')

    x4 = tf.py_function(func=tf_autoaugment, inp=[x], Tout=tf.float32)
    if map_f is not None:
        ds = ds.map(map_f)

    info.split = split
    info.batch_size = batch_size
    info.steps_per_epoch = n_batches if n_batches > 0 else math.ceil(info.splits[split].num_examples / batch_size)
    # batch the data
    if not batch_size is None:
        ds = ds.batch(batch_size)
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
