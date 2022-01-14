import numpy as np
import tensorflow as tf

from autoaugment import ImageNetPolicy


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