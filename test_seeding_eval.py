import tensorflow as tf
import numpy as np

from load_imagenet import load_imagenet, resize_with_crop

ds = load_imagenet('/raid/imagenet', None, 'train', resize_with_crop, batch_size=64).take(1)
x, y = iter(ds).next()

dirs = [
    '/raid/fischer/checkpoints/train_2021_12_15_13_31/vgg16',
    '/raid/fischer/checkpoints/train_2021_12_15_13_35/vgg16',
    '/raid/fischer/checkpoints/train_2021_12_15_14_38/vgg16',
    '/raid/fischer/checkpoints/train_2021_12_15_14_39/vgg16'
]

ms = []
for dir in dirs:
    ms.append(tf.keras.applications.VGG16(True, None))
    ms[-1].load_weights(dir)

ps = [m.predict(x) for m in ms]

print('0 - 1 MAX DIFF:', np.abs(ps[0] - ps[1]).max())
print('2 - 3 MAX DIFF:', np.abs(ps[2] - ps[3]).max())
print('0 - 2 MAX DIFF:', np.abs(ps[0] - ps[2]).max())
