import argparse

import tensorflow as tf

from load_imagenet import load_imagenet, resize_with_crop


def train_model(tensorflow_model, ds, optim, loss, epochs, checkpoint, metrics):
    new_model = tensorflow_model(include_top=True, weights=None)
    new_model.trainable = True
    new_model.compile(optimizer=optim, loss=loss, metrics=metrics)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                     save_weights_only=True,
                                                     verbose=1)

    new_model.fit(ds, epochs=epochs, callbacks=[cp_callback])

    return new_model


def load_model(tensorflow_model, optim, loss, metrics, weights=None):
    pretrained_model = tensorflow_model(include_top=True, weights='imagenet')
    pretrained_model.trainable = False
    pretrained_model.compile(optimizer=optim, loss=loss, metrics=metrics)

    if weights is not None:
        pretrained_model.load_weights(weights)

    return pretrained_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Load Imagenet for Tensorflow")

    parser.add_argument("--data_dir", default="/raid/imagenet/", type=str, help="directory with downloaded 'ILSVRC2012_img_train.tar' and 'ILSVRC2012_img_val.tar'")
    parser.add_argument("--checkpoint", default="/raid/fischer/checkpoints/mobilenetv2test", type=str, help="location to store checkpoint")

    args = parser.parse_args()

    batch_size = 32
    metrics = ['accuracy']

    ds = load_imagenet(args.data_dir, None, 'train', resize_with_crop, batch_size)

    # train your own

    # parameters from https://github.com/pytorch/vision/tree/main/references/classification
    tensorflow_model = tf.keras.applications.MobileNetV2
    epochs = 300 
    lr = 0.045
    momentum = 0.9
    weight_decay = 0.00004
    optim = tf.keras.optimizers.SGD(
        learning_rate=lr,
        momentum=momentum,
        decay=weight_decay
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = train_model(tensorflow_model, ds, optim, loss, epochs, args.checkpoint, metrics)

    pretrained = load_model(tensorflow_model, optim, loss, metrics)
    custom = load_model(tensorflow_model, optim, loss, metrics, args.checkpoint)

    result = pretrained.evaluate(ds)
    print(dict(zip(pretrained.metrics_names, result)))

    result = custom.evaluate(ds)
    print(dict(zip(custom.metrics_names, result)))

    # pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, 
    #                                                      weights='imagenet')
    # pretrained_model.trainable = False
    # pretrained_model.compile(optimizer='adam', 
    #                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    #                          metrics=['accuracy'])

    # result = pretrained_model.evaluate(ds)
    # print(dict(zip(pretrained_model.metrics_names, result)))
