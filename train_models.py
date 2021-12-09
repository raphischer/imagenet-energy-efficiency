import tensorflow as tf

from load_imagenet import load_imagenet, resize_with_crop


def train_model(tensorflow_model, ds, optim, loss, epochs, checkpoint_dir, metrics):
    new_model = tensorflow_model(include_top=True, weights=None)
    new_model.trainable = True
    new_model.compile(optimizer=optim, loss=loss, metrics=metrics)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
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
    data_dir = '/home/fischer/mnt_imagenet/ecml2021_enerob'
    write_dir = '/home/fischer/mnt_imagenet/tf-imagenet-dirs'
    batch_size = 32
    metrics = ['accuracy']

    ds = load_imagenet(data_dir, write_dir, 'train', resize_with_crop, batch_size)

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
    checkpoint_dir = "/home/fischer/mnt_imagenet/mobilenetv2cp/"

    # model = train_model(tensorflow_model, ds, optim, loss, epochs, checkpoint_dir)

    pretrained = load_model(tensorflow_model, optim, loss, metrics)
    custom = load_model(tensorflow_model, optim, loss, metrics, checkpoint_dir)

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
