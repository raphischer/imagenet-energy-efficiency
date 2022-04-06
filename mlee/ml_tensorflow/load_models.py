import inspect
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import larq as lq
import larq_zoo as lqz
from larq_zoo.training.learning_schedules import CosineDecayWithWarmup

KERAS_BUILTINS = [e for e in tf.keras.applications.__dict__.values() if inspect.ismodule(e) and hasattr(e, 'preprocess_input')]
KERAS_MODELS = {n: e for mod in KERAS_BUILTINS for n, e in mod.__dict__.items() if callable(e) and n[0].isupper()}
KERAS_PREPR = {n: mod.preprocess_input for mod in KERAS_BUILTINS for n, e in mod.__dict__.items() if callable(e) and n[0].isupper()}
KERAS_MODELS['MobileNetV3Large'] = tf.keras.applications.MobileNetV3Large # manually adding MobileNetV3 since for some reason it is not callable from the respective submodule
KERAS_MODELS['MobileNetV3Small'] = tf.keras.applications.MobileNetV3Small
KERAS_PREPR['MobileNetV3Large'] = tf.keras.applications.mobilenet_v3.preprocess_input
KERAS_PREPR['MobileNetV3Small'] = tf.keras.applications.mobilenet_v3.preprocess_input

QUICKNETS = {n: e for n, e in lqz.sota.__dict__.items() if callable(e)}
LARQ_PREP = {n: lqz.preprocess_input for n in QUICKNETS.keys()}

MODELS = {**KERAS_MODELS, **QUICKNETS}
BUILTIN_PREPR = {**KERAS_PREPR, **LARQ_PREP}

INCEPTION_INPUT = {
    mname: (299, 299) for mname in MODELS if 'ception' in mname
}
EFFICIENT_INPUT = {
    'EfficientNetB1': (240, 240),
    'EfficientNetB2': (260, 260),
    'EfficientNetB3': (300, 300),
    'EfficientNetB4': (380, 380),
    'EfficientNetB5': (456, 456),
    'EfficientNetB6': (528, 528),
    'EfficientNetB7': (600, 600)
}
NASNET_INPUT = {
    'NASNetLarge': (331, 331)
}
MODEL_CUSTOM_INPUT = {**INCEPTION_INPUT, **EFFICIENT_INPUT, **NASNET_INPUT}


def prepare_optimizer(model_name, opt_name, opt_decy, lr, momentum, weight_decay, ds_info, epochs):
    if model_name in QUICKNETS.keys():
        binary_opt = tf.keras.optimizers.Adam(
            learning_rate=CosineDecayWithWarmup(
                max_learning_rate=1e-2,
                # steps_per_epoch = dataset.num_examples("train") // batch_size
                warmup_steps=ds_info.steps_per_epoch * 5,
                decay_steps=ds_info.steps_per_epoch * epochs,
            )
        )
        fp_opt = tf.keras.optimizers.SGD(
            learning_rate=CosineDecayWithWarmup(
                max_learning_rate=0.1,
                warmup_steps=ds_info.steps_per_epoch * 5,
                decay_steps=ds_info.steps_per_epoch * epochs,
            ),
            momentum=0.9,
        )
        optimizer = lq.optimizers.CaseOptimizer(
            (lq.optimizers.Bop.is_binary_variable, binary_opt),
            default_optimizer=fp_opt,
        )
    else:
        if opt_name.startswith("sgd"):
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=momentum,
                decay=weight_decay,
                nesterov="nesterov" in opt_name
            )
        elif opt_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(lr, rho=opt_decy, momentum=momentum, epsilon=0.0316, decay=weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {opt_name}. Only SGD and RMSprop are supported.")
    return optimizer


def steplr(epoch, lr, gamma, step_size, init_lr):
    return init_lr * math.pow(gamma, math.floor(epoch / step_size))


def prepare_lr_scheduling(model_name, lr_scheduler, lr_gamma, lr_step_size, init_lr):
    lr_scheduler = lr_scheduler.lower()
    if model_name in QUICKNETS.keys() or lr_scheduler == "none": # QuickNet has there own optimizer with LR scheduling
        return None
    if lr_scheduler == "steplr":
        main_lr_scheduler = lambda e, lr: steplr(e, lr, lr_gamma, lr_step_size, init_lr)
    # elif lr_scheduler == "cosineannealinglr":
    #     main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=args.epochs - args.lr_warmup_epochs
    #     )
    # elif lr_scheduler == "exponentiallr":
    #     main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler}'. Only StepLR supported at the moment."
        )

    return tf.keras.callbacks.LearningRateScheduler(main_lr_scheduler)


def prepare_model(model_name, optimizer, metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'], weights=None):
    mfile = None
    try:
        model = MODELS[model_name]
    except (TypeError, KeyError) as e:
        avail = ', '.join(n for n, _ in MODELS.items())
        raise RuntimeError(f'Error when loading {model_name}! \n{e}\nAvailable models:\n{avail}')
    if weights == 'pretrained':
        print(f'Loading pretrained weights!')
        weights = 'imagenet'
    elif weights is not None and os.path.isdir(weights): # load the custom weights
        best_model = sorted([f for f in os.listdir(weights) if f.startswith('checkpoint')])[-1]
        print(f'Loading weights from {best_model}!')
        weights = os.path.join(weights, best_model)
        mfile = weights
    model = model(include_top=True, weights=weights)

    # criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if optimizer is None: # use any default optimizer
        optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss=criterion, metrics=metrics)

    return model, mfile


def calculate_flops(model):

    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
    graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    flops = graph_info.total_float_ops // 2
    return flops


def load_preprocessing(preprocessing, model, args):
    if preprocessing == 'builtin':
        preproc_f = load_simple_prepr(model)
    elif preprocessing == 'custom':
        crop_size = MODEL_CUSTOM_INPUT.get(model, (224, 224))[0]
        preproc_f = lambda img, lab: preprocessing_preset(img, lab, crop_size, args.interpolation, args.auto_augment, args.random_erase)
    else:
        try:
            preproc_f = load_simple_prepr(preprocessing)
        except KeyError as e:
            print(e)
            raise Exception(f'Error loading builtin preprocessing for {preprocessing}!')
    return preproc_f


def load_simple_prepr(model_name):
    prepr = BUILTIN_PREPR[model_name]
    input_size = MODEL_CUSTOM_INPUT.get(model_name, (224, 224))
    return lambda img, label : simple_prepr(img, label, prepr, input_size)


def simple_prepr(image, label, prepr, input_size):
    i = tf.cast(image, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, input_size[0], input_size[1]) # necessary for processing batches
    i = prepr(i)
    return (i, label)


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
    from mlee.ml_tensorflow.autoaugment import ImageNetPolicy
    img = tf.keras.preprocessing.image.array_to_img(img_in, scale=False)
    policy = ImageNetPolicy()
    img = policy(img)
    return tf.convert_to_tensor(tf.keras.preprocessing.image.img_to_array(img))