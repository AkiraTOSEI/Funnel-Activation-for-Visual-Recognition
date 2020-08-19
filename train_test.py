import os

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
import tensorflow_addons as tfa

from FReLU import FReLU
from resnet import ResnetBuilder

'''
create dataloader using tf.data
'''
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
batch_size = 32
img_size = 32
num_label = 10
# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# Convert class vectors to binary class matrices.
Y_train = to_categorical(y_train, num_label)
Y_test = to_categorical(y_test, num_label)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.


# data augmentation
# random flip
def flip_image(image):
    return tf.image.random_flip_left_right(image)


shift_range = 0.1
ch_num = 3
expand_target_size = int(img_size*(shift_range+1))


def image_shift(image):
    # padding
    image = tf.image.resize_with_crop_or_pad(
        image,
        expand_target_size,
        expand_target_size
    )
    # random crop
    image = tf.image.random_crop(
        image,
        size=[img_size, img_size, ch_num]
    )
    return image


# create train dataset with data augmentation
autotune = tf.data.experimental.AUTOTUNE
input_ds = tf.data.Dataset.from_tensor_slices(X_train)
input_ds = input_ds.map(flip_image).map(image_shift)
target_ds = tf.data.Dataset.from_tensor_slices(Y_train)
train_ds = tf.data.Dataset.zip((input_ds, target_ds)).batch(
    batch_size).repeat().shuffle(batch_size*32)
train_ds = train_ds.prefetch(autotune)

# create test
input_ds = tf.data.Dataset.from_tensor_slices(X_test)
target_ds = tf.data.Dataset.from_tensor_slices(Y_test)
test_ds = tf.data.Dataset.zip((input_ds, target_ds)).batch(batch_size).repeat()
test_ds = test_ds.prefetch(autotune)


'''
train and test 
'''
results_dict = {}
num_try = 3

for try_i in range(num_try):
    builder = ResnetBuilder(name='ResNet18', act='FReLU', classes=num_label,
                            include_top=True, input_shape=(img_size, img_size, 3))
    frelu_model = builder.builder()
    frelu_model.compile(loss='categorical_crossentropy',
                        optimizer='sgd',
                        metrics=['accuracy'])
    hist = frelu_model.fit(train_ds,
                           steps_per_epoch=int(np.ceil(X_train.shape[0]//32)),
                           validation_data=test_ds,
                           validation_steps=int(np.ceil(X_test.shape[0]/32)),
                           epochs=200,
                           verbose=1,
                           )
    results_dict["FReLU-try" +
                 str(try_i)] = hist.history['accuracy'], hist.history['loss'], hist.history['val_accuracy'], hist.history['val_loss']


for try_i in range(num_try):
    builder = ResnetBuilder(name='ResNet18', act='Swish', classes=num_label,
                            include_top=True, input_shape=(img_size, img_size, 3))
    swish_model = builder.builder()
    swish_model.compile(loss='categorical_crossentropy',
                        optimizer='sgd',
                        metrics=['accuracy'])
    hist = swish_model.fit(train_ds,
                           steps_per_epoch=int(np.ceil(X_train.shape[0]//32)),
                           validation_data=test_ds,
                           validation_steps=int(np.ceil(X_test.shape[0]/32)),
                           epochs=200,
                           verbose=1,
                           )
    results_dict["swish-try" +
                 str(try_i)] = hist.history['accuracy'], hist.history['loss'], hist.history['val_accuracy'], hist.history['val_loss']


for try_i in range(num_try):
    builder = ResnetBuilder(name='ResNet18', act='ReLU', classes=num_label,
                            include_top=True, input_shape=(img_size, img_size, 3))
    swish_model = builder.builder()
    swish_model.compile(loss='categorical_crossentropy',
                        optimizer='sgd',
                        metrics=['accuracy'])
    hist = swish_model.fit(train_ds,
                           steps_per_epoch=int(np.ceil(X_train.shape[0]//32)),
                           validation_data=test_ds,
                           validation_steps=int(np.ceil(X_test.shape[0]/32)),
                           epochs=200,
                           verbose=1,
                           )
    results_dict["ReLU-try" +
                 str(try_i)] = hist.history['accuracy'], hist.history['loss'], hist.history['val_accuracy'], hist.history['val_loss']


for act in ["ReLU", "swish", "FReLU"]:
    result = []
    for try_i in range(3):
        result.append(min(results_dict[act+"-try"+str(try_i)][3]))
    result = np.array(result)
    std = np.std(result)
    mean = np.mean(result)
    print(act, ":", np.round(mean, 3), "±", np.round(std, 3))
