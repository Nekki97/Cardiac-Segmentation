import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import random
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


# TODO: since unet basically has 2 parameters (starting number of filters and amount of layers)
#  make those two parameters easily changeable to test which works best

# TODO: read through original UNET paper and if necessary implement dropout

def unet(input_size, filters, layers, pretrained_weights=None):
    inputs = Input(input_size)
    print('Inputs in UNet Shape: ' + str(inputs.shape))
    conv_down = np.empty(layers, dtype=object)
    conv_up = np.empty(layers, dtype=object)
    temp = inputs
    for i in range(layers):
        print('----------Layer #' + str(i+1) + '----------')
        print('First Conv with ' + str(filters * 2**i) + ' filters at index ' + str(i))
        conv_down[i] = Conv2D(filters * 2**i, 3, activation='relu', padding='same', kernel_initializer='he_normal')(temp)
        print('Second Conv with ' + str(filters * 2**i) + ' filters at index ' + str(i))
        conv_down[i] = Conv2D(filters * 2**i, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_down[i])
        if i < layers-1:
            print('MaxPooling')
            temp = MaxPooling2D(pool_size=(2, 2))(conv_down[i])
        print('End Layer #' + str(i+1))

    for j in range(layers-2, -1, -1):
        print('----------Layer #' + str(j+1) + '----------')
        print('First Conv with ' + str(filters * 2**j) + ' filters at index ' + str(j))
        temp = Conv2D(filters * 2 ** j, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv_down[j+1]))
        print('Merging ' + str(conv_down[j].shape) + ' and ' + str(temp.shape))
        merge6 = concatenate([conv_down[j], temp], axis=3)
        print('Second Conv with ' + str(filters * 2**j) + ' filters at index ' + str(j))
        conv_up[j] = Conv2D(filters * 2 ** j, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            merge6)
        print('Third Conv with ' + str(filters * 2 ** j) + ' filters at index ' + str(j))
        conv_up[j] = Conv2D(filters * 2 ** j, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            conv_up[j])
        print('End Layer #' + str(j + 1))

    conv_almost_final = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_up[0])
    conv_final = Conv2D(1, 1, activation='sigmoid')(conv_almost_final)
    print('********** Resulting shape: ' + str(conv_final.shape) + ' **********')

    model = Model(input=inputs, output=conv_final)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# TODO: try dice loss

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
