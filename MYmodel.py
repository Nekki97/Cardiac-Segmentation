from keras.models import *
from keras.layers import *
from keras.optimizers import *

def param_unet(input_size, filters, layers, dropout_rate, pretrained_weights=None):
    inputs = Input(input_size)
    print('Inputs in UNet Shape: ' + str(inputs.shape))
    conv_down = np.empty(layers, dtype=object)
    conv_up = np.empty(layers, dtype=object)
    temp = inputs
    for i in range(layers):
        conv_down[i] = Conv2D(filters * 2**i, 3, activation='relu', padding='same', kernel_initializer='he_normal')(temp)
        conv_down[i] = Conv2D(filters * 2**i, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_down[i])
        conv_down[i] = Dropout(dropout_rate)(conv_down[i])
        if i < layers-1:
            temp = MaxPooling2D(pool_size=(2, 2))(conv_down[i])

    temp = conv_down[layers-1]

    for j in range(layers-2, -1, -1):
        conv_up[j]= Conv2D(filters * 2 ** j, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(temp))
        conv_up[j] = concatenate([conv_up[j], conv_down[j]], axis=3)
        conv_up[j] = Conv2D(filters * 2 ** j, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            conv_up[j])
        temp = Conv2D(filters * 2 ** j, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            conv_up[j])

    conv_almost_final = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(temp)
    conv_final = Conv2D(1, 1, activation='sigmoid')(conv_almost_final)
    print('********** Resulting shape: ' + str(conv_final.shape) + ' **********')

    model = Model(input=inputs, output=conv_final)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':      #only gets called if functions.py is run

    model = param_unet((96,96,1), 64, 6, 0.5)

    from keras.utils import plot_model

    plot_model(model, to_file='param_unet.svg', show_shapes=True)
