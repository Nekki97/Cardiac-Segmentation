from keras.models import *
from keras.layers import *
from keras.optimizers import *
from functions import *

def param_unet(input_size, filters, layers, dropout_rate, loss_name, pretrained_weights=None):
    inputs = Input(input_size)
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

    model = Model(input=inputs, output=conv_final)

    if loss_name == 'binary_crossentropy':
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    if loss_name == 'dice':
        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coeff_loss, metrics=['accuracy'])
    if loss_name == 'weighted_cross_entropy':
        model.compile(optimizer=Adam(lr=1e-4), loss=weighted_cross_entropy, metrics=['accuracy'])
    #model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def segnet(img_shape, kernel_size, Dropout_rate, loss_name):
    inputs = Input(img_shape)
    model = Sequential()

    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    # Decoder Layers
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(2,1, activation='relu', padding='same'))  #try this
    model.add(Conv2D(1, 1, activation='sigmoid', padding='same'))

    if loss_name == 'binary_crossentropy':
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    if loss_name == 'dice':
        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coeff_loss, metrics=['accuracy'])
    if loss_name == 'weighted_cross_entropy':
        model.compile(optimizer=Adam(lr=1e-4), loss=weighted_cross_entropy, metrics=['accuracy'])

    #model.summary()

    return model


if __name__ == '__main__':      #only gets called if functions.py is run

    model = param_unet((128,128,1), 64, 5, 0.5, "binary_crossentropy")

    from keras.utils import plot_model

    plot_model(model, to_file='Desktop/param_unet.svg', show_shapes=True)
