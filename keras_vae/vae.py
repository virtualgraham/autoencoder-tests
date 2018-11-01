from keras import layers
from keras import models
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LambdaCallback
from keras.layers import Input

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

img_width, img_height = 384, 128

train_data_dir = '/space/datasets/kitti/format1/train'
validation_data_dir = '/space/datasets/kitti/format1/test'
nb_train_samples = 32034
nb_validation_samples = 10112
nb_epoch = 2
batch_size = 10




###########
# Encoder #
###########


def build_encoder():

        img_input = Input(shape=(128, 384, 3))

        # Now 128x384x3

        enc_l1_c1 = layers.Conv2D(32, 5, activation='relu', padding='same')(img_input)
        enc_l1_n1 = layers.BatchNormalization()(enc_l1_c1)
        enc_l1_c2 = layers.Conv2D(32, 2, strides=2, activation='relu', padding='same')(enc_l1_n1)
        enc_l1_n2 = layers.BatchNormalization()(enc_l1_c2)
        # Now 64x192x32


        enc_l2_c1 = layers.Conv2D(48, 5, activation='relu', padding='same')(enc_l1_n2)
        enc_l2_n1 = layers.BatchNormalization()(enc_l2_c1)
        enc_l2_c2 = layers.Conv2D(48, 2, strides=2, activation='relu', padding='same')(enc_l2_n1)
        enc_l2_n2 = layers.BatchNormalization()(enc_l2_c2)
        # Now 32x96x48


        enc_l3_c1 = layers.Conv2D(64, 5, activation='relu', padding='same')(enc_l2_n2)
        enc_l3_n1 = layers.BatchNormalization()(enc_l3_c1)
        enc_l3_c2 = layers.Conv2D(64, 2, strides=2, activation='relu', padding='same')(enc_l3_n1)
        enc_l3_n2 = layers.BatchNormalization()(enc_l3_c2)
        # Now 16x48x64


        enc_l4_c1 = layers.Conv2D(96, 5, activation='relu', padding='same')(enc_l3_n2)
        enc_l4_n1 = layers.BatchNormalization()(enc_l4_c1)
        # Now 16x48x96
        enc_l4_c2 = layers.Conv2D(32, 2, strides=2, activation='relu', padding='same')(enc_l4_n1)
        enc_l4_n2 = layers.BatchNormalization()(enc_l4_c2)
        # Now 8x24x32

        return Model(img_input, enc_l4_n2)

###########
# Decoder #
###########

def build_decoder():

        enc_input = Input(shape=(8, 24, 32))

        # Now 8x24x32

        dec_l1_u1 = layers.UpSampling2D((2,2))(enc_input)
        dec_l1_c1 = layers.Conv2D(96, 2, activation='relu', padding='same')(dec_l1_u1)
        dec_l1_n1 = layers.BatchNormalization()(dec_l1_c1)
        # Now 16x48x96
        dec_l1_c2 = layers.Conv2D(96, 5, activation='relu', padding='same')(dec_l1_n1)
        dec_l1_n2 = layers.BatchNormalization()(dec_l1_c2)
        # Now 16x48x96


        dec_l2_u1 = layers.UpSampling2D((2,2))(dec_l1_n2)
        dec_l2_c1 = layers.Conv2D(64, 2, activation='relu', padding='same')(dec_l2_u1)
        dec_l2_n1 = layers.BatchNormalization()(dec_l2_c1)
        dec_l2_c2 = layers.Conv2D(64, 5, activation='relu', padding='same')(dec_l2_n1)
        dec_l2_n2 = layers.BatchNormalization()(dec_l2_c2)
        # Now 32x96x64


        dec_l3_u1 = layers.UpSampling2D((2,2))(dec_l2_n2)
        dec_l3_c1 = layers.Conv2D(48, 2, activation='relu', padding='same')(dec_l3_u1)
        dec_l3_n1 = layers.BatchNormalization()(dec_l3_c1)
        dec_l3_c2 = layers.Conv2D(48, 5, activation='relu', padding='same')(dec_l3_n1)
        dec_l3_n2 = layers.BatchNormalization()(dec_l3_c2)
        # Now 64x192x48


        dec_l4_u1 = layers.UpSampling2D((2,2))(dec_l3_n2)
        dec_l4_c1 = layers.Conv2D(32, 2, activation='relu', padding='same')(dec_l4_u1)
        dec_l4_n1 = layers.BatchNormalization()(dec_l4_c1)
        dec_l4_c2 = layers.Conv2D(32, 5, activation='relu', padding='same')(dec_l4_n1)
        dec_l4_n2 = layers.BatchNormalization()(dec_l4_c2)
        # Now 128x384x32


        dec_l4_out = layers.Conv2D(3, 3, activation='relu', padding='same')(dec_l4_n2)
        # Now 128x384x3

        return Model(enc_input, dec_l4_out)

###############
# Autoencoder #
###############

encoder_model = build_encoder()

print(encoder_model.summary())

decoder_model = build_decoder()

print(decoder_model.summary())


autoencoder_input = Input(shape=(128, 384, 3))
autoencoder = Model( autoencoder_input, decoder_model(encoder_model(autoencoder_input)) )

print(autoencoder.summary())


autoencoder.compile(optimizer='adam', loss='mse')

def save_models():
        print('Saving Models')
        encoder_model.save('encoder_model.h5')
        decoder_model.save('decoder_model.h5')

def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None)

callbacks_list = [LambdaCallback(on_epoch_end=lambda epoch, logs: save_models() )]

history = autoencoder.fit_generator(
        fixed_generator(train_generator),
        steps_per_epoch=nb_train_samples//batch_size,
        nb_epoch=nb_epoch,
        callbacks=callbacks_list,
        validation_data=fixed_generator(validation_generator),
        nb_val_samples=nb_validation_samples)

