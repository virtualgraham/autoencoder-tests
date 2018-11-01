from keras import layers
from keras import models
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LambdaCallback
from keras.layers import Input
import tensorflow as tf
from keras import backend as K
from keras import metrics

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

latent_dim = 8*24*32

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon



img_input = Input(shape=(128, 384, 3), name="encoder_input")

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

enc_flattened = layers.Flatten()(enc_l4_n2)
        
#enc_dense = layers.Dense(latent_dim, activation='relu')(enc_flattened)

z_mean = layers.Dense(latent_dim, name="z_mean")(enc_flattened)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(enc_flattened)

print('TYPES', type(z_mean), type(z_log_var))

z = layers.Lambda(sampling)([z_mean, z_log_var])


###########
# Decoder #
###########

def build_decoder():

        decoder_input = Input(shape=(latent_dim,), name="decoder_input")

        #dec_dense = layers.Dense(8*24*32, activation='relu')(decoder_input)

        dec_reshape = layers.Reshape((8,24,32))(decoder_input)

        # Now 8x24x32

        dec_l1_u1 = layers.UpSampling2D((2,2))(dec_reshape)
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


        dec_l4_out = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(dec_l4_n2)
        # Now 128x384x3

        return Model(decoder_input, dec_l4_out)

###############
# Autoencoder #
###############


decoder_model = build_decoder()

print(decoder_model.summary())


z_decoded = decoder_model(z)


class CustomVariationalLayer(layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = metrics.binary_crossentropy(x, z_decoded)

        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

zero_loss = tf.constant(0.0)

def customLoss(yTrue,yPred):
    return zero_loss


y = CustomVariationalLayer()([img_input, z_decoded])

vae = Model(img_input, y)
vae.compile(optimizer='rmsprop', loss=customLoss)
print(vae.summary())


# autoencoder = Model( autoencoder_input, decoder_model(encoder_model(autoencoder_input)) )

# print(autoencoder.summary())


# autoencoder.compile(optimizer='adam', loss='mse')

def save_models():
        print('Saving Models')
        vae.save('vae.h5')

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

history = vae.fit_generator(
        fixed_generator(train_generator),
        steps_per_epoch=nb_train_samples//batch_size,
        nb_epoch=nb_epoch,
        callbacks=callbacks_list,
        validation_data=fixed_generator(validation_generator),
        nb_val_samples=nb_validation_samples)

