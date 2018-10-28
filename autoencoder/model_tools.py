import numpy as np 
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from data_prep import list_training_files, list_testing_files


# 128x384x3 -> 8x24x32 -> 128x384x3
# 147456 -> 6144 -> 147456
# 1.00 -> 0.042 -> 1.00

batch_size = 5

model_path = "./autoencoder/model/model_v2.ckpt"

def read_image_file(filename):
        
        print('filename', filename)

        image_string = tf.read_file(filename)

        image = tf.image.decode_png(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        return image

is_training = True
training_files = list_training_files()
        

##########
# ENCODER #
##########

dataset = tf.data.Dataset.from_tensor_slices((training_files))
dataset = dataset.shuffle(len(training_files))
dataset = dataset.map(read_image_file, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
dataset = dataset.repeat(-1)

iterator = dataset.make_one_shot_iterator()
img_in = iterator.get_next()

encoderInputs = tf.placeholder_with_default(img_in, (None, 128, 384, 3), name='encoderInputs')

# Now 128x384x3

conv1 = tf.layers.conv2d(inputs=encoderInputs, filters=64, kernel_size=(5,5), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu, name="conv1")
conv1_norm = tf.layers.batch_normalization(conv1, training=is_training, name="conv1_norm")
print('conv1', conv1.get_shape())
# Now 128x384x32

maxpool1 = tf.layers.max_pooling2d(conv1_norm, pool_size=(2,2), strides=(2,2), padding='same', name="maxpool1")
print('maxpool1', maxpool1.get_shape())
# Now 64x192x32

conv2 = tf.layers.conv2d(inputs=maxpool1, filters=48, kernel_size=(5,5), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu, name="conv2")
conv2_norm = tf.layers.batch_normalization(conv2, training=is_training, name="conv2_norm")
print('conv2', conv2.get_shape())
# Now 64x192x32

maxpool2 = tf.layers.max_pooling2d(conv2_norm, pool_size=(2,2), strides=(2,2), padding='same', name="maxpool2")
print('maxpool2', maxpool2.get_shape())
# Now 32x96x32

conv3 = tf.layers.conv2d(inputs=maxpool2, filters=32, kernel_size=(5,5), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu, name="conv3")
conv3_norm = tf.layers.batch_normalization(conv3, training=is_training, name="conv3_norm")
print('conv3', conv3.get_shape())
# Now 32x96x32

maxpool3 = tf.layers.max_pooling2d(conv3_norm, pool_size=(2,2), strides=(2,2), padding='same', name="maxpool3")
print('maxpool3', maxpool3.get_shape())
# Now 16x48x32

conv4 = tf.layers.conv2d(inputs=maxpool3, filters=32, kernel_size=(5,5), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu, name="conv4")
conv4_norm = tf.layers.batch_normalization(conv4, training=is_training, name="conv4_norm")
print('conv4', conv4.get_shape())
# Now 16x48x32

encoded = tf.layers.max_pooling2d(conv4_norm, pool_size=(2,2), strides=(2,2), padding='same', name="encoded")
print('encoded', encoded.get_shape())
# Now 8x24x32

##########
# DECODER #
##########

def read_encoded_image_file(filename):
    data = np.load(filename.decode())
    return data.astype(np.float32)

encoded_files = ['./autoencoder/test/test0.npy','./autoencoder/test/test1.npy','./autoencoder/test/test2.npy','./autoencoder/test/test3.npy','./autoencoder/test/test4.npy']

decoder_dataset = tf.data.Dataset.from_tensor_slices((encoded_files))
decoder_dataset = decoder_dataset.map(lambda item: tf.py_func(read_encoded_image_file, [item], tf.float32))
decoder_dataset = decoder_dataset.batch(batch_size)
decoder_dataset = decoder_dataset.prefetch(1)
decoder_dataset = decoder_dataset.repeat(-1)

decoder_iterator = decoder_dataset.make_one_shot_iterator()
encoded_in = decoder_iterator.get_next()

decoderInputs = tf.placeholder_with_default(encoded_in, (None, 8, 24, 32), name='decoderInputs')

### Decoder
upsample1 = tf.layers.conv2d_transpose(decoderInputs, 32, 2, strides=2, padding='same', name="upsample1")
upsample1_norm = tf.layers.batch_normalization(upsample1, training=is_training, name="upsample1_norm")
print('upsample1', upsample1.get_shape())
# Now 16x48x16

conv5 = tf.layers.conv2d(inputs=upsample1_norm, filters=32, kernel_size=(5,5), padding='same', activation=tf.nn.relu, name="conv5")
conv5_norm = tf.layers.batch_normalization(conv5, training=is_training, name="conv5_norm")
print('conv5', conv5.get_shape())
# Now 16x48x16

upsample2 = tf.layers.conv2d_transpose(conv5_norm, 32, 2, strides=2, kernel_initializer=tf.glorot_normal_initializer(), padding='same', activation=tf.nn.relu, name="upsample2")
upsample2_norm = tf.layers.batch_normalization(upsample2, training=is_training, name="upsample2_norm")
print('upsample2', upsample2.get_shape())
# Now 32x96x16

conv6 = tf.layers.conv2d(inputs=upsample2_norm, filters=32, kernel_size=(5,5), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu, name="conv6")
conv6_norm = tf.layers.batch_normalization(conv6, training=is_training, name="conv6_norm")
print('conv6', conv6.get_shape())
# Now 32x96x16

upsample3 = tf.layers.conv2d_transpose(conv6_norm, 32, 2, strides=2, kernel_initializer=tf.glorot_normal_initializer(), padding='same', activation=tf.nn.relu, name="upsample3")
upsample3_norm = tf.layers.batch_normalization(upsample3, training=is_training, name="upsample3_norm")
print('upsample3', upsample3.get_shape())
# Now 64x192x16

conv7 = tf.layers.conv2d(inputs=upsample3_norm, filters=48, kernel_size=(5,5), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu, name="conv7")
conv7_norm = tf.layers.batch_normalization(conv7, training=is_training, name="conv7_norm")
print('conv7', conv7.get_shape())
# Now 64x192x32

upsample4 = tf.layers.conv2d_transpose(conv7_norm, 48, 2, strides=2, kernel_initializer=tf.glorot_normal_initializer(), padding='same', activation=tf.nn.relu, name="upsample4")
upsample4_norm = tf.layers.batch_normalization(upsample4, training=is_training, name="upsample4_norm")
print('upsample4', upsample4.get_shape())
# Now 128x384x32

conv8 = tf.layers.conv2d(inputs=upsample4_norm, filters=64, kernel_size=(5,5), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu, name="conv8")
conv8_norm = tf.layers.batch_normalization(conv8, training=is_training, name="conv8_norm")
print('conv8', conv8.get_shape())
# Now 128x384x32

decoded = tf.layers.conv2d(inputs=conv8_norm, filters=3, kernel_size=(5,5), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=None, name="decoded")
#decoded_norm = tf.layers.batch_normalization(decoded, training=is_training, name="decoded_norm")
print('decoded', decoded.get_shape())
# Now 128x384x3



def encode():

        saver = tf.train.Saver()

        with tf.Session() as sess:
                saver.restore(sess, model_path)
                print("Model restored.")
                
                _encoded = sess.run(encoded)
                print('_encoded', _encoded.shape)
                # print(_encoded)

                for i in range(0, _encoded.shape[0]):
                    f = './autoencoder/test/test' + str(i)
                    print('_encoded[i].shape', _encoded[i].shape)
                    np.save(f, _encoded[i])
                    print('saved', f)


def decode():

        saver = tf.train.Saver()

        with tf.Session() as sess:
                saver.restore(sess, model_path)
                print("Model restored.")
                
                
                _encoded_in = sess.run(encoded_in)
                print('_encoded_in.size', _encoded_in.shape)
                # print(_encoded_in)

                _decoded = sess.run(decoded)

                print('_decoded', _decoded.shape)
                print(_decoded)

                fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(20,4))

                for i in range(0,5):
                        axes[i].imshow(_decoded[i].reshape((128, 384, 3)))
                        axes[i].get_xaxis().set_visible(False)
                        axes[i].get_yaxis().set_visible(False)

                fig.tight_layout(pad=0.1)
                plt.show()

encode()
decode()