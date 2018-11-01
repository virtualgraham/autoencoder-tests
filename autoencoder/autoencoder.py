import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_prep import list_training_files, list_testing_files


# 128x384x3 -> 8x24x32 -> 128x384x3
# 147456 -> 6144 -> 147456
# 1.00 -> 0.042 -> 1.00

epochs = 15
learning_rate = 0.001
batch_size = 13

model_path = "./autoencoder/model/model_v3.ckpt"

def read_image_file(filename, _):
        
        print('filename', filename)

        image_string = tf.read_file(filename)

        image = tf.image.decode_png(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        target_image = tf.image.decode_png(image_string, channels=3)
        target_image = tf.image.convert_image_dtype(target_image, tf.float32)
        
        return image, target_image

is_training = True
training_files = list_training_files()
        
dataset = tf.data.Dataset.from_tensor_slices((training_files, training_files))
dataset = dataset.shuffle(len(training_files))
dataset = dataset.map(read_image_file, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
dataset = dataset.repeat(-1)

iterator = dataset.make_one_shot_iterator()
img_in, img_out = iterator.get_next()

inputs_ = tf.placeholder_with_default(img_in, (None, 128, 384, 3), name='inputs')
targets_ = tf.placeholder_with_default(img_out, (None, 128, 384, 3), name='target')



conv1 = tf.layers.conv2d(inputs=inputs_, filters=64, kernel_size=(5,5), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu, name="conv1")
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



### Decoder
upsample1 = tf.layers.conv2d_transpose(encoded, 32, 2, strides=2, padding='same', name="upsample1")
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

print('targets_', targets_.get_shape())

loss = tf.losses.mean_squared_error(targets_, decoded)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)


def test():

        saver = tf.train.Saver()

        with tf.Session() as sess:

                saver.restore(sess, model_path)
                print("Model restored.")


                fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(20,4))

                reconstructed = sess.run(decoded)

                for i in range(0,3):
                        #print(reconstructed[i])
                        axes[i].imshow(reconstructed[i].reshape((128,384, 3)))
                        axes[i].get_xaxis().set_visible(False)
                        axes[i].get_yaxis().set_visible(False)

                fig.tight_layout(pad=0.1)
                plt.show()


def train():

        opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())

                for e in range(epochs):

                        for ii in range(len(training_files)//batch_size):

                                batch_cost, _ = sess.run([cost, opt])

                                print("Epoch: {}/{}".format(e+1, epochs),
                                "Training loss: {:.4f}".format(batch_cost), "Iteration: {}".format(ii))

                        save_path = saver.save(sess, model_path)
                        print("Model saved in path: %s" % save_path)


train()
test()