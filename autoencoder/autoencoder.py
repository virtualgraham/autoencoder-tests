import numpy as npsdfsa 
import tensorflow as tf
import matplotlib.pyplot as plt
from data_prep import list_training_files


epochs = 1
learning_rate = 0.001
batch_size = 10

def read_image_file(filename, _):
        
        print('filename', filename)

        image_string = tf.read_file(filename)

        image = tf.image.decode_png(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        target_image = tf.image.decode_png(image_string, channels=3)
        target_image = tf.image.convert_image_dtype(target_image, tf.float32)
        
        return image, target_image


training_files = list_training_files()
print('len(training_files)', len(training_files))

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

# Now 128x384x3

conv1 = tf.layers.conv2d(inputs=inputs_, filters=64, kernel_size=(3,3), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu)
conv1 = tf.layers.batch_normalization(conv1, training=True)
print('conv1', conv1.get_shape())
# Now 128x384x32

maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
print('maxpool1', maxpool1.get_shape())
# Now 64x192x32

conv2 = tf.layers.conv2d(inputs=maxpool1, filters=48, kernel_size=(3,3), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu)
conv2 = tf.layers.batch_normalization(conv2, training=True)
print('conv2', conv2.get_shape())
# Now 64x192x32

maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
print('maxpool2', maxpool2.get_shape())
# Now 32x96x32

conv3 = tf.layers.conv2d(inputs=maxpool2, filters=32, kernel_size=(3,3), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu)
conv3 = tf.layers.batch_normalization(conv3, training=True)
print('conv3', conv3.get_shape())
# Now 32x96x32

maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
print('maxpool3', maxpool3.get_shape())
# Now 16x48x32

conv4 = tf.layers.conv2d(inputs=maxpool3, filters=32, kernel_size=(3,3), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu)
conv4 = tf.layers.batch_normalization(conv4, training=True)
print('conv4', conv4.get_shape())
# Now 16x48x32

encoded = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), padding='same')
print('encoded', encoded.get_shape())
# Now 8x24x32



### Decoder
upsample1 = tf.layers.conv2d_transpose(encoded, 32, 2, strides=2, padding='same')
upsample1 = tf.layers.batch_normalization(upsample1, training=True)
print('upsample1', upsample1.get_shape())
# Now 16x48x16

conv5 = tf.layers.conv2d(inputs=upsample1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
conv5 = tf.layers.batch_normalization(conv5, training=True)
print('conv5', conv5.get_shape())
# Now 16x48x16

upsample2 = tf.layers.conv2d_transpose(conv5, 32, 2, strides=2, kernel_initializer=tf.glorot_normal_initializer(), padding='same', activation=tf.nn.relu)
upsample2 = tf.layers.batch_normalization(upsample2, training=True)
print('upsample2', upsample2.get_shape())
# Now 32x96x16

conv6 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu)
conv6 = tf.layers.batch_normalization(conv6, training=True)
print('conv6', conv6.get_shape())
# Now 32x96x16

upsample3 = tf.layers.conv2d_transpose(conv6, 32, 2, strides=2, kernel_initializer=tf.glorot_normal_initializer(), padding='same', activation=tf.nn.relu)
upsample3 = tf.layers.batch_normalization(upsample3, training=True)
print('upsample3', upsample3.get_shape())
# Now 64x192x16

conv7 = tf.layers.conv2d(inputs=upsample3, filters=48, kernel_size=(3,3), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu)
conv7 = tf.layers.batch_normalization(conv7, training=True)
print('conv7', conv7.get_shape())
# Now 64x192x32

upsample4 = tf.layers.conv2d_transpose(conv7, 48, 2, strides=2, kernel_initializer=tf.glorot_normal_initializer(), padding='same', activation=tf.nn.relu)
upsample4 = tf.layers.batch_normalization(upsample4, training=True)
print('upsample4', upsample4.get_shape())
# Now 128x384x32

conv8 = tf.layers.conv2d(inputs=upsample4, filters=64, kernel_size=(3,3), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=tf.nn.relu)
conv8 = tf.layers.batch_normalization(conv8, training=True)
print('conv8', conv8.get_shape())
# Now 128x384x32

logits = tf.layers.conv2d(inputs=conv8, filters=3, kernel_size=(3,3), padding='same', kernel_initializer=tf.glorot_normal_initializer(), activation=None)
logits = tf.layers.batch_normalization(logits, training=True)
print('logits', logits.get_shape())
# Now 128x384x3



print('targets_', targets_.get_shape())
print('logits', logits.get_shape())

loss = tf.losses.mean_squared_error(targets_, logits)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()

with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):

                for ii in range(len(training_files)//batch_size):

                        batch_cost, _ = sess.run([cost, opt])

                        print("Epoch: {}/{}...".format(e+1, epochs),
                        "Training loss: {:.4f}".format(batch_cost))

        save_path = saver.save(sess, "./autoencoder/model/model.ckpt")
        print("Model saved in path: %s" % save_path)

        fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(20,4))

        reconstructed = sess.run(logits)

        for i in range(0,3):
                axes[i].imshow(reconstructed[i].reshape((128,384, 3)))
                axes[i].get_xaxis().set_visible(False)
                axes[i].get_yaxis().set_visible(False)

        fig.tight_layout(pad=0.1)
        plt.show()