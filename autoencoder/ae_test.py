import numpy as npsdfsa 
import tensorflow as tf
import matplotlib.pyplot as plt
from data_prep import list_training_files

def read_image_file(filename, _):
        
        print('filename', filename)

        image_string = tf.read_file(filename)

        image = tf.image.decode_png(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        target_image = tf.image.decode_png(image_string, channels=3)
        target_image = tf.image.convert_image_dtype(target_image, tf.float32)

        return image, target_image

training_files = list_training_files()[:3]
print('len(training_files)', len(training_files))


dataset = tf.data.Dataset.from_tensor_slices((training_files, training_files))
dataset = dataset.shuffle(len(training_files))
dataset = dataset.map(read_image_file, num_parallel_calls=4)
dataset = dataset.batch(3)
dataset = dataset.prefetch(1)

iterator = dataset.make_one_shot_iterator()
img_in, img_out = iterator.get_next()

with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        img_in, img_out = sess.run([img_in, img_out])


        print(img_in[0])
        print(img_out[0])

        fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(20,4))

        for i in range(0,3):
                axes[i].imshow(img_in[i])
                axes[i].get_xaxis().set_visible(False)
                axes[i].get_yaxis().set_visible(False)

        for i in range(0,3):
                axes[i].imshow(img_out[i])
                axes[i].get_xaxis().set_visible(False)
                axes[i].get_yaxis().set_visible(False)

        fig.tight_layout(pad=0.1)
        plt.show()