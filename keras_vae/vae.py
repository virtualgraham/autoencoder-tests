from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

img_width, img_height = 384, 128

train_data_dir = '/space/datasets/kitti/format1/train'
validation_data_dir = '/space/datasets/kitti/format1/test'
nb_train_samples = 32034
nb_validation_samples = 10112
nb_epoch = 10
batch_size = 13


model = models.Sequential()

# Now 128x384x3

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(384, 128, 3), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
# Now 64x192x32

model.add(layers.Conv2D(48, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
# Now 32x96x64

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
# Now 16x48x64

model.add(layers.Conv2D(96, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
# Now 8x24x128

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
# Now 4x12x128

  

model.add(layers.UpSampling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
# Now 8x24x128

model.add(layers.UpSampling2D((2,2)))
model.add(layers.Conv2D(96, (3, 3), activation='relu', padding='same'))
# Now 16x48x64

model.add(layers.UpSampling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# Now 32x96x64

model.add(layers.UpSampling2D((2,2)))
model.add(layers.Conv2D(48, (3, 3), activation='relu', padding='same'))
# Now 64x192x32

model.add(layers.UpSampling2D((2,2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# Now 128x384x32

model.add(layers.Conv2D(3, (3, 3), activation=None, padding='same'))
# Now 128x384x3


print(model.summary())


model.compile(optimizer='adadelta', loss='binary_crossentropy')


def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)

callbacks_list = [ModelCheckpoint('keras_ae.model', verbose=True)]

history =  model.fit_generator(
        fixed_generator(train_generator),
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        callbacks=callbacks_list,
        validation_data=fixed_generator(validation_generator),
        nb_val_samples=nb_validation_samples)