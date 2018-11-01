from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input

img_width, img_height = 384, 128

encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')

print(encoder_model.summary())
print(decoder_model.summary())

img_path = '/space/datasets/kitti/format1/test/2011_09_26_0079/0000000000.png'


img = image.load_img(img_path, target_size=(img_height, img_width))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.

# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

random = np.random.rand(1, 8, 24, 32)

rand_decoded = decoder_model.predict(random)

plt.imshow(rand_decoded[0])
plt.show()


# autoencoder_input = Input(shape=(128, 384, 3))
# autoencoder = Model( autoencoder_input, decoder_model(encoder_model(autoencoder_input)) )

# pred = autoencoder.predict(img_tensor)

# print(pred)

# plt.imshow(pred[0])
# plt.show()

# from keras import models

# # Extracts the outputs of the top 8 layers:
# layer_outputs = [layer.output for layer in encoder.layers[1:(len(encoder.layers)-1)]]
# print(layer_outputs)
# # Creates a model that will return these outputs, given the model input:
# activation_model = models.Model(inputs=encoder.input, outputs=layer_outputs)

# activations = activation_model.predict(img_tensor)


# first_layer_activation = activations[0]
# print(first_layer_activation.shape)

# import matplotlib.pyplot as plt

# plt.matshow(first_layer_activation[0, :, :, 10], cmap='viridis')
# plt.show()

# sec_layer_activation = activations[5]
# print(sec_layer_activation.shape)

# plt.matshow(sec_layer_activation[0, :, :, 29], cmap='viridis')
# plt.show()
