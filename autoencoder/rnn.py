def read_sequence(sequence):
        l = list()

        for i in range(0, len(sequence)):
                data = np.load(sequence[i].decode())
                l.append(data.flatten())
                
        return [l]

sequnces = list_training_sequences(3)

decoder_dataset = tf.data.Dataset.from_tensor_slices((sequnces))
decoder_dataset = decoder_dataset.map(lambda item: tf.py_func(read_sequence, [item], tf.float32))
decoder_dataset = decoder_dataset.batch(batch_size)
decoder_dataset = decoder_dataset.prefetch(1)
decoder_dataset = decoder_dataset.repeat(-1)

decoder_iterator = decoder_dataset.make_one_shot_iterator()
encoded_in = decoder_iterator.get_next()