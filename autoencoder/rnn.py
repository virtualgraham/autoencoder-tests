import numpy as np
import tensorflow as tf
from data_prep import list_training_sequences, list_testing_sequences

def read_sequence(sequence, _):
        input_seq = list()

        for i in range(0, len(sequence) - 1):
                data = np.load(sequence[i].decode())
                input_seq.append(data.flatten())
        
        output = np.load(sequence[len(sequence) - 1].decode())
        output = output.flatten()

        return input_seq, output

batch_size = 1
hidden_size = 3
seq_length = 4

sequences = list_training_sequences(seq_length+1)[:100]

dataset = tf.data.Dataset.from_tensor_slices((sequences, sequences))
dataset = dataset.shuffle(len(sequences))
dataset = dataset.map(lambda itema, itemb: tf.py_func(read_sequence, [itema, itemb], [tf.float32, tf.float32]))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
dataset = dataset.repeat(-1)

iterator = dataset.make_one_shot_iterator()
encoded_in, encoded_out = iterator.get_next()

X = tf.placeholder_with_default(encoded_in, (None, seq_length, 6144), name='inputs')

cell = tf.nn.rnn_cell.LSTMCell(num_units=6144, state_is_tuple=True)
outputs, states = tf.nn.dynamic_rnn(cell, X, sequence_length=[seq_length]*batch_size, dtype=tf.float32)

loss = tf.losses.mean_squared_error(encoded_out, outputs[:, -1, :])

train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
  sess.run(init_op) 
  
  for step in range(1200):    # training
    _, loss_ = sess.run([train_op, loss])
    print('train loss: %.4f' % loss_)


#   _encoded_in, _encoded_out = sess.run([encoded_in, encoded_out])
#   print(_encoded_in.shape)
#   print(_encoded_out.shape)
#   outputs_val, states_val = sess.run([outputs, states])
#   print(outputs_val.shape)
#   print(states_val)