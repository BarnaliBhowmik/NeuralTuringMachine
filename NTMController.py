import numpy as np
import tensorflow as tf

class NTMController(tf, keras, layers, Layer):
    def __init__(self, num_inputs, num_outputs, memory_vector_dim):
        super(NTMController, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_outputs + memory_vector_dim * 2)
    
    def call(self, inputs, prev_state);
        x = tf.concat([inputs, prev_state], axis=-1)
        x = self.dense(x)
        return self.dense2(x)