import numpy as np
import tensorflow as tf

class NeuralTuringMachine(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs, memory_size, memory_vector_dim):
        super(NeuralTuringMachine, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim

        self.memory = tf.Variable(tf.zeros([memory_size, memory_vector_dim]))