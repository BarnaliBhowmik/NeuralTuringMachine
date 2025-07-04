import numpy as np
import tensorflow as tf

def read_memory(memory, read_weights):
    return tf.reduce_sum(memory * read_weights[:, :, tf.newaxis],  axis=1)

def write_memory(memory, write_weights, erase_vector, add_vector):
    erase = tf.reduce_sum(write_weights[:, :, tf.newaxis] * erase_vector[:, tf.newaxis, :], axis=1)
    add = tf.reduce_sum(write_weights[:, :, tf.newaxis] * add_vector[:, tf.newaxis, :], axis=1)
    return memory * (1 - erase) + add