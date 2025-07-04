import numpy as np
import tensorflow as tf

def attention_mechanism(query, keys, values):
    attention_weights = tf.nn.softmax(tf.matmu(query, keys, transpose_b=True)/tf.sqrt(tf.cast(tf.shape(keys)[-1], tf.float32)))
    return tf.matmul(attention_weights, values)

def read_with_attention(memory, read_query):
    read_attention = attention_mechanism(read_query, memory, memory)
    return tf.reduce_sum(read_attention, axis=1)