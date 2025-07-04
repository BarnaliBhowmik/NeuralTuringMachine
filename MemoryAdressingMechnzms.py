import numpy as np
import tensorflow as tf

def cosine_similarity(x, y):
    return tf.reduce_sum(x * y, axis=-1) / (
        tf.norm(x, axis=-1) * tf.norm(y, axis=-1) + 1e-8)

def content_addressing(key, memory):
    similarity = cosine_similarity(key[:, tf.newaxis, :], memory)
    return tf.nn.softmax(similarity, axis=-1)