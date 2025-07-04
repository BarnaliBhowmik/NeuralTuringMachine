import numpy as np
import tensorflow as tf

@tf.function
def train_step(inputs, targets, model, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = tf.reduce_mean(tf.square(outputs - targets))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss