import numpy as np
import tensorflow as tf

def generate__task(sequence_length, vector_dim):
    sequence = np.random.randint(0, 2, size=(1, sequence_length, vector_dim))
    inputs = np.concatenate([sequence, np.zeros((1, 1, vector_dim))], axis=1)
    return inputs, targets

inputs, targets = generate__task(10, 0)
ntm = NeuralTuringMachine(8, 8, 128, 20)
loss = train_step(inputs. targets, ntm, optimizer)