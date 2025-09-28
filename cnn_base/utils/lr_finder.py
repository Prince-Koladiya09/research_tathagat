import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class _LR_Finder_Callback(keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, steps):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps = steps
        self.learning_rates = []
        self.losses = []
        self.lr_multiplier = (max_lr / min_lr) ** (1 / steps)

    def on_train_begin(self, logs=None):
        self.weights = self.model.get_weights()
        keras.backend.set_value(self.model.optimizer.learning_rate, self.min_lr)

    def on_train_batch_end(self, batch, logs=None):
        lr = keras.backend.get_value(self.model.optimizer.learning_rate)
        self.learning_rates.append(lr)
        self.losses.append(logs["loss"])
        
        new_lr = lr * self.lr_multiplier
        keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        
        if logs["loss"] > 5 * (self.losses[0] or 1.0) or new_lr > self.max_lr:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.weights)
        self.plot()
    
    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.learning_rates, self.losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True)
        plt.show()

def find_learning_rate(model, dataset, min_lr, max_lr):
    if not model.optimizer:
        raise ValueError("Model must be compiled before finding the learning rate.")
    
    num_batches = tf.data.experimental.cardinality(dataset).numpy()
    if num_batches == tf.data.experimental.UNKNOWN_CARDINALITY:
        num_batches = 100
        
    lr_finder_callback = _LR_Finder_Callback(min_lr, max_lr, num_batches)
    
    model.fit(dataset.take(num_batches), epochs=1, callbacks=[lr_finder_callback], verbose=0)