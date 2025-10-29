import tensorflow as tf
from keras.metrics import Metric, Precision as Keras_Precision
from keras.saving import register_keras_serializable

@register_keras_serializable()
class Precision(Metric):
    def __init__(self, num_classes, name='multi_class_precision', **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.true_positives = self.add_weight(name = 'true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name = 'false_positives', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions from probabilities to class indices
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_true_classes = tf.argmax(y_true, axis=-1)
        
        # Create one-hot encodings
        y_pred_oh = tf.one_hot(y_pred_classes, depth=self.num_classes)
        y_true_oh = tf.one_hot(y_true_classes, depth=self.num_classes)
        
        # Calculate true positives and false positives
        tp = tf.cast(tf.reduce_sum(y_true_oh * y_pred_oh), tf.float32)
        fp = tf.cast(tf.reduce_sum((1 - y_true_oh) * y_pred_oh), tf.float32)
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        return precision
    
    def reset_states(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def get_precision(name : str = "precision", num_classes : int = 4) -> Metric :
    if num_classes > 2 :
        return Precision(name = name, num_classes = num_classes)
    else :
        return Keras_Precision(name = name)