import tensorflow as tf
from keras.metrics import Metric, AUC as Keras_AUC
from keras.saving import register_keras_serializable

@register_keras_serializable()
class AUC(Metric):
    def __init__(self, num_classes, name='multi_class_auc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Create an AUC metric for each class
        self.auc_per_class = [Keras_AUC(name=f'auc_class_{i}') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true is sparse (batch,), y_pred is (batch, num_classes)
        # Convert y_true to one-hot
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes)
        for i in range(self.num_classes):
            self.auc_per_class[i].update_state(y_true_one_hot[:, i], y_pred[:, i], sample_weight)

    def result(self):
        aucs = [m.result() for m in self.auc_per_class]
        # Macro-average AUC
        return tf.reduce_mean(aucs)

    def reset_states(self):
        for m in self.auc_per_class:
            m.reset_states()

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def get_auc(name : str = "recall", num_classes : int = 4) -> Metric :
    if num_classes > 2 :
        return AUC(name = name, num_classes = num_classes)
    else :
        return Keras_AUC(name = "recall")