from keras.metrics import Accuracy, SparseCategoricalAccuracy

def get_accuracy(name : str = "accuracy", num_classes : int = 4) :
    if num_classes > 2 :
        return SparseCategoricalAccuracy(name)
    else :
        return Accuracy(name)