from cnn_base.Models import get_model
from cnn_base.Data import Data_Loader
from cnn_base.utils import Visualizer
from cnn_base.loggers import Logger
import numpy as np

# 1. Load Data
# Assuming data is in a folder 'my_dataset/'
loader = Data_Loader()
loader.fetch_and_save_data(data_dir='my_dataset/', output_dir='processed_data')
X_train = np.load('processed_data/X_train.npy')
y_train = np.load('processed_data/y_train.npy')
X_val = np.load('processed_data/X_val.npy')
y_val = np.load('processed_data/y_val.npy')

# 2. Get and Fine-Tune a Model
model = get_model("efficientnetb0")

# Fine-tune the last 30 layers
model.freeze_all()
model.unfreeze_later_n(30)
model.compile()

# 3. Train the model (tracking is automatic)
history = model.fit(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    epochs=15,
    tags={"project": "initial_tests"}
)

# 4. Analyze Results
model.load_best_model() # Load the best weights from training
visualizer = Visualizer(Logger())
visualizer.plot_training_history(history)