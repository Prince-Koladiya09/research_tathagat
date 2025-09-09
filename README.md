# 🧠 cnn\_base – CNN Training & Visualization Library

A **lightweight, modular deep learning library** for **training, fine-tuning, and interpreting CNN models** using **Keras/TensorFlow**.

It provides:

* Prebuilt CNN architectures: ResNet, DenseNet, EfficientNet, MobileNet, etc.
* Custom callbacks for **progressive unfreezing & learning rate scheduling**.
* Easy **dataset preprocessing & augmentation**.
* Rich **visualizations**: training metrics, confusion matrix, Grad-CAM, embeddings.
* **Fine-tuning** support: freeze/unfreeze layers, cut/add layers, add custom layers.
* **Explainable AI (XAI)** support: LIME & SHAP.

---

## 🚀 Features

| Feature              | Description                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **Data Handling**    | Auto train/val/test split, supports folder-based datasets, easy batch loading.            |
| **Model Building**   | Use `Model` class to quickly build any CNN. Availability of many models from Keras        |
| **Custom Callbacks** | Progressive unfreezing, discriminative learning rates, early stopping, ReduceLROnPlateau. |
| **Visualization**    | Metrics plots, confusion matrix, ROC/PR curves, embeddings (t-SNE/UMAP), Grad-CAM.        |
| **Explainability**   | LIME and SHAP for image-level explanations.                                               |
| **Logging**          | Centralized logging for experiments.                                                      |

---

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/Prince-Koladiya09/research_tathagat.git
cd <repo-name>

# Install library in editable mode
pip install -e .
```

Or manually:

```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Examples

### 1️⃣ Load & preprocess dataset

```python
from cnn_base.Data.data import Data

# Automatically split dataset into train/val/test
train_ds, val_ds, test_ds = Data.fetch_and_save_data("datasets/cats_vs_dogs")
```

### 2️⃣ Build a model

```python
from cnn_base.Models.base_model import Model

# Using the custom Model wrapper for training, callbacks, and fine-tuning
model = Model(update_config_kwargs={"num_classes": 2})
model.get_base_model("resnet50")
model.add_custom_layers()  # adds pooling + dropout + softmax
model.compile(lr=1e-4)
model.summary()
```

### 3️⃣ Fine-tuning & custom layers

```python
from keras import layers

# Freeze first 100 layers
model.freeze_early_N(100)

# Add custom classification head
custom_layers = [
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(4, activation='softmax')
]
model.add_custom_layers(custom_layers)

# Optional: unfreeze last 20 layers for gradual fine-tuning
model.unfreeze_later_N(20)
```

### 4️⃣ Train with callbacks

```python
from cnn_base.Callbacks import ProgressiveUnfreezer, DiscriminativeLRScheduler

progressive_unfreezer = ProgressiveUnfreezer(logger = logger)
discriminative_lr_scheduler = DiscriminativeLRScheduler(logger = logger)

callbacks = [model_checkpoint, early_stopping, reduce_lr, progressive_unfreezer, discriminative_lr_scheduler]

history = model.fit(train_dataset, val_dataset, epochs=20, batch_size=32, callbacks=callbacks)
```

### 5️⃣ Visualize performance

```python
from cnn_base.utils.visualization import Visualizer

viz = Visualizer(logger)
y_true = [y for _, y in test_ds]
y_pred = model.predict(test_ds)

# Confusion matrix
viz.plot_confusion_matrix(y_true, y_pred, class_names=["Cat", "Dog"])

# ROC curve
viz.plot_roc_curve(y_true, y_pred, n_classes=2)

# Training history
viz.plot_training_history(history)

# Grad-CAM for a single image
img = next(iter(test_ds))[0][0].numpy()
viz.plot_grad_cam(model.model, img_array=np.expand_dims(img, axis=0), last_conv_layer_name="conv5_block3_out", original_img=img)
```

###  Embeddings & Explainability

```python
# t-SNE embeddings
features, reduced = viz.plot_embeddings(model=model, layer_name="avg_pool", data=test_ds, labels=y_true, method="tsne")

# LIME explanation
sample_img = next(iter(test_ds))[0][0].numpy()
viz.plot_lime_explanation(model.model, sample_img)

# SHAP explanation (requires small background dataset)
background = next(iter(train_ds))[0][:50]  # 50 images for background
viz.plot_shap_explanation(model.model, background_data=background, images_to_explain=sample_img, class_names=["Cat", "Dog"])
```

---

## 🗂️ Project Structure

```
mylib/
│
│── Callbacks/ # Custom training callbacks
│ ├── discriminative_lr.py
│ └── progressive_unfreeze.py
│
├── Data/ # Data preprocessing + loading
│ └── data.py
│
├── loggers/ # Logging utilities
│ └── logger.py
│
├── Models/ # Model definitions
│ ├── base_model.py
│ └── get_model.py
│
├── utils/ # Visualization tools
│ └── visualization.py
│
└── config.py # Global config
```

---

## 🛠️ Requirements

Core:

* tensorflow>=2.12
* keras>=2.12
* numpy
* pandas
* matplotlib
* scikit-learn
* pillow

Optional:

* opencv-python → Grad-CAM
* umap-learn, plotly → embeddings
* lime, shap → explainability

---

## 👨‍💻 Authors

* Meet Vyas
* Prince Koladiya