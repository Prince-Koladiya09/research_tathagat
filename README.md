# Vision-Tuner: An Advanced Library for Fine-Tuning Vision Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-TensorFlow_2.12+_%7C_Keras_3-orange.svg" alt="Framework">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

**Vision-Tuner** is a high-level, research-focused library built on Keras 3 and TensorFlow 2 for rapidly fine-tuning, evaluating, and analyzing state-of-the-art image classification models.

It moves beyond generic wrappers by providing specialized, architecture-aware classes for both **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)**, enabling you to leverage advanced, model-specific fine-tuning techniques out of the box.

## Core Philosophy

This library is built on four pillars designed to accelerate your research workflow:

1.  **🧠 Specialized Architectures:** Instead of a one-size-fits-all approach, the library provides distinct `CNN_Model` and `Transformer_Model` classes. This allows for implementing and using fine-tuning strategies that are tailored to each architecture, such as progressive unfreezing for CNNs or Layer-wise Learning Rate Decay (LLRD) for Transformers.

2.  **⚙️ Reproducible Workflow:** Experiments are driven by a robust, Pydantic-based configuration system. This ensures that all settings are type-safe and clearly defined within your Python scripts, making your experimental setup transparent and easier to reproduce.

3.  **🚀 Advanced Fine-Tuning:** The library exposes powerful fine-tuning methods directly on the model objects. Freeze the patch embeddings of a ViT, unfreeze the last N blocks, or progressively unfreeze a CNN with simple, intuitive method calls.

4.  **📊 Deep Analysis & Interpretation:** Go beyond accuracy scores. The integrated `Visualizer` provides essential tools for model interpretation, including Grad-CAM and Attention Map plotting, alongside robust evaluation metrics and error analysis.

## 📦 Model Zoo

The library provides a comprehensive collection of pre-trained models, ready for fine-tuning.

<details>
<summary><b>Click to expand the full list of supported models</b></summary>

| Type            | Model Family             | Variants                                                                                                  |
|-----------------|--------------------------|-----------------------------------------------------------------------------------------------------------|
| **CNN**         | VGG                      | `vgg16`, `vgg19`                                                                                          |
|                 | ResNet                   | `resnet50`, `resnet101`, `se_resnet50`                                                                    |
|                 | ResNeXt                  | `resnext101`                                                                                              |
|                 | DenseNet                 | `densenet121`, `densenet201`                                                                              |
|                 | Inception                | `inceptionv3`, `inceptionresnetv2`                                                                        |
|                 | Xception                 | `xception`                                                                                                |
|                 | EfficientNet             | `efficientnetb0`, `efficientnetb7`                                                                        |
|                 | MobileNet                | `mobilenetv2`                                                                                             |
|                 | NASNet                   | `nasnetmobile`                                                                                            |
|                 | ConvNeXt                 | `convnext_tiny`                                                                                           |
|                 | RegNet                   | `regnety_800mf`                                                                                           |
|                 | HRNet                    | `hrnet`                                                                                                   |
|                 | **BiT (Big Transfer)**   | `bit_r50x1`, `bit_r101x3`, `bit_r152x4`                                                                   |
|                 | **Noisy Student**        | `noisy_student_efficientnet_l2`, `noisy_student_efficientnet_b1` through `b6`                             |
| **Transformer** | **ViT (Vision Trans.)**  | `vit_base`, `vit_large`                                                                                   |
|                 | Swin Transformer         | `swin_transformer`                                                                                        |
|                 | DeiT                     | `deit_base`                                                                                               |
|                 | BEiT                     | `beit`                                                                                                    |
|                 | MobileViT                | `mobilevit`                                                                                               |
|                 | **PVT (Pyramid VT)**     | `pvt`                                                                                                     |
|                 | **T2T-ViT**              | `t2t-vit`                                                                                                 |
|                 | **PoolFormer**           | `poolformer`                                                                                              |
|                 | **Twins-SVT**            | `twins-svt`                                                                                               |
|                 | **EfficientFormer**      | `efficientformer-l1`                                                                                      |

</details>

## ⚡ Installation

As this library is packaged with `setup.py`, you can easily install it in your environment.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Install the library
# This command installs all dependencies from requirements.txt and makes
# the 'cnn_base' package available for import in your projects.
pip install .

# For developers: install in "editable" mode to reflect code changes instantly
# pip install -e .
```

## 🚀 Core Workflow: A Quick Start

This example demonstrates the core power of the library: easily instantiating, fine-tuning, and training different model architectures.

```python
from cnn_base.Models import get_model
from cnn_base.Data import DataLoader
import numpy as np

# 1. Load and preprocess your data
# This will create a 'processed_data' directory with .npy files
loader = DataLoader()
loader.fetch_and_save_data(data_dir='path/to/your/dataset')
X_train = np.load('processed_data/X_train.npy')
y_train = np.load('processed_data/y_train.npy')
X_val = np.load('processed_data/X_val.npy')
y_val = np.load('processed_data/y_val.npy')

# -------------------------------------------------------------------
# Example 1: Fine-tuning a Convolutional Neural Network (CNN)
# -------------------------------------------------------------------
print("--- Training a CNN ---")
model_cnn = get_model("efficientnetb0") # The factory returns a CNN_Model instance

# Apply a CNN-specific fine-tuning strategy
model_cnn.freeze_all()
model_cnn.unfreeze_later_n(30) # Unfreeze the last 30 layers
model_cnn.compile()

# Train the model
cnn_history = model_cnn.fit(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    epochs=10
)

# -------------------------------------------------------------------
# Example 2: Fine-tuning a Vision Transformer (ViT)
# -------------------------------------------------------------------
print("\n--- Training a Vision Transformer ---")
model_vit = get_model("vit_base") # The factory returns a Transformer_Model instance

# Apply a Transformer-specific fine-tuning strategy
model_vit.unfreeze_last_n_blocks(2) # Unfreeze only the last 2 transformer blocks

# Use an optimizer with Layer-wise Learning Rate Decay (LLRD)
model_vit.compile_with_llrd()

# Train the model
vit_history = model_vit.fit(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    epochs=10
)
```

### **Visualizing Model Performance and Interpretability**

After training your model, use the `Visualizer` class to gain deep insights into its performance and decision-making process.

```python
from cnn_base.utils import Visualizer
from cnn_base.loggers import Logger
from cnn_base.Models import CNN_Model, Transformer_Model
import numpy as np

# --- Setup -------------------------------------------------------------------
# Assume you have these variables available after training your model:
#
# model: The trained model object (either a CNN_Model or Transformer_Model).
# history: The history object returned by the model.fit() method.
# X_test, y_test: Your test dataset and corresponding true labels.
# class_names: A list of your class names, e.g., ['Cat', 'Dog'].
# -----------------------------------------------------------------------------

# 1. Initialize the Visualizer
visualizer = Visualizer()

# 2. Get model predictions for the test set
# We need both the class probabilities and the final predicted classes.
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# 3. Plot Training and Evaluation Metrics
print("--- Plotting Training and Evaluation Metrics ---")

# Visualize the training & validation loss and accuracy over epochs
visualizer.plot_training_history(history)

# Plot a confusion matrix to see class-wise performance
visualizer.plot_confusion_matrix(y_test, y_pred_classes, class_names=class_names)

# Plot the Receiver Operating Characteristic (ROC) curve for each class
visualizer.plot_roc_curve(y_test, y_pred_probs, n_classes=len(class_names))


# 4. Plot Model Interpretability to Understand "Why"
print("\n--- Plotting Model Interpretability ---")

# Select a single image from the test set to explain
sample_image = X_test[0]

# The library can use the best visualization method based on the model's architecture.
if isinstance(model, CNN_Model):
    # For CNNs, Grad-CAM highlights the "hotspots" the model used for its decision.
    # The method automatically tries to find the last convolutional layer.
    print("Model is a CNN. Generating Grad-CAM...")
    visualizer.plot_grad_cam(model.model, sample_image)

elif isinstance(model, Transformer_Model):
    # For Transformers, plotting the attention maps from the final block shows
    # which parts of the image the model "paid attention" to.
    print("Model is a Transformer. Generating Attention Maps...")
    visualizer.plot_attention_maps(model.model, sample_image)

```

## Advanced Usage Snippets

#### Cross-Validation

Easily run k-fold cross-validation to get a robust estimate of your model's performance.

```python
from cnn_base.Cross_Validation import Cross_Validator

# Pass a list of model names to validate
validator = Cross_Validator(
    model_names=["resnet50", "mobilenetv2"],
    X=X_train,
    y=y_train,
    n_splits=5
)

# The run method handles model creation, fine-tuning, and training for each fold
results_df = validator.run(epochs=5)
print(results_df)
validator.save_results("cross_validation_report.csv")
```

#### Hyperparameter Tuning

Use the built-in KerasTuner integration with predefined strategies for complex hyperparameter sweeps.

```python
from cnn_base.Tune_Hyperparameters import tune, strategies

# Use a predefined search space and strategy for Transformers
tuner = tune(
    strategy_function=strategies.transformer_fine_tuning_strategy_B,
    search_space_fn=strategies.search_space_B,
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    project_name="ViT_LLRD_Tuning"
)

# The results are automatically saved to a CSV file in 'storage/tuning_results/'
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:", best_hps.values)
```


## 🔮 Coming Soon

This library is under active development. Upcoming features planned to enhance the research workflow include:

-   **[Coming Soon] Automated Experiment Tracking:** A dedicated module to automatically log all experiment parameters, metrics, and artifacts to a central, queryable location.
-   **[Coming Soon] Advanced Evaluation:** Modules for robust evaluation techniques like Model Ensembling and Test-Time Augmentation (TTA).

## 🗂️ Project Structure

```
cnn_base/
├── configs/
│   ├── base_config.py
│   ├── cnn_config.py
│   └── transformers_config.py
├── Cross_Validation/
│   └── validator.py
├── Data/
│   └── data.py
├── loggers/
│   └── logger.py
├── Models/
│   ├── CNN/
│   │   ├── Callbacks/
│   │   ├── model.py
│   │   └── providers.py
│   ├── Transformers/
│   │   ├── Callbacks/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── providers.py
│   ├── base_model.py
│   └── get_model.py
├── Tune_Hyperparameters/
│   ├── strategies.py
│   └── tuner.py
└── utils/
    └── visualization.py
```