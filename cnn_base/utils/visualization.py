# utils/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
import math
import os
from cnn_base.loggers import Logger
from cnn_base.Models.base_model import Base_Model


class Visualizer:
    def __init__(self, logger = None):
        if not logger :
            logger = Logger("Visualization_Logger", "visualization_info.log", "visualization_error.log")
        self.logger = logger

    # --------------------------- MUST HAVE --------------------------- #
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, normalize=False, filepath=None):
        """
        **Description:**
        Plots a confusion matrix heatmap using Seaborn and saves to file.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_pred: array-like, predicted labels.
        - class_names: list of strings, optional class names for axes.
        - normalize: bool, if True normalizes the confusion matrix.
        - filepath: str, path to save the plot.

        **Example:**
        ```python
        visualizer.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, class_names=class_names, filepath="confusion_matrix.png")
        ```
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap="Blues",
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto",
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"Confusion matrix saved to {filepath}")
            else:
                plt.show()
            plt.close()
            return cm
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {e}")
            return None

    def plot_roc_curve(self, y_true, y_prob, n_classes=2, filepath=None):
        """
        **Description:**
        Plots the ROC curve for binary or multi-class classification and saves to file.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_prob: array-like, predicted probabilities.
        - n_classes: int, number of classes.
        - filepath: str, path to save the plot.

        **Example:**
        ```python
        visualizer.plot_roc_curve(y_true=y_test, y_prob=y_prob, n_classes=4, filepath="roc_curve.png")
        ```
        """
        try:
            plt.figure(figsize=(8, 6))
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc_score:.2f})")
            else:
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
                    auc_score = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc_score:.2f})")

            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"ROC curve saved to {filepath}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {e}")

    def plot_precision_recall(self, y_true, y_prob, n_classes=2, filepath=None):
        """
        **Description:**
        Plots the precision-recall curve for binary or multi-class classification and saves to file.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_prob: array-like, predicted probabilities.
        - n_classes: int, number of classes.
        - filepath: str, path to save the plot.

        **Example:**
        ```python
        visualizer.plot_precision_recall(y_true=y_test, y_prob=y_prob, n_classes=4, filepath="precision_recall.png")
        ```
        """
        try:
            plt.figure(figsize=(8, 6))
            if n_classes == 2:
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                ap = average_precision_score(y_true, y_prob[:, 1])
                plt.plot(recall, precision, label=f"AP={ap:.2f}")
            else:
                for i in range(n_classes):
                    precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_prob[:, i])
                    ap = average_precision_score((y_true == i).astype(int), y_prob[:, i])
                    plt.plot(recall, precision, label=f"Class {i} (AP={ap:.2f})")

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend(loc="best")
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"Precision-Recall curve saved to {filepath}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting Precision-Recall curve: {e}")

    # --------------------------- NICE TO HAVE --------------------------- #
    def plot_training_history(self, history, filepath=None):
        """
        **Description:**
        Plots training and validation accuracy and loss over epochs and saves to file.

        **Parameters:**
        - history: Keras History object.
        - filepath: str, path to save the plot.

        **Example:**
        ```python
        visualizer.plot_training_history(history, filepath="training_history.png")
        ```
        """
        try:
            acc = history.history.get("accuracy", [])
            val_acc = history.history.get("val_accuracy", [])
            loss = history.history.get("loss", [])
            val_loss = history.history.get("val_loss", [])
            params = {}

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(acc, label="train acc", marker=".", **params.get("accuracy", {}))
            if val_acc:
                plt.plot(val_acc, label="val acc", **params.get("val_accuracy", {}))
            plt.legend()
            plt.title("Accuracy")

            plt.subplot(1, 2, 2)
            plt.plot(loss, label="train loss", **params.get("loss", {}))
            if val_loss:
                plt.plot(val_loss, label="val loss", **params.get("val_loss", {}))
            plt.legend()
            plt.title("Loss")

            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"Training history saved to {filepath}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}")

    def plot_classwise_metrics(self, y_true, y_pred, class_names=None, filepath=None):
        """
        **Description:**
        Plots precision, recall, and F1-score for each class and saves to file.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_pred: array-like, predicted labels.
        - class_names: list of strings, optional class names.
        - filepath: str, path to save the plot.

        **Example:**
        ```python
        visualizer.plot_classwise_metrics(y_true=y_test, y_pred=y_pred, class_names=class_names, filepath="classwise_metrics.png")
        ```
        """
        try:
            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names if class_names else None,
                output_dict=True
            )
            classes = class_names if class_names else list(report.keys())[:-3]
            precision = [report[c]['precision'] for c in classes]
            recall = [report[c]['recall'] for c in classes]
            f1 = [report[c]['f1-score'] for c in classes]

            df = {
                "Class": classes,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            }

            plt.figure(figsize=(10, 6))
            sns.barplot(x="Class", y="value", hue="variable",
                        data=pd.melt(pd.DataFrame(df), ["Class"]))
            plt.title("Class-wise Metrics")
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"Classwise metrics saved to {filepath}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting classwise metrics: {e}")

    def plot_confidence_histogram(self, y_prob, bins=20, filepath=None):
        """
        Plots a histogram of prediction confidence (max probability per sample) and saves to file.

        **Parameters**
        * `y_prob` : array-like
        Predicted probabilities.
        * `bins` : int, default=20
        Number of bins in histogram.
        * `filepath` : str, path to save the plot.

        **Example**
        ```python
        visualizer.plot_confidence_histogram(y_prob, bins=20, filepath="confidence_histogram.png")
        ```
        """
        try:
            confidences = np.max(y_prob, axis=1)
            sns.histplot(confidences, bins=bins, kde=True)
            plt.xlabel("Prediction Confidence")
            plt.ylabel("Frequency")
            plt.title("Prediction Confidence Histogram")
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"Confidence histogram saved to {filepath}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting confidence histogram: {e}")

    def plot_cumulative_gain(self, y_true, y_prob, filepath=None):
        """
        **Description:**
        Plots cumulative gain curve and saves to file.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_prob: array-like, predicted probabilities.
        - filepath: str, path to save the plot.

        **Example:**
        ```python
        y_prob = model.predict(X_test)[:, 1]  # For binary classification
        visualizer.plot_cumulative_gain(y_true=y_test, y_prob=y_prob, filepath="cumulative_gain.png")
        ```
        """
        try:
            order = np.argsort(y_prob)[::-1]
            y_true_sorted = np.array(y_true)[order]
            cum_gains = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
            percentages = np.arange(1, len(y_true) + 1) / len(y_true)

            plt.figure(figsize=(8, 6))
            plt.plot(percentages, cum_gains, label="Model")
            plt.plot([0, 1], [0, 1], "--", label="Random")
            plt.xlabel("Proportion of sample")
            plt.ylabel("Cumulative Gain")
            plt.title("Cumulative Gain Curve")
            plt.legend()
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"Cumulative Gain curve saved to {filepath}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting cumulative gain curve: {e}")

    def plot_grad_cam(self, model, img_array, last_conv_layer_name, original_img=None, alpha=0.4, filepath=None):
        """
        Plots Grad-CAM heatmap for a CNN model and saves to file.

        **Parameters**
        * `model` : keras.Model
        Trained model.
        * `img_array` : np.ndarray
        Preprocessed input image array of shape (1, H, W, C).
        * `last_conv_layer_name` : str
        Name of the last convolutional layer.
        * `original_img` : np.ndarray, optional
        Raw image for overlay. Defaults to `img_array[0]`.
        * `alpha` : float, default=0.4
        Weight of heatmap overlay.
        * `filepath` : str, path to save the plot.

        **Example**
        ```python
        visualizer.plot_grad_cam(model, img_array, last_conv_layer_name='conv5_block3_out', original_img=img, filepath="grad_cam.png")
        ```
        """
        try:
            import cv2
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                pred_index = tf.argmax(predictions[0])
                loss = predictions[:, pred_index]

            grads = tape.gradient(loss, conv_outputs)[0]
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1).numpy()
            heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
            
            img = original_img if original_img is not None else img_array[0]
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            superimposed_img = cv2.addWeighted(img.astype('uint8'), 1, heatmap.astype('uint8'), alpha, 0)

            plt.figure(figsize=(8, 8))
            plt.imshow(superimposed_img)
            plt.axis("off")
            plt.title("Grad-CAM")
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"Grad-CAM saved to {filepath}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting Grad-CAM: {e}")

    def plot_attention_maps(self, model, preprocessed_image, target_size, filepath=None):
        """
        Visualizes the attention maps from a Vision Transformer model and saves to file.
        
        Args:
            model (tf.keras.Model): The trained transformer model.
            preprocessed_image (np.ndarray): A single preprocessed image with batch dimension (1, H, W, C).
            target_size (tuple): The (height, width) of the original image for resizing maps.
            filepath (str): Path to save the plot.
        """
        self.logger.info("Attempting to visualize attention maps...")
        try:
            outputs = [layer.output for layer in model.layers if "attention" in layer.name.lower() and hasattr(layer, 'attention_scores')]
            
            if not outputs:
                 outputs = [layer.output for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]

            if not outputs:
                self.logger.error("Could not find any attention layers in the model.")
                return

            attention_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
            
            att_maps = attention_model.predict(preprocessed_image)
            
            last_layer_maps = att_maps[-1]

            # Shape is likely (batch_size, num_heads, sequence_len, sequence_len)
            # We are interested in the attention from the [CLS] token to other patches
            cls_token_attention = last_layer_maps[0, :, 0, 1:] # Exclude CLS token itself
            
            num_heads = cls_token_attention.shape[0]
            num_patches = cls_token_attention.shape[1]
            patch_dim = int(math.sqrt(num_patches))
            
            if patch_dim * patch_dim != num_patches:
                self.logger.error(f"Cannot form a square grid from {num_patches} patches.")
                return

            # Plot all heads
            fig, axes = plt.subplots(math.ceil(num_heads / 4), 4, figsize=(12, 12))
            fig.suptitle("Attention Maps from CLS Token (Last Layer)")
            axes = axes.ravel()
            
            for i in range(num_heads):
                ax = axes[i]
                attention_grid = cls_token_attention[i].reshape(patch_dim, patch_dim)
                im = ax.imshow(attention_grid, cmap='viridis')
                ax.set_title(f'Head {i+1}')
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"Attention maps saved to {filepath}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Failed to plot attention maps: {e}")


    def plot_embeddings(self, model=None, layer_name=None, data=None, labels=None, method="tsne", features=None, random_state=42, filepath=None):
        """
        Visualizes embeddings from a model layer using t-SNE or UMAP and saves to file.

        **Parameters**
        * `model` : keras.Model, optional
        Model instance to extract features. Required if `features` not provided.
        * `layer_name` : str, optional
        Name of layer to extract embeddings from. Required if `features` not provided.
        * `data` : np.ndarray or tf.data.Dataset, optional
        Input images for feature extraction. Required if `features` not provided.
        * `labels` : array-like, optional
        Labels for coloring points.
        * `method` : str, default="tsne"
        Dimensionality reduction method: "tsne" or "umap".
        * `features` : np.ndarray, optional
        Precomputed embeddings to visualize.
        * `random_state` : int, default=42
        Random seed for reproducibility.
        * `filepath` : str, path to save the plot.

        **Example**
        ```python
        features, reduced = visualizer.plot_embeddings(
            model=model,
            layer_name='fc1',
            data=X_test,
            labels=y_test,
            method='tsne',
            filepath='embeddings.png'
        )
        ```
        """
        try:
            from sklearn.manifold import TSNE
            import umap
            
            # Extract features if model is provided
            if features is None:
                if model is None or layer_name is None or data is None:
                    raise ValueError("Either provide features directly, or model, layer_name, and data.")
                
                intermediate_layer_model = tf.keras.Model(
                    inputs=model.inputs,
                    outputs=model.get_layer(layer_name).output
                )

                if isinstance(data, tf.data.Dataset):
                    features_list = []
                    for batch in data:
                        x_batch = batch[0] if isinstance(batch, tuple) else batch
                        features_list.append(intermediate_layer_model.predict(x_batch))
                    features = np.concatenate(features_list, axis=0)
                else:
                    features = intermediate_layer_model.predict(data, batch_size = 32   )

                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)

            # Dimensionality reduction
            if method.lower() == "tsne":
                reducer = TSNE(n_components=2, random_state=random_state)
            elif method.lower() == "umap":
                reducer = umap.UMAP(n_components=2, random_state=random_state)
            else:
                raise ValueError("method must be 'tsne' or 'umap'")

            reduced = reducer.fit_transform(features)

            # Plot using matplotlib instead of plotly for easier saving
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
            plt.title(f"{method.upper()} Embeddings")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"{method.upper()} embeddings saved to {filepath}")
            else:
                plt.show()
            plt.close()

            return features, reduced

        except Exception as e:
            self.logger.error(f"Error plotting embeddings: {e}")
            return None, None
    
     # ------------------- EXPLAINABLE AI (XAI) METHODS ------------------- #

    def plot_lime_explanation(self, model, image, num_features=5, hide_rest=True, filepath=None):
        """
        Visualizes the LIME explanation for a single image prediction and saves to file.

        **Parameters**
            model: The trained model.
            image (np.ndarray): The input image as a NumPy array.
            num_features (int): The number of superpixels to highlight.
            hide_rest (bool): If True, greys out the rest of the image.
            filepath (str): Path to save the plot.

        **Example**
        ```python
        visualizer.plot_lime_explanation(model, image=X_test[0], num_features=5, hide_rest=True, filepath="lime_explanation.png")
        ```
        """
        try:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries

            self.logger.info("Generating LIME explanation...")
            explainer = lime_image.LimeImageExplainer()

            # LIME needs a function that takes a NumPy array of images and returns predictions
            def predict_fn(images):
                return model.predict(images)

            explanation = explainer.explain_instance(
                image.astype('double'),
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000  # Number of perturbed samples to generate
            )

            # Get the explanation for the top class
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=num_features,
                hide_rest=hide_rest
            )

            # Plot the explanation
            plt.figure(figsize=(6, 6))
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            plt.title("LIME Explanation")
            plt.axis('off')
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"LIME explanation saved to {filepath}")
            else:
                plt.show()
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting LIME explanation: {e}")

    def plot_shap_explanation(self, model, background_data, images_to_explain, class_names=None, filepath=None):
        """
        Visualizes SHAP explanations for one or more images and saves to file.

        **Parameters**
            model: The trained model.
            background_data (np.ndarray): A subset of the training data to use as a background for SHAP.
            images_to_explain (np.ndarray): A single image or a batch of images to explain.
            class_names (list of str): A list of class names for the plot labels.
            filepath (str): Path to save the plot.

        **Example**
        ```python
        visualizer.plot_shap_explanation(model, background_data=X_train[:100], images_to_explain=X_test[:5], class_names=class_names, filepath="shap_explanation.png")
        ```
        """
        try:
            import shap
            self.logger.info("Generating SHAP explanations...")

            explainer = shap.DeepExplainer(model, background_data)

            if len(images_to_explain.shape) == 3:
                images_to_explain = np.expand_dims(images_to_explain, axis=0)

            shap_values = explainer.shap_values(images_to_explain)
            
            # shap_values_transposed = [np.transpose(shap_values[i], (0, 2, 1, 3)) for i in range(len(shap_values))]
            # shap_values_for_plot = np.transpose(np.array(shap_values_transposed), (1, 2, 3, 4, 0))

            plt.figure(figsize=(12, 8))
            
            shap.image_plot(
                shap_values,
                # shap_values_for_plot,
                images_to_explain,  # Pass the original images, not negative ones
                # -images_to_explain,  # SHAP expects the original image values
                labels=np.array([class_names] * len(images_to_explain)) if class_names else None,
                show = False
            )
            
            if filepath:
                plt.savefig(filepath, bbox_inches='tight', dpi=300)
                self.logger.info(f"SHAP explanations saved to {filepath}")
            else:
                plt.show()
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting SHAP explanations: {e}")
    
    # ------------------- CROSS-VALIDATION VISUALIZATION METHODS ------------------- #

    def create_cv_plots(self, model_wrapper : Base_Model, X_val, y_val, history, fold_dir, cnn_model=True, class_names=None, 
                       create_xai_plots=True, create_embedding_plots=True, background_data=None, verbose = 0):
        """
        Create all visualization plots for a cross-validation fold and save to directory.
        
        Parameters:
        - model_wrapper: The model wrapper containing the trained model
        - X_val: Validation features
        - y_val: Validation labels
        - history: Training history object
        - fold_dir: Directory to save plots for this fold
        - cnn_model: bool, if True creates CNN-specific plots, else transformer-specific plots
        - class_names: List of class names for labeling
        - create_xai_plots: bool, whether to create XAI plots (LIME, SHAP)
        - create_embedding_plots: bool, whether to create embedding plots
        - background_data: Background data for SHAP explanations
        """
        try:
            # Get predictions
            y_pred_prob = model_wrapper.predict(X_val)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val
            
            n_classes = y_pred_prob.shape[1] if len(y_pred_prob.shape) > 1 else 2
            
            # Create directory if it doesn't exist
            os.makedirs(fold_dir, exist_ok=True)


            
            # Basic plots for all models
            self.plot_confusion_matrix(
                y_true, y_pred, class_names=class_names, 
                filepath=os.path.join(fold_dir, "confusion_matrix.png")
            )
            
            self.plot_roc_curve(
                y_true, y_pred_prob, n_classes=n_classes,
                filepath=os.path.join(fold_dir, "roc_curve.png")
            )
            
            self.plot_precision_recall(
                y_true, y_pred_prob, n_classes=n_classes,
                filepath=os.path.join(fold_dir, "precision_recall.png")
            )
            
            self.plot_training_history(
                history, 
                filepath=os.path.join(fold_dir, "training_history.png")
            )
            
            self.plot_classwise_metrics(
                y_true, y_pred, class_names=class_names,
                filepath=os.path.join(fold_dir, "classwise_metrics.png")
            )
            
            self.plot_confidence_histogram(
                y_pred_prob,
                filepath=os.path.join(fold_dir, "confidence_histogram.png")
            )
            
            if n_classes == 2 :
                self.plot_cumulative_gain(
                    y_true, y_pred_prob[:, 1],
                    filepath=os.path.join(fold_dir, "cumulative_gain.png")
                )
            
            # Model-specific plots
            if cnn_model:
                self._create_cnn_specific_plots(model_wrapper, X_val, fold_dir)
            else:
                self._create_transformer_specific_plots(model_wrapper, X_val, fold_dir)
            
            # XAI plots
            if create_xai_plots:
                self._create_xai_plots(model_wrapper, X_val, fold_dir, background_data, class_names)
            
            # Embedding plots
            # if create_embedding_plots:
            #     self._create_embedding_plots(model_wrapper, X_val, y_true, fold_dir)
                
            self.logger.info(f"All plots saved to {fold_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating cross-validation plots: {e}")

    def _create_cnn_specific_plots(self, model_wrapper, X_val, fold_dir):
        """Create CNN-specific visualization plots."""
        try:
            # Try to create Grad-CAM for a sample image
            if len(X_val) > 0:
                sample_idx = 0
                sample_image = X_val[sample_idx:sample_idx+1]
                
                # Find convolutional layers for Grad-CAM
                conv_layers = [layer.name for layer in model_wrapper.model.layers 
                             if isinstance(layer, layers.Conv2D)]
                
                if conv_layers:
                    last_conv_layer = conv_layers[-1]
                    self.plot_grad_cam(
                        model_wrapper.model, sample_image, last_conv_layer,
                        filepath=os.path.join(fold_dir, "grad_cam.png")
                    )
        except Exception as e:
            self.logger.warning(f"Could not create CNN-specific plots: {e}")

    def _create_transformer_specific_plots(self, model_wrapper, X_val, fold_dir):
        """Create transformer-specific visualization plots."""
        try:
            # Try to create attention maps for a sample image
            if len(X_val) > 0:
                sample_idx = 0
                sample_image = X_val[sample_idx:sample_idx+1]
                
                self.plot_attention_maps(
                    model_wrapper.model, sample_image, 
                    target_size=(224, 224),  # Adjust based on your model
                    filepath=os.path.join(fold_dir, "attention_maps.png")
                )
        except Exception as e:
            self.logger.warning(f"Could not create transformer-specific plots: {e}")

    def _create_xai_plots(self, model_wrapper, X_val, fold_dir, background_data, class_names):
        """Create XAI (LIME and SHAP) plots."""
        try:
            if len(X_val) > 0:
                # LIME explanation for a sample image
                sample_idx = 0
                sample_image = X_val[sample_idx]
                
                self.plot_lime_explanation(
                    model_wrapper.model, sample_image,
                    filepath=os.path.join(fold_dir, "lime_explanation.png")
                )
                
                # # SHAP explanations (use a small subset for performance)
                # if background_data is not None and len(X_val) >= 3:
                #     shap_samples = X_val[:3]  # Use first 3 samples for SHAP
                #     self.plot_shap_explanation(
                #         model_wrapper.model, background_data, shap_samples, class_names,
                #         filepath=os.path.join(fold_dir, "shap_explanation.png")
                #     )
                    
        except Exception as e:
            self.logger.warning(f"Could not create XAI plots: {e}")

    def _create_embedding_plots(self, model_wrapper, X_val, y_true, fold_dir):
        """Create embedding visualization plots."""
        try:
            if len(X_val) > 0:
                # Try to find suitable layers for embedding extraction
                embedding_layers = []
                for layer in model_wrapper.model.layers:
                    # layer_name = layer.name.lower()
                    # Look for dense, embedding, or flatten layers
                    if isinstance(layer, (layers.Dense, layers.Embedding, layers.Flatten, layers.GlobalAveragePooling1D, layers.GlobalAveragePooling2D)):
                        try :
                            if len(layer.output_shape) == 2:  # 2D output suitable for embeddings
                                embedding_layers.append(layer.name)
                        except :
                            pass
                
                if embedding_layers:
                    # Use the last suitable layer
                    embedding_layer = embedding_layers[-1]
                    
                    # Create t-SNE embeddings
                    self.plot_embeddings(
                        model=model_wrapper.model,
                        layer_name=embedding_layer,
                        data=X_val[:100],  # Use subset for performance
                        labels=y_true[:100],
                        method="tsne",
                        filepath=os.path.join(fold_dir, "tsne_embeddings.png")
                    )
                    
                    # Create UMAP embeddings
                    self.plot_embeddings(
                        model=model_wrapper.model,
                        layer_name=embedding_layer,
                        data=X_val[:100],
                        labels=y_true[:100],
                        method="umap",
                        filepath=os.path.join(fold_dir, "umap_embeddings.png")
                    )
                    
        except Exception as e:
            self.logger.warning(f"Could not create embedding plots: {e}")