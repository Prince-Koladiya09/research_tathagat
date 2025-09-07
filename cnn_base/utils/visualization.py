# utils/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)


class Visualizer:
    def __init__(self, logger):
        """
        logger: instance of your Logger class\n
        **Example:**

        ```python
        from loggers import Logger
        visualizer = Visualizer(logger=Logger())
        ```
        """
        self.logger = logger

    # --------------------------- MUST HAVE --------------------------- #
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, normalize=False):
        """
        **Description:**
        Plots a confusion matrix heatmap using Seaborn.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_pred: array-like, predicted labels.
        - class_names: list of strings, optional class names for axes.
        - normalize: bool, if True normalizes the confusion matrix.

        **Example:**
        ```python
        # y_true from dataset, y_pred from model predictions
        y_pred = np.argmax(model.predict(X_test), axis=1)
        visualizer.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, class_names=class_names)

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
            plt.show()
            self.logger.info("Confusion matrix plotted successfully")
            return cm
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {e}")
            return None

    def plot_roc_curve(self, y_true, y_prob, n_classes=2):
        """
        **Description:**
        Plots the ROC curve for binary or multi-class classification.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_prob: array-like, predicted probabilities.
        - n_classes: int, number of classes.

        **Example:**
        ```python
        # y_prob from model.predict (softmax outputs)
        y_prob = model.predict(X_test)
        visualizer.plot_roc_curve(y_true=y_test, y_prob=y_prob, n_classes=4)
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
            plt.show()
            self.logger.info("ROC curve plotted successfully")
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {e}")

    def plot_precision_recall(self, y_true, y_prob, n_classes=2):
        """
        **Description:**
        Plots the precision-recall curve for binary or multi-class classification.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_prob: array-like, predicted probabilities.
        - n_classes: int, number of classes.

        **Example:**
        ```python
        y_prob = model.predict(X_test)
        visualizer.plot_precision_recall(y_true=y_test, y_prob=y_prob, n_classes=4)
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
            plt.show()
            self.logger.info("Precision-Recall curve plotted successfully")
        except Exception as e:
            self.logger.error(f"Error plotting Precision-Recall curve: {e}")

    # --------------------------- NICE TO HAVE --------------------------- #
    def plot_training_history(self, history):
        """
        **Description:**
        Plots training and validation accuracy and loss over epochs.

        **Parameters:**
        - history: Keras History object.

        **Example:**
        ```python
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)
        visualizer.plot_training_history(history)
        ```
        """
        try:
            acc = history.history.get("accuracy", [])
            val_acc = history.history.get("val_accuracy", [])
            loss = history.history.get("loss", [])
            val_loss = history.history.get("val_loss", [])

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(acc, label="train acc")
            if val_acc:
                plt.plot(val_acc, label="val acc")
            plt.legend()
            plt.title("Accuracy")

            plt.subplot(1, 2, 2)
            plt.plot(loss, label="train loss")
            if val_loss:
                plt.plot(val_loss, label="val loss")
            plt.legend()
            plt.title("Loss")

            plt.show()
            self.logger.info("Training history plotted successfully")
        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}")

    def plot_classwise_metrics(self, y_true, y_pred, class_names=None):
        """
        **Description:**
        Plots precision, recall, and F1-score for each class.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_pred: array-like, predicted labels.
        - class_names: list of strings, optional class names.

        **Example:**
        ```python
        y_pred = np.argmax(model.predict(X_test), axis=1)
        visualizer.plot_classwise_metrics(y_true=y_test, y_pred=y_pred, class_names=class_names)
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
            plt.show()
            self.logger.info("Classwise metrics plotted successfully")
        except Exception as e:
            self.logger.error(f"Error plotting classwise metrics: {e}")

    def plot_cumulative_gain(self, y_true, y_prob):
        """
        **Description:**
        Plots cumulative gain curve.

        **Parameters:**
        - y_true: array-like, true labels.
        - y_prob: array-like, predicted probabilities.

        **Example:**
        ```python
        y_prob = model.predict(X_test)[:, 1]  # For binary classification
        visualizer.plot_cumulative_gain(y_true=y_test, y_prob=y_prob)
        ```
        """
        try:
            order = np.argsort(y_prob)[::-1]
            y_true_sorted = np.array(y_true)[order]
            cum_gains = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
            percentages = np.arange(1, len(y_true) + 1) / len(y_true)

            plt.plot(percentages, cum_gains, label="Model")
            plt.plot([0, 1], [0, 1], "--", label="Random")
            plt.xlabel("Proportion of sample")
            plt.ylabel("Cumulative Gain")
            plt.title("Cumulative Gain Curve")
            plt.legend()
            plt.show()
            self.logger.info("Cumulative Gain curve plotted successfully")
        except Exception as e:
            self.logger.error(f"Error plotting cumulative gain curve: {e}")

    def plot_confidence_histogram(self, y_prob, bins=20):
        """
        Plots a histogram of prediction confidence (max probability per sample).

        **Parameters**

        * `y_prob` : array-like
        Predicted probabilities.
        * `bins` : int, default=20
        Number of bins in histogram.

        **Example**

        ```python
        y_prob = model.predict(X_test)
        visualizer.plot_confidence_histogram(y_prob, bins=20)
        ```
        """
        try:
            confidences = np.max(y_prob, axis=1)
            sns.histplot(confidences, bins=bins, kde=True)
            plt.xlabel("Prediction Confidence")
            plt.ylabel("Frequency")
            plt.title("Prediction Confidence Histogram")
            plt.show()
            self.logger.info("Confidence histogram plotted successfully")
        except Exception as e:
            self.logger.error(f"Error plotting confidence histogram: {e}")

    def plot_grad_cam(self, model, img_array, last_conv_layer_name, original_img=None, alpha=0.4):
        """
        Plots Grad-CAM heatmap for a CNN model.

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

        **Example**

        ```python
        img_array = preprocess_input(np.expand_dims(img, axis=0))
        visualizer.plot_grad_cam(model, img_array, last_conv_layer_name='conv5_block3_out', original_img=img)
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
            conv_outputs = conv_outputs[0]

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
            heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

            img = original_img if original_img is not None else img_array[0]
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img.astype("uint8"), 1, heatmap, alpha, 0)

            plt.imshow(superimposed_img)
            plt.axis("off")
            plt.title("Grad-CAM")
            plt.show()
            self.logger.info("Grad-CAM plotted successfully")
        except Exception as e:
            self.logger.error(f"Error plotting Grad-CAM: {e}")

    def plot_embeddings(self, model=None, layer_name=None, data=None, labels=None, method="tsne", features=None, random_state=42):
        """
        Visualizes embeddings from a model layer using t-SNE or UMAP.

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

        **Example**

        ```python
        # Extract embeddings from a layer
        features, reduced = visualizer.plot_embeddings(
            model=model,
            layer_name='fc1',
            data=X_test,
            labels=y_test,
            method='tsne'
        )
        ```
        """
        try:
            import plotly.express as px
            from sklearn.manifold import TSNE
            import umap
            # Extract features if model is provided
            if features is None:
                if model is None or layer_name is None or data is None:
                    raise ValueError("Either provide features directly, or model, layer_name, and data.")
                
                intermediate_layer_model = tf.keras.Model(
                    inputs=model.model.input,
                    outputs=model.model.get_layer(layer_name).output
                )

                if isinstance(data, tf.data.Dataset):
                    features_list = []
                    for batch in data:
                        x_batch = batch[0] if isinstance(batch, tuple) else batch
                        features_list.append(intermediate_layer_model.predict(x_batch))
                    features = np.concatenate(features_list, axis=0)
                else:
                    features = intermediate_layer_model.predict(data)

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

            # Plot
            fig = px.scatter(x=reduced[:, 0], y=reduced[:, 1], color=labels, title=f"{method.upper()} Embeddings")
            fig.show()
            self.logger.info(f"{method.upper()} embeddings plotted successfully")

            return features, reduced

        except Exception as e:
            self.logger.error(f"Error plotting embeddings: {e}")
            return None, None
    
     # ------------------- EXPLAINABLE AI (XAI) METHODS ------------------- #

    def plot_lime_explanation(self, model, image, num_features=5, hide_rest=True):
        """
        Visualizes the LIME explanation for a single image prediction.

        **Parameters**
            model: The trained model.
            image (np.ndarray): The input image as a NumPy array.
            num_features (int): The number of superpixels to highlight.
            hide_rest (bool): If True, greys out the rest of the image.

        **Example**

        ```python
        visualizer.plot_lime_explanation(model, image=X_test[0], num_features=5, hide_rest=True)
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
            plt.show()
            self.logger.info("LIME explanation plotted successfully.")

        except Exception as e:
            self.logger.error(f"Error plotting LIME explanation: {e}")

    def plot_shap_explanation(self, model, background_data, images_to_explain, class_names=None):
        """
        Visualizes SHAP explanations for one or more images.

        **Parameters**
            model: The trained model.
            background_data (np.ndarray): A subset of the training data to use as a background for SHAP.
                                          A smaller, representative sample (e.g., 100 images) is usually sufficient.
            images_to_explain (np.ndarray): A single image or a batch of images to explain.
            class_names (list of str): A list of class names for the plot labels.

        **Example**

        ```python
        visualizer.plot_shap_explanation(model, background_data=X_train[:100], images_to_explain=X_test[:5], class_names=class_names)
        ```
        """
        try:
            import shap
            self.logger.info("Generating SHAP explanations...")

            # SHAP's DeepExplainer is optimized for deep learning models.
            explainer = shap.DeepExplainer(model, background_data)

            # Compute SHAP values
            shap_values = explainer.shap_values(images_to_explain)

            # If a single image was passed, wrap it in a list for consistent plotting
            if len(images_to_explain.shape) == 3:
                images_to_explain = np.expand_dims(images_to_explain, axis=0)
            
            # The output of shap_values for a multi-class model is a list of arrays,
            # one for each class. We need to reformat it for image_plot.
            # For TensorFlow models, the output shape is often [n_classes, n_samples, height, width, channels]
            # We need to transpose it to [n_samples, height, width, channels, n_classes]
            shap_values_transposed = [np.transpose(shap_values[i], (0, 2, 1, 3)) for i in range(len(shap_values))]
            shap_values_for_plot = np.transpose(np.array(shap_values_transposed), (1, 2, 3, 4, 0))


            # Visualize the explanations
            shap.image_plot(
                shap_values_for_plot,
                -images_to_explain,  # SHAP expects the original image values, we use negative for better visualization contrast
                labels=np.array([class_names] * len(images_to_explain)) if class_names else None
            )

            self.logger.info("SHAP explanations plotted successfully.")

        except Exception as e:
            self.logger.error(f"Error plotting SHAP explanations: {e}")