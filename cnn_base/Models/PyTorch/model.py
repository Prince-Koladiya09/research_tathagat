import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from cnn_base.loggers import Logger
from cnn_base.configs.base_config import Global_Config
from .providers import get_model as get_pytorch_model

class PyTorch_Model:
    """
    A wrapper for PyTorch timm models to provide a Keras-like interface.
    """
    def __init__(self, name: str, config: Global_Config = None):
        self.name = name
        self.logger = Logger(name=self.name)
        self.config = config or Global_Config()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"PyTorch model will run on device: {self.device}")

        # Get the base model from timm (feature extractor)
        self.base_model = get_pytorch_model(name).to(self.device)
        
        # Determine the number of features from the base model
        # This is a common way to get the feature dimension from timm models
        num_features = self.base_model.num_features
        if num_features == 0 and hasattr(self.base_model, 'feature_info'):
             num_features = self.base_model.feature_info[-1]['num_chs']
        
        # Create a new classification head
        self.classifier_head = nn.Linear(num_features, self.config.model.num_classes).to(self.device)
        
        # Combine into a single sequential model
        self.model = nn.Sequential(
            self.base_model,
            self.classifier_head
        ).to(self.device)

        self.optimizer = None
        self.loss_fn = None
        self.logger.info(f"PyTorch model '{name}' built successfully.")

    def compile(self, **kwargs):
        """Sets up the optimizer and loss function."""
        opt_config = self.config.optimizer
        if opt_config.name.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=opt_config.learning_rate)
        else: # Default to Adam
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt_config.learning_rate)
            
        self.loss_fn = nn.CrossEntropyLoss()
        self.logger.info(f"Model compiled with Optimizer: {opt_config.name} and Loss: CrossEntropyLoss")

    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: tuple = None, **kwargs):
        """Runs the training and validation loops."""
        if not self.optimizer:
            self.logger.warning("Optimizer not set. Compiling with default settings.")
            self.compile()
            
        # Convert numpy arrays to PyTorch Tensors and create DataLoaders
        train_dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True)
        
        if validation_data:
            val_dataset = TensorDataset(torch.from_numpy(validation_data[0]).float(), torch.from_numpy(validation_data[1]).long())
            val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size)

        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        for epoch in range(self.config.training.epochs):
            # Training loop
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            # Change image format from NHWC (TensorFlow) to NCHW (PyTorch)
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs} [Train]"):
                inputs = inputs.permute(0, 3, 1, 2).to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            history['loss'].append(avg_train_loss)
            history['accuracy'].append(train_acc)
            
            self.logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Validation loop
            if validation_data:
                self.model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs} [Val]"):
                        inputs = inputs.permute(0, 3, 1, 2).to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(val_acc)
                self.logger.info(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Create a simple history object for compatibility
        class History:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return History(history)

    def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Generates predictions for the input data."""
        self.model.eval()
        all_preds = []
        
        dataset = TensorDataset(torch.from_numpy(data).float())
        loader = DataLoader(dataset, batch_size=self.config.training.batch_size)
        
        with torch.no_grad():
            for (inputs,) in tqdm(loader, desc="Predicting"):
                inputs = inputs.permute(0, 3, 1, 2).to(self.device)
                outputs = self.model(inputs)
                # Apply softmax to get probabilities and move to CPU
                all_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
        
        return np.vstack(all_preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        """Evaluates the model and returns a dictionary of metrics."""
        self.model.eval()
        
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=self.config.training.batch_size)
        
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Evaluating"):
                inputs = inputs.permute(0, 3, 1, 2).to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total
        }
        
    def freeze_all(self):
        """Freezes all layers of the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.logger.info("Froze all base model layers.")
            
    def unfreeze_later_n(self, n: int):
        """Unfreezes the last n blocks/layers of the base model."""
        self.freeze_all()
        # Timm models often have a 'blocks' attribute
        if hasattr(self.base_model, 'blocks'):
            layers_to_unfreeze = self.base_model.blocks[-n:]
        elif hasattr(self.base_model, 'stages'):
             layers_to_unfreeze = self.base_model.stages[-n:]
        else: # Fallback to generic parameters
            layers_to_unfreeze = list(self.base_model.parameters())[-n*20:] # Heuristic
        
        for param in layers_to_unfreeze:
            param.requires_grad = True
        self.logger.info(f"Unfroze the last {n} blocks/stages.")

    def summary(self):
        """Prints a summary of the model."""
        try:
            from torchinfo import summary
            # We need to specify the input size and convert to NCHW format
            input_size = (self.config.training.batch_size, 3, *self.config.model.img_size)
            summary(self.model, input_size=input_size)
        except ImportError:
            self.logger.warning("`torchinfo` is not installed. Cannot print model summary. Run `pip install torchinfo`.")
            print(self.model)