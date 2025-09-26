import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import re
import kagglehub
from ..loggers import Logger

class DataLoader:
    def __init__(self, logger: Logger = None):
        self.logger = logger if logger else Logger("Data_Loader", "data_info.log", "data_error.log")

    def _load_and_preprocess_image(self, image_path: str, image_size: tuple, mode: str) -> np.ndarray:
        try:
            img = Image.open(image_path).convert(mode)
            img = img.resize(image_size)
            img_array = np.array(img) / 255.0
            return img_array
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None

    def _detect_classes(self, data_dir: str) -> dict:
        extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        class_dict = {}
        for root, _, files in os.walk(data_dir):
            image_files = [f for f in files if f.lower().endswith(extensions)]
            if image_files:
                class_name = os.path.basename(root)
                class_dict[class_name] = [os.path.join(root, f) for f in image_files]
        return class_dict

    def _process_folder(self, folder_path: str, image_size: tuple, mode: str) -> tuple:
        class_dict = self._detect_classes(folder_path)
        all_images, all_labels = [], []
        class_names = sorted(class_dict.keys())
        self.logger.info(f"Found classes: {class_names} in {folder_path}")

        for label, class_name in enumerate(class_names):
            for image_path in class_dict[class_name]:
                img_array = self._load_and_preprocess_image(image_path, image_size, mode=mode)
                if img_array is not None:
                    all_images.append(img_array)
                    all_labels.append(label)
        
        if not all_images:
            raise ValueError(f"No valid images found in {folder_path}")
        
        return np.array(all_images), np.array(all_labels), class_names

    def _find_subfolder(self, base_dir: str, pattern: str) -> str:
        regex = re.compile(pattern + r"s?.*", re.IGNORECASE)
        for sub in os.listdir(base_dir):
            sub_path = os.path.join(base_dir, sub)
            if os.path.isdir(sub_path) and regex.search(sub):
                return sub_path
        return None

    def fetch_and_save_data(self, data_dir: str, output_dir: str = "processed_data", image_size: tuple = (224, 224), mode: str = "RGB", random_state: int = 42):
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Starting data processing from '{data_dir}' with image size {image_size}.")

        train_dir = self._find_subfolder(data_dir, "train")
        val_dir = self._find_subfolder(data_dir, "val(idation)?")
        test_dir = self._find_subfolder(data_dir, "test(ing)?")

        if train_dir:
            self.logger.info(f"Found training folder: {train_dir}")
            X_train, y_train, class_names = self._process_folder(train_dir, image_size, mode)
            
            if val_dir:
                self.logger.info(f"Found validation folder: {val_dir}")
                X_val, y_val, _ = self._process_folder(val_dir, image_size, mode)
            else:
                self.logger.warning("No validation folder found. Splitting train data 80/20 for train/validation.")
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state)

            if test_dir:
                self.logger.info(f"Found testing folder: {test_dir}")
                X_test, y_test, _ = self._process_folder(test_dir, image_size, mode)
            else:
                self.logger.warning("No testing folder found. Test data will be empty.")
                X_test, y_test = np.array([]), np.array([])
        else:
            self.logger.info("No structured train/val/test folders found. Splitting data 60/20/20...")
            X, y, class_names = self._process_folder(data_dir, image_size, mode)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=random_state)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state)

        data_splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'class_names': np.array(class_names)
        }

        for name, data in data_splits.items():
            if data.size > 0:
                np.save(os.path.join(output_dir, f'{name}.npy'), data)
                self.logger.info(f"Saved {name}.npy with shape: {data.shape}")

        self.logger.info(f"Data processing complete. Saved to '{output_dir}'.")

    @staticmethod
    def download_from_kaggle(path: str = None, dataset_name: str = "uraninjo/augmented-alzheimer-mri-dataset") -> str:
        logger = Logger("KaggleDownloader", "data_info.log", "data_error.log")
        logger.info(f"Downloading dataset '{dataset_name}' from Kaggle Hub...")
        if path :
            download_path = kagglehub.dataset_download(dataset_name, path)
        else :
            download_path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Dataset downloaded and unzipped to: {download_path}")
        return download_path