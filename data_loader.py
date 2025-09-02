import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import re

def load_and_preprocess_image(image_path,image_size=(128,128),mode='RGB'):
    try:
        img=Image.open(image_path).convert(mode)
        img=img.resize(image_size)
        img_array = np.array(img) / 255.0  
        return img_array
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
    
def detect_classes(data_dir, extensions=('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
    class_dict={}

    for root,dirs,files in os.walk(data_dir):
        image_files=[f for f in files if f.lower().endswith(extensions)]

        if image_files:
            class_name=os.path.basename(root)
            class_dict[class_name]=[os.path.join(root,f) for f in image_files]

    return class_dict


def process_folder (folder_path,mode='RGB'):
    
    class_dict = detect_classes(folder_path)
    all_images=[]
    all_labels=[]
    class_names=sorted(os.listdir(folder_path))

    print(f"Found classes: {class_names}")

    for label,class_name in enumerate(class_names):
        for image_path in class_dict[class_name]:
            img_array = load_and_preprocess_image(image_path, mode=mode)
            if img_array is not None:
                all_images.append(img_array)
                all_labels.append(label)
    if not all_images:
        raise ValueError(f"No valid images found in {folder_path}")

    return np.array(all_images), np.array(all_labels), class_names

def find_subfolder(base_dir, pattern):
    
    regex = re.compile(pattern + r"s?.*", re.IGNORECASE)

    for sub in os.listdir(base_dir):
        sub_lower = sub.lower()
        sub_path = os.path.join(base_dir, sub)

        if os.path.isdir(sub_path) and regex.search(sub_lower):
            return sub_path

    return None

def fetch_and_save_data(data_dir, output_dir="processed_data", mode="RGB", random_state=42):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = find_subfolder(data_dir, "train")
    val_dir   = find_subfolder(data_dir, "val(idation)?")   
    test_dir  = find_subfolder(data_dir, "test(ing)?") 

    if train_dir:
        print(f" Found training folder: {train_dir}")

        X_train,y_train,class_names =process_folder(train_dir,mode)
        np.save(os.path.join(output_dir,'X_train.npy'),X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'),y_train)


        if val_dir:
            print(f" Found validation folder: {val_dir}")
            X_val,y_val,_ = process_folder(val_dir, mode)
            np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
            np.save(os.path.join(output_dir, 'y_val.npy'), y_val)

        if test_dir:
            print(f" Found testing folder: {test_dir}")
            X_test,y_test,_ = process_folder(test_dir, mode)
            np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

        np.save(os.path.join(output_dir, 'class_names.npy'), np.array(class_names))
    
    else:
        # No structured folders found → do random split 60/20/20
        print(" No structured train/val/test folders found. Splitting data 60/20/20...")
        X, y, class_names = process_folder(data_dir, mode)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
        )

        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        np.save(os.path.join(output_dir, 'class_names.npy'), np.array(class_names))

    print(f" Data saved to {output_dir}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")