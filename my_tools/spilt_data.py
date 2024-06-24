import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, train_size=0.8):
    """
    Split each class folder in the dataset into training and testing sets.
    
    :param data_dir: Root directory containing class folders with images.
    :param train_size: Proportion of the dataset to include in the train split.
    """
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        train_files, test_files = train_test_split(image_files, train_size=train_size, random_state=42)
        
        # Create directories for train/test splits if not exist
        train_dir = os.path.join(data_dir, 'train', cls)
        test_dir = os.path.join(data_dir, 'test', cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Move files to the corresponding directories
        for file in train_files:
            shutil.move(os.path.join(class_dir, file), os.path.join(train_dir, file))
        for file in test_files:
            shutil.move(os.path.join(class_dir, file), os.path.join(test_dir, file))

# Example usage:
data_directory = '/home/zhangyang/ssd/Classification/mmpretrain/data/FUSAR/MarineCategory'  # Update this path to your dataset directory
split_dataset(data_directory)
