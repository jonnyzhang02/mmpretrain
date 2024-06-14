import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_mean_std(directory):
    """
    Calculate the mean and standard deviation for grayscale images in the dataset without normalizing.
    This function will traverse all subdirectories to find images.

    :param directory: Path to the directory containing images.
    :return: (mean, std) tuple.
    """
    pixel_sum = 0.0
    pixel_sum_squared = 0.0
    num_pixels = 0

    # Traverse directory for image files
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                try:
                    with Image.open(filepath) as img:
                        img = np.array(img, dtype=np.float64)  # Use original pixel values
                        if img.ndim == 2:  # Ensure image is grayscale
                            pixels = img.shape[0] * img.shape[1]
                            num_pixels += pixels
                            pixel_sum += img.sum()
                            pixel_sum_squared += (img ** 2).sum()
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    if num_pixels > 0:
        mean = pixel_sum / num_pixels
        std = np.sqrt(pixel_sum_squared / num_pixels - mean ** 2)
        return mean, std
    else:
        return None, None

# Example usage
if __name__ == "__main__":
    dataset_path = 'data/SAR-ACD/train'  # Adjust this path to point to your 'train' directory
    mean, std = calculate_mean_std(dataset_path)
    if mean is not None and std is not None:
        print(f"Mean: {mean}")
        print(f"Standard Deviation: {std}")
    else:
        print("No images were processed. Check the dataset path and image formats.")
