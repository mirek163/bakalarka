import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def display_random_images(folder_path):
    # Get a list of image files in the specified folder
    image_files = glob.glob("C:/.develop/bakalarka/data/input/region/*.png")

    # Randomly select 3x4 image files
    selected_files = random.sample(image_files, 3 * 4)

    # Load and display the selected images in a grid
    fig, axes = plt.subplots(3, 4, figsize=(5, 2))

    for i, file in enumerate(selected_files):
        image = Image.open(file)
        axes[i // 4, i % 4].imshow(image)
        axes[i // 4, i % 4].axis("off")

    plt.tight_layout()
    plt.show()

# Specify the folder path where the images are located
folder_path = "data/input/region"

# Display a 3x4 grid of random images from the specified folder
display_random_images(folder_path)
