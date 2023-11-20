import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def random_color():
    return tuple(np.random.choice(range(256), size=3))

def create_mock_dataset(base_dir, num_images=10, image_size=(256, 256), more_circles_and_squares=False):
    # Define the paths
    train_image_path = os.path.join(base_dir, 'train', 'images')
    train_mask_path = os.path.join(base_dir, 'train', 'masks')
    val_image_path = os.path.join(base_dir, 'val', 'images')
    val_mask_path = os.path.join(base_dir, 'val', 'masks')

    # Create directories
    for path in [train_image_path, train_mask_path, val_image_path, val_mask_path]:
        os.makedirs(path, exist_ok=True)

    # Function to create an image with shapes and a corresponding mask
    def create_image_and_mask(image_id, folder):
        # Initialize image with a random background color
        image = Image.new('RGB', image_size, random_color())
        mask = Image.new('L', image_size, 0)

        draw = ImageDraw.Draw(image)
        mask_draw = ImageDraw.Draw(mask)

        # Add shapes
        num_shapes = np.random.randint(1, 6)  # Random number of shapes
        if more_circles_and_squares:
            num_shapes += 2  # Increase the number of shapes

        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rectangle'])
            shape_size = np.random.randint(20, 70)  # Random shape size
            x1, y1 = np.random.randint(0, image_size[0] - shape_size, size=2)
            x2, y2 = x1 + shape_size, y1 + shape_size

            if shape_type == 'circle':
                draw.ellipse([x1, y1, x2, y2], fill=random_color())
                mask_draw.ellipse([x1, y1, x2, y2], fill=1)  # Class 1
            elif shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=random_color())
                mask_draw.rectangle([x1, y1, x2, y2], fill=2)  # Class 2

        # Add random noise
        noise = np.random.randint(0, 50, image_size + (3,))  # Noise intensity
        noisy_image = np.array(image) + noise
        noisy_image = np.clip(noisy_image, 0, 255)  # Ensure values are within RGB range
        image = Image.fromarray(noisy_image.astype('uint8'))

        # Save the image and mask
        image.save(os.path.join(folder, 'images', f'image_{image_id}.jpg'))
        mask.save(os.path.join(folder, 'masks', f'image_{image_id}.png'))

    # Create images and masks for training and validation sets
    for i in range(num_images):
        if i < int(0.8 * num_images):
            create_image_and_mask(i, os.path.join(base_dir, 'train'))
        if i < int(0.2 * num_images):
            create_image_and_mask(i, os.path.join(base_dir, 'val'))

# Usage
create_mock_dataset('./mock_dataset', num_images=1000, image_size=(256, 256), more_circles_and_squares=True)
