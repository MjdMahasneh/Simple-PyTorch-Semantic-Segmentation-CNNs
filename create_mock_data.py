import os
import random
import numpy as np
from PIL import Image, ImageDraw

def random_color():
    return tuple(np.random.choice(range(256), size=3))

def create_mock_dataset(base_dir, num_images=10, image_size=(256, 256), more_shapes=False):
    # Define the paths
    train_image_path = os.path.join(base_dir, 'train', 'images')
    train_mask_path = os.path.join(base_dir, 'train', 'masks')
    val_image_path = os.path.join(base_dir, 'val', 'images')
    val_mask_path = os.path.join(base_dir, 'val', 'masks')

    # Create directories
    for path in [train_image_path, train_mask_path, val_image_path, val_mask_path]:
        os.makedirs(path, exist_ok=True)

    def create_image_and_mask(image_id, folder):
        image = Image.new('RGB', image_size, random_color())
        mask = Image.new('L', image_size, 0)  # Background class (0)

        draw = ImageDraw.Draw(image)
        mask_draw = ImageDraw.Draw(mask)

        num_shapes = np.random.randint(1, 100)  # Random number of shapes

        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rectangle', 'other'])
            shape_size = np.random.randint(20, 70)  # Random shape size
            x1, y1 = np.random.randint(0, image_size[0] - shape_size, size=2)
            x2, y2 = x1 + shape_size, y1 + shape_size

            if shape_type == 'circle':
                draw.ellipse([x1, y1, x2, y2], fill=random_color())
                mask_draw.ellipse([x1, y1, x2, y2], fill=1)  # Class 1
            elif shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=random_color())
                mask_draw.rectangle([x1, y1, x2, y2], fill=2)  # Class 2
            elif shape_type == 'other' and more_shapes:
                # Additional shapes treated as background
                draw_polygon(draw, mask_draw, x1, y1, x2, y2)

        image.save(os.path.join(folder, 'images', f'image_{image_id}.jpg'))
        mask.save(os.path.join(folder, 'masks', f'image_{image_id}.png'))

    def draw_polygon(draw, mask_draw, x1, y1, x2, y2):
        # Choose the number of points for the polygon (3, 5, or 6)
        num_points = random.choice([3, 5, 6, 8, 10, 12])
        points = []

        for _ in range(num_points):
            point_x = random.randint(x1, x2)
            point_y = random.randint(y1, y2)
            points.append((point_x, point_y))

        # Draw a polygon with the chosen points
        draw.polygon(points, fill=random_color())
        # Treat as background in the mask
        mask_draw.polygon(points, fill=0)



    for i in range(num_images):
        if i < int(0.8 * num_images):
            create_image_and_mask(i, os.path.join(base_dir, 'train'))
        if i < int(0.2 * num_images):
            create_image_and_mask(i, os.path.join(base_dir, 'val'))

create_mock_dataset('./mock_dataset', num_images=1000, image_size=(256, 256), more_shapes=True)
