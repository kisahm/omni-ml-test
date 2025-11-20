"""
Create test images for benchmarking
"""
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_test_images(output_dir: str, count: int = 100):
    """Create a set of test images"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {count} test images in {output_dir}/")

    for i in range(count):
        # Create random image or simple pattern
        if i % 2 == 0:
            # Random noise
            img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
        else:
            # Simple colored square
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
            ]
            color = colors[i % len(colors)]
            img = Image.new('RGB', (32, 32), color)

            # Add some pattern
            draw = ImageDraw.Draw(img)
            draw.rectangle([8, 8, 24, 24], outline=(255, 255, 255), width=2)

        # Save
        img.save(os.path.join(output_dir, f'test_{i:04d}.png'))

        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{count} images")

    print(f"Done! Created {count} images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create test images')
    parser.add_argument('--output', '-o', default='./test_images',
                        help='Output directory')
    parser.add_argument('--count', '-n', type=int, default=100,
                        help='Number of images to create')

    args = parser.parse_args()
    create_test_images(args.output, args.count)
