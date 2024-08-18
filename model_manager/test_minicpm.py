import math  # noqa: F401
import os  # noqa: F401
import sys  # noqa: F401

import matplotlib.pyplot as plt
# First, import the slice_image function
from minicpm25.modeling_minicpmv import slice_image
from PIL import Image


def test_minicpm_slice_image(image_path):
    # Load a sample image
    image = Image.open(image_path)

    # Call the slice_image function
    source_image, patches, best_grid = slice_image(image)

    # Display the results
    fig = plt.figure(figsize=(15, 10))

    # Display original image
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image)
    ax.set_title("Original Image")
    ax.axis('off')

    # Display source image (resized/resampled image)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(source_image)
    ax.set_title(f"Source Image (Resized to {source_image.size})")
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    # If patches were created, display them
    if patches:
        cols, rows = best_grid
        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        fig.suptitle(f"Image Patches (Grid: {rows}x{cols})")

        for i in range(rows):
            for j in range(cols):
                if rows > 1 and cols > 1:
                    ax = axs[i, j]
                elif rows > 1:
                    ax = axs[i]
                elif cols > 1:
                    ax = axs[j]
                else:
                    ax = axs
                print(f"Patch image size: {patches[i][j].size}")
                ax.imshow(patches[i][j])
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    # Print additional information
    print(f"Original image size: {image.size}")
    print(f"Source image size: {source_image.size}")
    print(f"Best grid: {best_grid}")
    print(f"Number of patches: {sum(len(row) for row in patches)}")


if __name__ == "__main__":
    image_path = "path/to/your/sample/image.jpg"  # Replace with actual path
    test_minicpm_slice_image(image_path)
