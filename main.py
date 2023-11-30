from PIL import Image
from collections import OrderedDict
import numpy as np


def greyscale(image):
    # Go through each pixel and change it to black and white
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = image.getpixel((x, y))

            # calculate the average value of rgb
            avg = (r + g + b) // 3

            # set the pixel to the new color 'avg'
            image.putpixel((x, y), (avg, avg, avg))

    return image


def linear_blur(image, strength):
    img_array = np.array(image)
    output_array = np.zeros_like(img_array)

    height = image.height
    width = image.width

    strength = int(strength * min(height, width))

    for y in range(height):
        for x in range(width):
            x_min = max(0, x - strength)
            x_max = min(width, x + strength + 1)
            y_min = max(0, y - strength)
            y_max = min(height, y + strength + 1)

            # Extract the surrounding box and calculate the mean
            surrounding_box = img_array[y_min:y_max, x_min:x_max]
            output_array[y, x] = np.mean(surrounding_box, axis=(0, 1)).astype(img_array.dtype)

    # Convert the numpy array back to PIL image
    return Image.fromarray(output_array)


def gaussian_kernel(size, sigma):
    # Create a Gaussian kernel
    kernel = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            kernel[x, y] = np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def gaussian_blur(image, strength, sigma):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    height = image.height
    width = image.width

    strength = int(strength * min(height, width))

    kernel_size = 2 * strength + 1
    kernel = gaussian_kernel(kernel_size, sigma)

    # Initialize the output array
    output_array = np.zeros_like(img_array)

    # Apply Gaussian blur
    for y in range(height):
        for x in range(width):
            x_min = max(0, x - strength)
            x_max = min(width, x + strength + 1)
            y_min = max(0, y - strength)
            y_max = min(height, y + strength + 1)

            surrounding_box = img_array[y_min:y_max, x_min:x_max]

            kernel_y_min = max(0, strength - y)
            kernel_y_max = kernel_size - max(0, strength + y + 1 - height)
            kernel_x_min = max(0, strength - x)
            kernel_x_max = kernel_size - max(0, strength + x + 1 - width)

            relevant_kernel = kernel[kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]

            for c in range(3):
                output_array[y, x, c] = np.sum(surrounding_box[:, :, c] * relevant_kernel) / np.sum(relevant_kernel)

    return Image.fromarray(output_array.astype(np.uint8))


input_image = Image.open('images/butterfly.jpg')
# output_image = linear_blur(input_image, 0.1)
output_image = gaussian_blur(input_image, 0.4, 10)
output_image.save('output.jpg')
