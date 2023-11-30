from PIL import Image
from collections import OrderedDict
import numpy as np


def greyscale(image):
    # go through each pixel and change it to black and white
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


input_image = Image.open('images/butterfly.jpg')
output_image = linear_blur(input_image, 0.1)
output_image.save('output.jpg')
