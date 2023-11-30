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
    # make each pixel the average of the pixels around it

    # cache the recent most strength * strength pixels
    cache = OrderedDict({(-i, -i): (0, 0, 0) for i in range(strength * strength)})

    for y in range(image.height):
        print(y)
        for x in range(image.width):
            surrounding_pixels = []
            for i in range(-strength, strength + 1):
                for j in range(-strength, strength + 1):
                    try:
                        if (x + i, y + j) not in cache:
                            cache[(x + i, y + j)] = image.getpixel((x + i, y + j))
                        surrounding_pixels.append(cache[(x + i, y + j)])
                        cache.popitem(last=False)
                    except IndexError:
                        pass
            output_pixel = tuple(map(lambda pixel_list: sum(pixel_list) // len(pixel_list), zip(*surrounding_pixels)))
            image.putpixel((x, y), output_pixel)
    return image



input_image = Image.open('images/smallEarth.jpg')
output_image = linear_blur(input_image, 10)
output_image.save('output.jpg')
