from PIL import Image


def greyscale(image):
    # go through each pixel and change it to black and white
    for x in range(image.width):
        for y in range(image.height):
            r, g, b = image.getpixel((x, y))

            # calculate the average value of rgb
            avg = (r + g + b) // 3

            # set the pixel to the new color 'avg'
            image.putpixel((x, y), (avg, avg, avg))

    return image


input_image = Image.open('images/butterfly.jpg')
output_image = greyscale(input_image)
output_image.save('output.jpg')

