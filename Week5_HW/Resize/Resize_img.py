import cv2
import numpy as np
def resize_image(image, new_width, new_height):
    # Get the dimensions of the original image
    height, width, channels = image.shape

    # Compute scaling factors for width and height
    width_scale = new_width / width
    height_scale = new_height / height

    # Create a new blank image with the desired dimensions
    resized_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    # Resize the image using bilinear interpolation
    for y in range(new_height):
        for x in range(new_width):
            # Find the corresponding location in the original image
            src_x = (x + 0.5) / width_scale - 0.5
            src_y = (y + 0.5) / height_scale - 0.5

            # Compute the four nearest pixels in the original image
            src_x0 = int(src_x)
            src_x1 = min(src_x0 + 1, width - 1)
            src_y0 = int(src_y)
            src_y1 = min(src_y0 + 1, height - 1)

            # Compute the fractional parts of src_x and src_y
            src_x_frac = src_x - src_x0
            src_y_frac = src_y - src_y0

            # Perform bilinear interpolation for each channel
            for c in range(channels):
                top = image[src_y0, src_x0, c] * (1 - src_x_frac) + image[src_y0, src_x1, c] * src_x_frac
                bottom = image[src_y1, src_x0, c] * (1 - src_x_frac) + image[src_y1, src_x1, c] * src_x_frac
                resized_image[y, x, c] = top * (1 - src_y_frac) + bottom * src_y_frac

    return resized_image


# Read the image
image = cv2.imread('Berlin.jpg')

# Resize the image
resized_image = resize_image(image, 800, 450)

# Save the resized image
cv2.imwrite('Berlin_resized.jpg', resized_image)
