import cv2
import numpy as np
# Read the image
image = cv2.imread('Berlin.jpg')

# Get the shape of the image
height, width, channels = image.shape

# Create a new image with rotated dimensions
rotated_image = [[0] * height for _ in range(width)]

# Rotate the image by 90 degrees
for i in range(height):
    for j in range(width):
        rotated_image[width - j -1][i] = image[i, j]

# Convert the rotated image back to NumPy array
rotated_image = np.array(rotated_image, dtype=np.uint8)

# Save the rotated image
cv2.imwrite('re_rotated_image.jpg', rotated_image)
