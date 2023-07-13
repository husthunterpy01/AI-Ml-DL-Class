import cv2

# Read the image
image = cv2.imread('Berlin.jpg')

# Create a copy of the image
flipped_image = image.copy()

# Get the shape of the image
height, width, channels = image.shape

# Flip the image horizontally
for i in range(height):
    for j in range(width):
        flipped_image[i, j] = image[height - i -1,j]

# Save the flipped image
cv2.imwrite('Berlin_vertical2.jpg', flipped_image)