import numpy as np
from matplotlib import pyplot as plt

file_path = "./full_numpy_bitmap_bird.npy"
images = np.load(file_path).astype(np.float32)
print(images.shape)
train_images = images[:-10]
test_images = images[-10:]

# Calculate an average image of the training
images_array = np.mean(train_images,axis=0)
avg_image = np.reshape(images_array, (28,28))

plt.imshow(avg_image)
plt.show()
# Test_image
index = 2
test_image = test_images[index]

test_image = np.reshape(test_image, (28, 28))

# Dot product score
score = np.dot(avg_image, test_image)
print(score)