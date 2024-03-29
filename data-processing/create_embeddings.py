import numpy as np
import os

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([model, GlobalMaxPooling2D()])

folder_path = "../polyvore_outfits/images/"
batch = 1
batch_file_path = "batch_files/batch_{}.txt".format(batch)
print("Batch", batch)

file = open(batch_file_path, 'r')
filenames = file.read().splitlines()
jpg_files = [os.path.join(folder_path, filename) for filename in filenames]
embeddings = []
batch_size = 64

# Process images in batches
for i in range(0, len(jpg_files), batch_size):
    print("Batch", i // batch_size, "of", len(jpg_files) // batch_size)
    batch_files = jpg_files[i:i+batch_size]
    batch_images = []

    # Go within the batch to get image
    for path in batch_files:
        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        batch_images.append(img_array)

    # Make prediction and normalize
    batch_images = preprocess_input(np.array(batch_images))
    batch_embeddings = model.predict(batch_images)
    batch_embeddings /= np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
    embeddings.extend(batch_embeddings)

# Convert embeddings list to numpy array
embeddings = np.array(embeddings)
np.save('embeddings.npy', embeddings)