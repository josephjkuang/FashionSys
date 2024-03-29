import requests
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import time

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = Sequential([model, GlobalMaxPooling2D()])

start_time = time.time()
img_path = './samples/shoes.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

embedding = model.predict(img_array)
flatten_embedding = embedding.flatten()
result_normlized = flatten_embedding / norm(flatten_embedding)  # Normalizing

response = requests.post(
    'http://localhost:8000/recommendation_only/',
    json=result_normlized.tolist()  # Ensure this matches the expected format
)

if response.status_code == 200:
    end_time = time.time() 
    duration = end_time - start_time 
    print(f"Time taken for reading image and calling api: {duration} seconds")

    print("Recommendations:", response.json())
else:
    print("Error:", response.status_code, response.text)