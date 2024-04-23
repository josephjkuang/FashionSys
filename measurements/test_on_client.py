import requests
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import time

# TODO: replace this with loading client-side model, embeddings, etc.
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = Sequential([model, GlobalMaxPooling2D()])

start_time = time.time()

# read a test image
img_path = './samples/shoes-2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# TODO: replace this with inferencing on client
embedding = model.predict(img_array)
flatten_embedding = embedding.flatten()
# result_normlizded is the embedding
result_normlized = flatten_embedding / norm(flatten_embedding)  # Normalizing

# pass embedding to server. the ip address here is VM1's 
response = requests.post(
    'http://172.22.151.173:8000/recommendation_only/',
    json=result_normlized.tolist() 
)

if response.status_code == 200:
    end_time = time.time() 
    duration = end_time - start_time 
    # print(f"Time taken for reading image and calling api: {duration} seconds")

    print("Recommendations:", response.json())
else:
    print("Error:", response.status_code, response.text)


# ------------- code starting from this point is exactly same as the part above. --------------
# I had to do the whole process twice because in the first time the latency measurement would inplicilty include the time it takes to load model into DRAM
start_time = time.time()
print(start_time)
img_path = './samples/shoes.jpg'
img = image.load_img(img_path, target_size=(224, 224))
print("finished loading image", time.time())
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
print("finished preprocessing image", time.time())

embedding = model.predict(img_array)
flatten_embedding = embedding.flatten()
result_normlized = flatten_embedding / norm(flatten_embedding)  # Normalizing

print(time.time(), "finished processing")

response = requests.post(
    'http://172.22.151.173:8000/recommendation_only/',
    json=result_normlized.tolist()  # Ensure this matches the expected format
)

if response.status_code == 200:
    print(time.time())
    end_time = time.time() 
    duration = end_time - start_time 
    print(f"Time taken for reading image and calling api: {duration} seconds")

    print("Recommendations:", response.json())
else:
    print("Error:", response.status_code, response.text)