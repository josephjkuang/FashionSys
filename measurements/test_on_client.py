import os
import logging
import base64

# Suppress cuda errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('urllib3').setLevel(logging.ERROR)

from utils.ClientResNet import ClientResNet

import numpy as np
import requests
import tensorflow as tf
import time

# Instantiate Model
ClientModel = ClientResNet()
start_time = time.time()

# Read in embedding and perform inference
img_path = './samples/shoes-2.jpg'
embeddings = ClientModel.predict(img_path, True, False)
embeddings_json = embeddings.numpy().tolist()
print("Got to embeddings", embeddings.shape)

print("Sending embeddings:", time.time())
# Pass embedding to server. the ip address here is VM1's 
response = requests.post(
    'http://172.22.151.173:8000/recommendation_only/',
    json=embeddings_json
)

# Check response
if response.status_code == 200:
    end_time = time.time() 
    duration = end_time - start_time 
    print(f"Time taken for reading image and calling api: {duration} seconds")

    # parse response
    json_response = response.json()
    descriptions = json_response['descriptions']
    encoded_images = json_response['images']
    for index, base64_image in enumerate(encoded_images):
        image_data = base64.b64decode(base64_image)
        filename = f"image{index + 1}.png"
        # Write the image data to a file
        with open(filename, 'wb') as image_file:
            image_file.write(image_data)
        print(f"Saved {filename} with description: {descriptions[index]}")
else:
    print("Error:", response.status_code, response.text)

# ----------------------- same as above ---------------------------

