import base64
import numpy as np
import requests
import time

from transformers import AutoTokenizer, TFDistilBertForMultipleChoice
from utils.ClientResNet import ClientResNet
import tensorflow as tf


# Instantiate Model
ClientModel = ClientResNet()
start_time = time.time()

# Instantiate LLM
prompt = "Womens Clothing to wear during Hot, sunny. Not like A fashion look from January 2015 featuring preppy shirts, beige jeans and leather oxfords. Browse and shop related looks"

# Create LLM
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = TFDistilBertForMultipleChoice.from_pretrained("distilbert-base-cased")
print("LLM Initialized")

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

    # Decode the images
    images = []
    for index, base64_image in enumerate(encoded_images):
        image_data = base64.b64decode(base64_image)
        images.append(image_data)

    # Format to add prompt to descriptions
    tokenizer_input = []
    for description in descriptions:
        tokenizer_input.append([prompt, description])
    encoding = tokenizer(tokenizer_input, return_tensors="tf", padding=True)

    # Make predictions
    outputs = model(**{k: tf.expand_dims(v, axis=0) for k, v in encoding.items()}) # batch size is 1
    logits = outputs.logits
    predicted_label = np.argmax(logits, axis=1)
    logits_list = logits.numpy().tolist()

    # Sort the elements
    paired_list = list(zip(descriptions, logits_list[0]))
    sorted_pairs = sorted(enumerate(paired_list), key=lambda x: x[1][1], reverse=True)

    # Save the images and print descriptions
    for idx, pair in enumerate(sorted_pairs):
        old_idx, description = pair[0], pair[1][0]
        filename = f"image{idx + 1}.png"

        # Write the image data to a file
        with open(filename, 'wb') as image_file:
            image_file.write(images[old_idx])
        print(f"Saved {filename} with description: {descriptions[idx]}")

else:
    print("Error:", response.status_code, response.text)

# ----------------------- probably need to repeat the above code again for accuracte latency ---------------------------
