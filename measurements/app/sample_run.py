import json
import logging
import numpy as np
import os
import base64

from sklearn.neighbors import NearestNeighbors
from utils.ServerResNet import ServerResnet
from utils.ClientResNet import ClientResNet

# Data paths
data_dir = "../polyvore_outfits/"
images_path = data_dir + "images/"

# Load in embeddings
embeddings_path = data_dir + "embeddings.npy"
embeddings = np.load(embeddings_path)

# Load in filenames
filenames_path = data_dir + "filenames.txt"
filenames_file = open(filenames_path, 'r')
filenames = [line.strip() for line in filenames_file.readlines()]

# Error checking for embeddings + filenames
if len(embeddings) != len(filenames):
    print("STOP. The lengths of embeddings and filenames don't match")
    print(len(embeddings), len(filenames))

# Load in outfits metadata
outfits_metadata_path = data_dir + "outfits_metadata.json"
outfits_file = open(outfits_metadata_path, 'r')
outfit_map = json.load(outfits_file)

# Load in item (clothing) metadata
item_metadata_path = data_dir + "item_metadata.json"
items_file = open(item_metadata_path, 'r')
items_map = json.load(items_file)

# Inverted index from image to boards they are part of
item_to_outfits = {}
for outfit_id, metadata in outfit_map.items():
    for item in metadata['items']:
        item_id = item['item_id']
        item_to_outfits[item_id] = item_to_outfits.get(item_id, []) + [outfit_id]

# Create nearest neighbor search for embeddings
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')

# Seperated Models ResNet
ClientModel = ClientResNet()
ServerModel = ServerResnet()

in_path = "../polyvore_outfits/images/"

# Loading in initial model for 
def load_model():
    # Fit the nearest neighbors
    neighbors.fit(embeddings)
    print("Loaded in embeddings for nearest neighbors")

# Full prediction
def resnet_and_knn(img):
    client_embeddings = ClientModel.predict(img, True, False)
    server_embeddings = ServerModel.call(client_embeddings)
    distance, indices = neighbors.kneighbors([server_embeddings])

    result = []

    # Loop through files
    for file_idx in indices[0][1:6]:
        in_img_path = in_path + filenames[file_idx]
        result.append(in_img_path)

        # This result only sends original image path
        # More code would need to be moved for the whole board + description - Joseph

    return result

# Server only
def knn(embedding):
    embedding = np.array(embedding)
    print(embedding.shape)
    # client_embeddings = embedding.reshape((1, 28, 28, 512))
    server_embeddings = ServerModel.call(embedding)
    distance, indices = neighbors.kneighbors([server_embeddings])

    images = []
    descriptions = [] 
    for file_idx in indices[0][1:6]:
        in_img_path = in_path + filenames[file_idx]
        
        # This result only sends original image path
        # More code would need to be moved for the whole board + description - Joseph


        # example of sending images
        # TODO: change this to returning style board instead. right now it's just returning similar items
        with open(in_img_path, "rb") as image_file:
            # encode the image so it can be send with JSON
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_image)
    
    # TODO: get descriptions
    descriptions = ["test"] * len(images)

    return images, descriptions