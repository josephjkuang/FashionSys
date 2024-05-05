import json
import logging
import numpy as np
import os
import base64
import time
from PIL import Image
from io import BytesIO

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
neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')

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
    images, descriptions = knn(client_embeddings)
    return images, descriptions

# Server only
def knn(embedding):
    # Finish inference
    embedding = np.array(embedding)
    server_embeddings = ServerModel.call(embedding)

    # Get nearest neighbor and boards
    distance, indices = neighbors.kneighbors([server_embeddings])
    board_ids, matched_ids = display_items(distance, indices)

    # Encode the information for transfer
    descriptions, images = [], []
    # Information to be forwarded to client
    for i, board_id in enumerate(board_ids):
        description, buffer = aggregate_board(board_id, matched_ids[i])
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        descriptions.append(description)
        images.append(encoded_image)

    return images, descriptions

# Helper to get boards that match item
def get_boards(distance, indices):
    boards, matched_ids = [], []

    # Display similar items
    for file_idx in indices[0]:
        filename = filenames[file_idx]

        # Get boards for associated images
        img_path = filenames[file_idx][:-4]
        img_board = item_to_outfits.get(img_path, [])
        boards.extend(img_board)
        matched_ids.extend([filename[:-4] for i in range(len(img_board))])

    return boards, matched_ids

# Combine board into one image and get descriptions
def aggregate_board(board_id, matched_item_id):
    description = ""
    concatenated_image = np.zeros((256, 0, 3), dtype=np.uint8)

    # Loop through items and collect descriptions + images
    for item in outfit_map[board_id]['items']:
        item_id = item["item_id"]

        # Add description and semantic category
        description += items_map[item_id]["url_name"]
        description += " (" + items_map[item_id]['semantic_category'] + "), "

        # Get the image
        image = Image.open(images_path + item_id + ".jpg")

        # Resize
        image = np.array(image.resize((256, 256), Image.LANCZOS))
        image_array = np.array(image)

        if item_id == matched_item_id:
            continue
        
        concatenated_image = np.concatenate((concatenated_image, image_array), axis=1)


    # Convert NumPy array to a Pillow Image
    img = Image.fromarray(concatenated_image)

    # Save the image to a buffer
    buffer = BytesIO()
    img.save(buffer, format="PNG")  # You can use different formats as needed
    buffer.seek(0)

    return description[:-2], buffer

def display_items(distance, indices):
    # html_string, boards, matched_ids = "", [], []
    boards, matched_ids = [], []

    # Display similar items
    for file_idx in indices[0]:
        # Load in images
        filename = filenames[file_idx]
        # img_tag = f"<img src='{images_path + filename}' style='width:{100}px;height:{100}px;margin:0;padding:0;'>"
        # html_string += img_tag

        # Get boards for associated images
        img_path = filenames[file_idx][:-4]
        img_board = item_to_outfits.get(img_path, [])
        boards.extend(img_board)
        matched_ids.extend([filename[:-4] for i in range(len(img_board))])

    return boards, matched_ids