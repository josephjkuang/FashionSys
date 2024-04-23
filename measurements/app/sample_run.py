import os
import logging

# Suppress cuda errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('urllib3').setLevel(logging.ERROR)

from sklearn.neighbors import NearestNeighbors
# import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import numpy as np
import json

from utils.ServerResNet import ServerResnet

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

# Server ResNet
server_model = ServerResnet()

in_path = "/home/xinshuo3/images/"
out_path = "/home/xinshuo3/outputs/"

out_list = []

def load_model():
    # Fit the nearest neighbors
    neighbors.fit(embeddings)
    print("Loaded in embeddings for nearest neighbors")

def resnet_and_knn(img):

    # img = image.load_img('./samples/shoes.jpg',target_size=(224,224))
    # img_array = image.img_to_array(img)
    img_array = np.array(img)
    expand_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)

    print("finished processing input images")

    distance, indices = neighbors.kneighbors([result_normlized])

    result = []

    for file in indices[0][1:6]:
        # print(out_list[file])
        out_img_path = out_path + out_list[file]
        in_img_path = in_path + out_list[file]
        result.append(in_img_path)
        # print(in_img_path)
        # img = mpimg.imread(in_img_path)
        # plt.imshow(img)  # Plot the image
        # plt.savefig(out_img_path)  # Save the image to out_img_path
        # plt.close() 

    return result


def knn(embedding):
    distance, indices = neighbors.kneighbors([embedding])

    result = []
    for file in indices[0][1:6]:  # Assuming you want to skip the first result as it's likely the input image
        out_img_path = out_path + out_list[file]
        in_img_path = in_path + out_list[file]
        result.append(in_img_path)
    return result