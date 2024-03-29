import pickle
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

neighbors = NearestNeighbors(n_neighbors = 6, algorithm='brute', metric='euclidean')

# resnet50
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

in_path = "/home/xinshuo3/images/"
out_path = "/home/xinshuo3/outputs/"

out_list = []

def load_model():
    features_list = pickle.load(open("./embeddings.pkl", "rb"))
    img_files_list = pickle.load(open("./filenames.pkl", "rb"))

    print("finished loading pkl files")


    for img_name in img_files_list:
        end = img_name.split("/")[-1]
        out_list.append(end)

    print("resnet loaded")

    # result_normlized = features_list[0]
    neighbors.fit(features_list)
    print("finished fitting model")

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