import os
import logging

# Set TensorFlow logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '0' to display all logs, '1' to suppress INFO logs, '2' to suppress INFO and WARNING logs
# You can set '3' to suppress ERROR logs as well, but it's generally not recommended as you might miss important error messages

# Suppress other libraries' loggers if needed
logging.getLogger('urllib3').setLevel(logging.ERROR)  # Suppress urllib3 logs to suppress CUDA-related warnings

import utils.color_classify
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image

# Half of the Inference Model Stored on Client Side
class ClientResNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(ClientResNet, self).__init__(*args, **kwargs)
        
        # Load ResNet50 pre-trained model without top (fully connected) layers
        resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_base.trainable = False
        
        # Get the output of the first three convolutional layers
        middle_layer = resnet_base.get_layer('conv3_block4_out')
        self.seq0 = Model(inputs=resnet_base.input, outputs=middle_layer.output)
        
        resnet_base = None

        # Get cluster centers for cluster-based differential privacy
        self.color_centers = {}
        for i in range(0, 30):
            self.color_centers[i] = np.load(f"../polyvore_outfits/noise_embeddings/color_emb_{i}.npy")

        self.gmm_model = joblib.load('../polyvore_outfits/gmm_color_model_0_24_2.pkl')

    # Convert image into right dimensions
    def preprocess_image(self, img):
        img_array = kimage.img_to_array(img)
        expand_img = np.expand_dims(img_array, axis=0)
        return preprocess_input(expand_img)
    
    # Add laplacian noise to embedding
    def add_laplacian_noise(self, emb, noise_scale):
        laplace_noise = np.random.laplace(0.0, noise_scale, size=emb.shape)
        emb += laplace_noise
        return emb
    
    # Add laplacian noise to embedding that is sign-controlled
    def add_cluster_based_noise(self, emb, item_rgb, noise_scale):
        # Get probabilities that it belongs to a certain cluster
        cluster_idx = self.gmm_model.predict(np.array(item_rgb).reshape(1, -1))[0]

        # Add controlled-random noise
        laplacian_noise = np.random.laplace(0.0, noise_scale, size=emb.shape)

        # Calculate distances after adding/subtracting the Laplacian noise
        distance_after_add = np.abs(emb + laplacian_noise - self.color_centers[cluster_idx])
        distance_after_sub = np.abs(emb - laplacian_noise - self.color_centers[cluster_idx])

        # Compare distances to determine direction of noise addition
        add_mask = distance_after_add > distance_after_sub
        subtract_mask = ~add_mask

        # Apply noise according to the determined direction
        laplacian_noise *= add_mask.astype(float) - subtract_mask.astype(float)

        return emb + laplacian_noise
        
    # Forward pass of the client model
    def predict(self, image_path, laplace_noise=False, cluster_noise=False, noise_scale=0.35):
        # Open and process the image
        img = Image.open(image_path)
        img = img.resize((224, 224)).convert("RGB")

        preprocessed_img = self.preprocess_image(img)
        img.close()

        # Feed through ResNet
        emb = self.seq0(preprocessed_img)

        # Apply noise
        if laplace_noise:
            emb = self.add_laplacian_noise(emb, noise_scale)
        elif cluster_noise:
            item_rgb = color_classify.get_color(image_path)
            emb = self.add_cluster_based_noise(emb, item_rgb, noise_scale)

        return emb
    
    # Needed for subclass
    def call(self, inputs):
        return self.seq0(inputs)

# Instantiate Model
ClientModel = ClientResNet()
print("Model Initialized")