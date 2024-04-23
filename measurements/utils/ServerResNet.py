import os
import logging

# Suppress cuda errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('urllib3').setLevel(logging.ERROR)

import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model


# Half of the Inference Model Stored on Server Side
class ServerResnet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(ServerResnet, self).__init__(*args, **kwargs)
        
        # Load ResNet50 pre-trained model without top (fully connected) layers
        resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_base.trainable = False
        
        # Get the output of the last two convolutional layer before Pooling
        middle_layer = resnet_base.get_layer('conv3_block4_out')
        last_conv_layer = resnet_base.get_layer('conv5_block3_out')

        self.seq1 = tf.keras.Sequential([
            Model(inputs=middle_layer.output, outputs=last_conv_layer.output),
            tf.keras.layers.GlobalMaxPooling2D(),
        ])

    # Forward pass of the server model
    def call(self, inputs):
        server_embeddings = self.seq1(inputs)

        # Normalize data
        flatten_result = tf.keras.layers.Flatten()(server_embeddings)
        result_normalized = flatten_result / tf.norm(flatten_result)
        return result_normalized[0]

ServerModel = ServerResnet()
print("Initialized Server ResNet")