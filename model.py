import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define U-Net model
def unet_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape) # (512, 512, 3): 512x512 RGB images

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs) # (512, 512, 64): 64 filters, 3x3 kernel size
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) # (256, 256, 64): 2x2 max pooling

    # Decoder
    up1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(pool1) # (512, 512, 64): 2x2 upsampling
    concat1 = tf.keras.layers.concatenate([conv1, up1], axis=3) # (512, 512, 128): concatenate along the channels axis
    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(concat1)    # (512, 512, 1): 1x1 convolution

    model = tf.keras.models.Model(inputs, output)   # Create the model
    return model

# Load and preprocess your dataset (images and masks)
def load_and_preprocess_data(data_dir):
    images = [] # List of images
    masks = []  # List of masks corresponding to the images

    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):  # Assuming images are in PNG format
            img_path = os.path.join(data_dir, filename) # Assuming images are in the root directory
            mask_path = os.path.join(data_dir, 'mask', filename.replace('.png', '_mask.png'))  # Assuming mask filenames are similar with "_manual1" suffix

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            images.append(img)
            masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0  # Normalize pixel values
    masks = np.array(masks, dtype=np.float32) / 255.0  # Normalize pixel values

    return images, masks

# Split the dataset into training and validation sets
def split_data(images, masks, validation_split=0.2, random_state=42):
    return train_test_split(images, masks, test_size=validation_split, random_state=random_state)

# Define model parameters and compile the model
input_shape = (512, 512, 3) # (height, width, channels)
batch_size = 8  # Number of images used in each optimization step
num_epochs = 50 # Number of times the entire training dataset is used to learn the weights of the network
learning_rate = 0.0001  # How fast the model learns (model parameters are updated)

model = unet_model(input_shape) # Build the model
model.compile(optimizer=Adam(learning_rate), loss=binary_crossentropy, metrics=['accuracy'])    # Compile the model

# Load and preprocess your dataset
data_dir = 'resized/training'
images, masks = load_and_preprocess_data(data_dir)

# Split the dataset into training and validation sets
train_images, val_images, train_masks, val_masks = split_data(images, masks)

# Train the model
model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=num_epochs, batch_size=batch_size)

# Evaluate the model (you can use a test set for this)
test_images, test_masks = load_and_preprocess_data('resized/test')
test_loss, test_accuracy = model.evaluate(test_images, test_masks)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
