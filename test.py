from tensorflow import keras
from keras.models import load_model
import cv2
import numpy as np
import os

# Load and preprocess your dataset (images and masks)
def load_and_preprocess_data(data_dir):
    images = [] # List of images
    masks = []  # List of masks corresponding to the images
    data_dir_images = os.path.join(data_dir, 'images')
    data_dir_masks = os.path.join(data_dir, '1st_manual')
    for filename in os.listdir(data_dir_images):
        if filename.endswith('.png'):  # Assuming images are in PNG format
            img_path = os.path.join(data_dir_images, filename) 
            mask_path = os.path.join(data_dir_masks, filename.replace('test.png', 'manual1.png'))  # Get corresponding mask filename

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            images.append(img)
            masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0  # Normalize pixel values
    masks = np.array(masks, dtype=np.float32) / 255.0  # Normalize pixel values

    return images, masks

# Load the trained model from the file
loaded_model = load_model('unet_model.h5')


# Evaluate the model (you can use a test set for this)
test_images, test_masks = load_and_preprocess_data('E:\\acadamic\\sem 5\\EN3160_Image Processing and Machine Vision\\project\\Project\\Data\\test')
print(f'Test images shape: {test_images.shape}, Test masks shape: {test_masks.shape}')
test_loss, test_accuracy = loaded_model.evaluate(test_images, test_masks)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')