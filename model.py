import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# Define model parameters and compile the model
input_shape = (512, 512, 3) # (height, width, channels)
batch_size = 16  # Number of images used in each optimization step
num_epochs = 25 # Number of times the entire training dataset is used to learn the weights of the network
learning_rate = 0.0001  # How fast the model learns (model parameters are updated)

# Define U-Net model
'''def unet_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape) # (512, 512, 3): 512x512 RGB images

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs) # (512, 512, 64): 64 filters, 3x3 kernel size
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) # (256, 256, 64): 2x2 max pooling

    # Decoder
    up1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(pool1) # (512, 512, 64): 2x2 upsampling
    concat1 = tf.keras.layers.concatenate([conv1, up1], axis=3) # (512, 512, 128): concatenate along the channels axis
    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(concat1)    # (512, 512, 1): 1x1 convolution

    model = tf.keras.models.Model(inputs, output)   # Create the model
    return model'''

def unet_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    #s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.models.Model(inputs, outputs)   # Create the model
    return model


# Load and preprocess your dataset (images and masks)
def load_and_preprocess_data(data_dir):
    images = [] # List of images
    masks = []  # List of masks corresponding to the images
    data_dir_images = os.path.join(data_dir, 'images')
    data_dir_masks = os.path.join(data_dir, '1st_manual')
    for filename in os.listdir(data_dir_images):
        if filename.endswith('.png'):  # Assuming images are in PNG format
            img_path = os.path.join(data_dir_images, filename) 
            mask_path = os.path.join(data_dir_masks, filename.replace('training.png', 'manual1.png'))  # Get corresponding mask filename

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            images.append(img)
            masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0  # Normalize pixel values
    masks = np.array(masks, dtype=np.float32) / 255.0  # Normalize pixel values

    return images, masks

# Split the dataset into training and validation sets
def split_data(images, masks, validation_split=0.2, random_state=42): # 80% training, 20% validation
    return train_test_split(images, masks, test_size=validation_split, random_state=random_state)



Model = unet_model(input_shape) # Build the model

Model.compile(optimizer=Adam(learning_rate), loss=binary_crossentropy, metrics=['accuracy'])    # Compile the model
print(Model.summary())  # Print model summary
print('Model compiled successfully!')
 
# Modelcheckpoint callback
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_vessel.h5', verbose=1, save_best_only=True)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
                tf.keras.callbacks.TensorBoard(log_dir='logs')]

# Load and preprocess your dataset
data_dir = 'E:\\acadamic\\sem 5\\EN3160_Image Processing and Machine Vision\\project\\Project\\Data\\training'
images, masks = load_and_preprocess_data(data_dir)
print(f'Images shape: {images.shape}, Masks shape: {masks.shape}')

# Split the dataset into training and validation sets
train_images, val_images, train_masks, val_masks = split_data(images, masks)
print(f'Train images shape: {train_images.shape}, Train masks shape: {train_masks.shape}')
print(f'Validation images shape: {val_images.shape}, Validation masks shape: {val_masks.shape}')

# Train the model
history = Model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=num_epochs, batch_size=batch_size, callbacks=callbacks)
print('Model trained successfully!')

# save the model
Model.save('unet_model2.h5')
print ('Model saved successfully!')



# Compute additional validation metrics: precision, recall, and F1-score
val_predictions = Model.predict(val_images)
# Convert validation masks to binary
threshold = 0.8
val_masks_binary = (val_masks > threshold).astype(np.uint8)

# Convert validation predictions to binary
val_predictions_binary = (val_predictions > threshold).astype(np.uint8)

# Compute additional validation metrics: precision, recall, and F1-score
precision = precision_score(val_masks_binary.flatten(), val_predictions_binary.flatten())
recall = recall_score(val_masks_binary.flatten(), val_predictions_binary.flatten())
f1 = f1_score(val_masks_binary.flatten(), val_predictions_binary.flatten())

# Get validation accuracy using Model.evaluate()
_, accuracy = Model.evaluate(val_images, val_masks)

# Save the model summary, accuracy, precision, recall, and F1-score to a .txt file
with open('model_summary.txt', 'w') as summary_file:
    Model.summary(print_fn=lambda x: summary_file.write(x + '\n'))

with open('validation_metrics.txt', 'w') as metrics_file:
    metrics_file.write(f'Validation Accuracy: {accuracy}\n')
    metrics_file.write(f'Validation Precision: {precision}\n')
    metrics_file.write(f'Validation Recall: {recall}\n')
    metrics_file.write(f'Validation F1-Score: {f1}\n')





