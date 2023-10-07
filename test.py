from tensorflow import keras
from keras.models import load_model
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

# display the first image and mask in the test set
fig, ax2 = plt.subplots(1, 2, figsize=(10, 10))
ax2[0].imshow(cv2.cvtColor(test_images[0], cv2.COLOR_BGR2RGB))
ax2[1].imshow(test_masks[0], cmap='gray')
plt.show()

print(f'Test images shape: {test_images.shape}, Test masks shape: {test_masks.shape}')
test_loss, test_accuracy = loaded_model.evaluate(test_images, test_masks)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# print validation matrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Predict on the first 10 test images
predictions = loaded_model.predict(test_images[:10])
print(f'Predictions shape: {predictions.shape}')

# Display the first 10 test images and their corresponding masks and predictions   
fig, ax2 = plt.subplots(10, 3, figsize=(40, 30))
for i in range(0, 10):
    ax2[i, 0].imshow(cv2.cvtColor(test_images[i], cv2.COLOR_BGR2RGB))
    ax2[i, 1].imshow(test_masks[i], cmap='gray')
    ax2[i, 2].imshow(predictions[i], cmap='gray')
plt.show()

# print confusion matrix
y_true = test_masks.flatten()
y_pred = predictions.flatten()
cm = confusion_matrix(y_true, y_pred)
print(cm)

# print classification report
print(classification_report(y_true, y_pred))

# print accuracy
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: %f' % accuracy)

# print precision, recall, f1-score
precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1-score: %f' % fscore)

# print f1-score
f1 = f1_score(y_true, y_pred)
print('F1 score: %f' % f1)

# print precision
precision = precision_score(y_true, y_pred)
print('Precision: %f' % precision)

# print recall
recall = recall_score(y_true, y_pred)
print('Recall: %f' % recall)

  



