import os
os.environ["SM_FRAMEWORK"] = "tf.keras"


import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

# Test images and masks paths
test_images_path = "new_data/test/image"
test_masks_path = "new_data/test/mask"
result_dir = "result"

# Create the result directory if it doesn't exist
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Load the pre-trained model
H = 512
W = 512
BACKBONE = 'vgg16'
model = Unet(
    backbone_name=BACKBONE,  # Define BACKBONE as per your training code
    input_shape=(H, W, 3),
    classes=1,
    activation='sigmoid',
    encoder_weights='imagenet',
    encoder_freeze=True,
    encoder_features='default',
    decoder_filters=(256, 128, 64, 32, 16),
    decoder_use_batchnorm=True,
)

# Load the model weights
model_path = os.path.join("files", "model.h5")
model.load_weights(model_path)

# Initialize lists to store evaluation results
accuracy_scores = []
f1_scores = []
jaccard_scores = []
precision_scores = []
recall_scores = []

# List of test image files
test_image_files = sorted(os.listdir(test_images_path))

# Loop through test images
for image_file in tqdm(test_image_files, desc="Testing"):
    # Read the test image
    test_image = cv2.imread(os.path.join(test_images_path, image_file))
    test_image = test_image / 255.0
    test_image = test_image.astype(np.float32)

    # Make predictions
    predicted_mask = model.predict(np.expand_dims(test_image, axis=0))[0]

    # Threshold the predicted mask
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

    # Save the predicted mask as a PNG file
    predicted_mask_filename = os.path.splitext(image_file)[0] + "_predicted.png"
    predicted_mask_path = os.path.join(result_dir, predicted_mask_filename)
    cv2.imwrite(predicted_mask_path, predicted_mask)

    # Read the corresponding ground truth mask
    mask_filename = os.path.splitext(image_file)[0] + ".jpg"
    mask = cv2.imread(os.path.join(test_masks_path, mask_filename), cv2.IMREAD_GRAYSCALE)

    # Compute evaluation metrics
    accuracy = accuracy_score(mask.flatten(), predicted_mask.flatten())
    f1 = f1_score(mask.flatten(), predicted_mask.flatten(), average=None)
    jaccard = jaccard_score(mask.flatten(), predicted_mask.flatten(), average=None)  # Set average=None
    precision = precision_score(mask.flatten(), predicted_mask.flatten(), average=None)
    recall = recall_score(mask.flatten(), predicted_mask.flatten(), average=None)



    # Append evaluation results to the lists
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    jaccard_scores.append(jaccard)
    precision_scores.append(precision)
    recall_scores.append(recall)

# Calculate mean values for evaluation metrics
mean_accuracy = np.mean(accuracy_scores)
mean_f1 = np.mean(f1_scores)
mean_jaccard = np.mean(jaccard_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)

# Save the evaluation results to a CSV file
evaluation_results = {
    "Image File": test_image_files,
    "Accuracy": accuracy_scores,
    "F1 Score": f1_scores,
    "Jaccard Score": jaccard_scores,
    "Precision": precision_scores,
    "Recall": recall_scores,
}
df = pd.DataFrame(evaluation_results)
evaluation_csv_path = "evaluation_results.csv"
df.to_csv(evaluation_csv_path, index=False)

print("Evaluation results saved to", evaluation_csv_path)
print("Mean Accuracy:", mean_accuracy)
print("Mean F1 Score:", mean_f1)
print("Mean Jaccard Score:", mean_jaccard)
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
