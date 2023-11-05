# Retinal_Vessel_Segmentation

Retinal illnesses such as diabetic retinopathy (DR) are the main causes of vision loss. In the early recognition of eye diseases, the segmentation of blood vessels in retina images plays an important role. However, due to the complex construction of the blood vessels and their different thicknesses, segmenting the retina image is a challenging task. This repository contains our approach to retinal vessel segmentation with deep learning, using Kaggle DRIVE 2004 dataset.

We tried out several architectures and the most efficent architecture contained in Model4.ipynb. Here used a U-Net model from the segmentation library which is a convolutional neural network designed for semantic image segmentation tasks with a VGG19 backbone and a sigmoid activation function for retinal vessel segmentation. You can find the results of the evaluation metrics for each model in the 
