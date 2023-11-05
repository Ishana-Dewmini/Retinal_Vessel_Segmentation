import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion
import matplotlib.pyplot as plt
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting names """
        name = os.path.splitext(os.path.basename(x))[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        # Check the bit depth of the image
        bit_depth = x.dtype

        
        # Normalize based on the bit depth
        if bit_depth == np.uint8:
            # Normalize 8-bit image to [0, 1]
            print("8 bit")
            x = x.astype(np.float32) / 255.0
        elif bit_depth == np.uint16:
            # Normalize 16-bit image to [0, 1]
            print("16 bit")
            x = x.astype(np.float32) / 65535.0
        else:
            # Handle other bit depths as needed
            print("Unsupported bit depth")

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(X) == 1:
                tmp_image_name = f"{name}.{'tif'}"
                tmp_mask_name = f"{name}.{'gif'}"
            else:
                tmp_image_name = f"{name}_{index}.{'tif'}"
                tmp_mask_name = f"{name}_{index}.{'gif'}"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            imageio.mimsave(mask_path, [m], format='GIF')  # Save the mask as GIF with a list

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "DRIVE"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Creating directories """
    create_dir("new_data/train/image")
    create_dir("new_data/train/mask")
    create_dir("new_data/test/image")
    create_dir("new_data/test/mask")

    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)


# Display first image and augmented images
fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(20, 4), sharey=True, sharex=True)
ax[0].imshow(cv2.cvtColor(cv2.imread("new_data/train/image/21_training_0.tif"), cv2.COLOR_BGR2RGB))
ax[0].set_title("Original image")
ax[1].imshow(cv2.cvtColor(cv2.imread("new_data/train/image/21_training_1.tif"), cv2.COLOR_BGR2RGB))
ax[1].set_title("Horizontal Flip")
ax[2].imshow(cv2.cvtColor(cv2.imread("new_data/train/image/21_training_2.tif"), cv2.COLOR_BGR2RGB))
ax[2].set_title("Vertical Flip")
ax[3].imshow(cv2.cvtColor(cv2.imread("new_data/train/image/21_training_3.tif"), cv2.COLOR_BGR2RGB))
ax[3].set_title("Elastic Transform")
ax[4].imshow(cv2.cvtColor(cv2.imread("new_data/train/image/21_training_4.tif"), cv2.COLOR_BGR2RGB))
ax[4].set_title("Grid Distortion")
ax[5].imshow(cv2.cvtColor(cv2.imread("new_data/train/image/21_training_5.tif"), cv2.COLOR_BGR2RGB))
ax[5].set_title("Optical Distortion")
plt.show()



