
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard, EarlyStopping

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)




H = 512
W = 512

def load_data(path):
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    """ Dataset """
    dataset_path = path
    train_path = os.path.join(dataset_path, "train")
    x = sorted(glob(os.path.join(train_path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(train_path, "mask", "*.jpg")))
    train_x, train_y = shuffle(x, y, random_state=42)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)
    # convert to np.array
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)
    return train_x, train_y, valid_x, valid_y


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)  
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset


if __name__ == "__main__":
    # load your data
    path = 'new_data'
    x_train, y_train, x_val, y_val = load_data(path)
    print(f"Train: {len(x_train)} - {len(y_train)}")
    print(f"Valid: {len(x_val)} - {len(y_val)}")

    """ Directory to save files """
    if not os.path.exists("files"):
        os.makedirs("files")

    """ Hyperparameters """
    batch_size = 8
    lr = 1e-5
    num_epochs = 100
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    """ Dataset """

    train_dataset = tf_dataset(x_train, y_train, batch_size=batch_size)
    valid_dataset = tf_dataset(x_val, y_val, batch_size=batch_size)

    train_steps = len(x_train)//batch_size
    valid_setps = len(x_val)//batch_size

    if len(x_train) % batch_size != 0:
        train_steps += 1
    if len(x_val) % batch_size != 0:
        valid_setps += 1

    # define model
    # Load the pre-trained model
    BACKBONE = 'vgg16'
    model = Unet(backbone_name=BACKBONE, input_shape=(H, W, 3), 
                classes=1, activation='sigmoid',  
                encoder_weights='imagenet', encoder_freeze=True, 
                encoder_features='default',  
                decoder_filters=(256, 128, 64, 32, 16), 
                decoder_use_batchnorm=True, )

   

    '''# Add a new head to the model
    new_head = tf.keras.Sequential([
        tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid'),
    ])
    # Add the new head to the output of the model
    model.outputs = [new_head(model.outputs[0])]'''

    # Compile the model
    model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
    model.summary()


    # Define callbacks
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=False)
    ]

    # Load the previously trained weights
    if os.path.exists(model_path):
        model.load_weights(model_path)

    # Train the model
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_setps,
        callbacks=callbacks
    )


   