import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Flatten, MaxPool2D, Dense
import os
from const import IMG_WIDTH, IMG_HEIGHT, csv_path, image_dir, saved_model_path


# Constants
BATCH_SIZE = 16
MAX_LENGTH = 8  # число координат bbox

# Function to display images with bounding boxes
# Function to display images with bounding boxes
def display_image(img, bbox_coords=[], pred_coords=[], normalize=False):
    if normalize:
        img = img / 255

    while len(pred_coords) > 0:
        xcenter, ycenter, xwidth, ywidth, xheight, yheight, xlength, ylength = pred_coords[:8]
        pred_coords = pred_coords[8:]
        cv2.line(img, (int(xcenter), int(ycenter)), (int(xwidth), int(ywidth)), (255, 0, 0), 2)  # Red channel
        cv2.line(img, (int(xcenter), int(ycenter)), (int(xheight), int(yheight)), (0, 255, 0), 2)  # Green channel
        cv2.line(img, (int(xcenter), int(ycenter)), (int(xlength), int(ylength)), (0, 0, 255), 2)  # Blue channel

    while len(bbox_coords) > 0:
        xcenter, ycenter, xwidth, ywidth, xheight, yheight, xlength, ylength = bbox_coords[:8]
        bbox_coords = bbox_coords[8:]
        cv2.line(img, (int(xcenter), int(ycenter)), (int(xwidth), int(ywidth)), (255, 255, 0), 2)  # Yellow channel
        cv2.line(img, (int(xcenter), int(ycenter)), (int(xheight), int(yheight)), (255, 0, 255), 2)  # Magenta channel
        cv2.line(img, (int(xcenter), int(ycenter)), (int(xlength), int(ylength)), (0, 255, 255), 2)  # Cyan channel

    plt.imshow(img)
    plt.axis("off")

# Function to create a data generator
# Function to create a data generator
def data_generator(csv_path, image_dir, batch_size=BATCH_SIZE, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    df = pd.read_csv(csv_path)
    while True:
        images = np.zeros((batch_size, img_height, img_width, 3))
        bbox_coords = np.zeros((batch_size, MAX_LENGTH))

        for i in range(batch_size):
            rand_index = np.random.randint(0, df.shape[0])
            row = df.iloc[rand_index]
            image_path = os.path.join(image_dir, row['image'])
            img = cv2.imread(image_path)
            img = cv2.resize(img, (img_width, img_height)) / 255.0
            images[i] = img
            bbox_coords[i] = np.array([
                row['xcenter'], row['ycenter'],
                row['xwidth'], row['ywidth'],
                row['xheight'], row['yheight'],
                row['xlength'], row['ylength']
            ])

        yield {'image': images}, {'coordinates': bbox_coords}

# Test the generator
def test(model):
    plt.figure(figsize=(15, 7))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        example, label = next(data_generator(csv_path, image_dir, batch_size=1))
        X = example['image']
        y = label['coordinates']

        img = X[0]
        gt_coords = y[0]
        pred_bbox = model.predict(X)[0]
        display_image(img, bbox_coords=gt_coords, pred_coords=pred_bbox)
    plt.show()


class TestImages(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test(self.model)

def train():
    img, label = next(data_generator(csv_path, image_dir, batch_size=1))
    img = img["image"][0]
    bbox_coords = label['coordinates'][0]
    display_image(img, bbox_coords=bbox_coords)

    # Create the model
    shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    inputs = Input(shape=shape, name="image")
    x = inputs
    for i in range(5):
        n_filters = 2 ** (i + 2)
        x = Conv2D(n_filters, 3, activation="relu", padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(MAX_LENGTH, activation='relu', name='coordinates')(x)

    model = tf.keras.models.Model(inputs, output)
    model.summary()

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Function to test the model during training

    # Training the model
    with tf.device('/GPU:0'):
        _ = model.fit(data_generator(csv_path, image_dir), epochs=9, steps_per_epoch=500, callbacks=[TestImages()])

    # Save the model
    model.save(saved_model_path)

