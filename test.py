import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from const import IMG_WIDTH, IMG_HEIGHT, saved_model_path





MAX_LENGTH = 8  # число координат bbox

def data_generator(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # Змінюємо розмір до очікуваного
    image = image / 255.  # Нормалізуємо значення пікселів до [0, 1]
    images = np.expand_dims(image, axis=0)
    bbox_coords = np.zeros((1, MAX_LENGTH))  # Assuming single image, so batch size is 1
    return {'image': images}, {'coordinates': bbox_coords}

def display_image(img, bbox_coords=[], pred_coords=[], normalize=False):
    if normalize:
        img *= 255.
        img = img.astype(np.uint8)

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
def test(model, image_path):
    plt.figure(figsize=(5, 5))
    example, label = data_generator(image_path)
    X = example['image']
    y = label['coordinates']

    img = X[0]
    gt_coords = y[0]
    pred_bbox = model.predict(X)[0]
    print("Predicted coordinates:", pred_bbox)

    # Ensure predicted coordinates are within the image bounds
    pred_bbox = np.clip(pred_bbox, 0, max(IMG_WIDTH, IMG_HEIGHT))

    # Display both ground truth and predicted bounding boxes
    display_image(img, bbox_coords=gt_coords, pred_coords=pred_bbox, normalize=True)
    plt.show()

def test_f(image_dir):
    loaded_model = tf.keras.models.load_model(saved_model_path)
    print("Testing on images in directory:", image_dir)

    # Iterate through all files in the directory
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):
            print("Testing on image:", image_path)
            test(loaded_model, image_path)


test_f(r'C:\Users\PC\PycharmProjects\3D_model_car_object_detection\json3d\norm_img')