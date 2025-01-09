import albumentations as A
import cv2
import numpy as np

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("C:/Users/jingc/OneDrive/Documents/Y4S1/FYP/PD/Test/images/FLIR0032.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bbox = [0.5, 0.5, 0.5, 0.5]

# Define the augmentations you want to apply
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
], bbox_params=A.BboxParams(format='yolo'))

# Apply the augmentations to the image and its bounding boxes
augmented = transform(image=image, bboxes=bbox)

# Retrieve the augmented image and its bounding boxes
augmented_image = augmented['image']
augmented_bboxes = augmented['bboxes']



# Augment an image
transformed_image, transformed_bbox = transform(image, bbox)
# transformed_image = transformed["image"]
# transformed_bbox = transformed["bboxes"]