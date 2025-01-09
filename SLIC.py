import cv2
import numpy as np

# Load input image
img = cv2.imread("C:/Users/jingc/OneDrive/Documents/Y4S1/FYP/PD/Test/images/FLIR0032.jpg")

# Apply SLIC segmentation
num_segments = 1
slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=150, ruler=10.0)
slic.iterate(num_iterations=10)
labels = slic.getLabels()

# Obtain boundary mask
mask = slic.getLabelContourMask()

# Create bounding boxes around superpixels
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    boxes.append((x, y, x + w, y + h))

# Slice image according to bounding boxes
slices = []
for box in boxes:
    slices.append(img[box[1]:box[3], box[0]:box[2]])

# Display output
cv2.imshow('Input Image', img)
cv2.imshow('SLIC Segmentation', mask)
for i, slice in enumerate(slices):
    # cv2.imshow(f'Slice {i}', slice)
    print(i)

cv2.waitKey(0)
cv2.destroyAllWindows()