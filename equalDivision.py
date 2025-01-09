import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread("C:/Users/jingc/OneDrive/Documents/Y4S1/FYP/PD/Test/images/FLIR0032.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the number of segments
num_segments = 4

# Calculate the height and width of each segment
height, width = gray.shape[:2]
segment_height = height // num_segments
segment_width = width // num_segments

# Calculate the mean value of each segment
segment_means = []
for i in range(num_segments):
    for j in range(num_segments):
        # Determine the region of interest
        y_start = i * segment_height
        y_end = (i + 1) * segment_height
        x_start = j * segment_width
        x_end = (j + 1) * segment_width
        roi = gray[y_start:y_end, x_start:x_end]

        # Draw the boundary box on the original image
        color = (0, 0, 255) # Red color
        thickness = 2
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), color, thickness)
        
        # Calculate the mean value of the region of interest
        mean_val = np.mean(roi)
        segment_means.append(mean_val)

# Print the mean value of each segment
print(segment_means)

# Show the image with the boundary boxes
cv2.imshow('Segmented Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()