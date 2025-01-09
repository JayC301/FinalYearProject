import albumentations as A
from PIL import ImageTk, Image
import numpy as np
import os

# Load the image

# Define the transformation pipeline
transform = A.Compose(
    [
    A.HorizontalFlip(p=0.5),   # Flip the image horizontally with probability 0.5
    A.Rotate(limit=90, p=0.5), # Rotate the image by a random angle up to 90 degrees with probability 0.5
    A.RandomBrightnessContrast(p=0.2), # Randomly adjust the brightness and contrast of the image with probability 0.2
    A.VerticalFlip(p=0.5),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    A.OneOf(
        [
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5)
        ], p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=[])
)

img_folder = r"C:\Users\jingc\OneDrive\Desktop\images"
my_img = os.listdir(img_folder)
label_folder = r"C:\Users\jingc\OneDrive\Desktop\labels"
my_label = os.listdir(label_folder)

image_list = []
saved_boxxes = []
# Apply the transformation pipeline to the image
for i in range(len(my_img)):
    for j in range(5):
        current_path = img_folder + "/" + my_img[i]
        image = np.array(Image.open(current_path))
        # image = np.array(image)
        img_name_w_filetype = current_path.split('/')[-1]
        img_name = img_name_w_filetype.split('.')[0]

        with open(label_folder + "/" + my_label[i+1], "r") as label:
            coord = label.read().split()
        bboxes = [(float(coord[1]), float(coord[2]), float(coord[3]), float(coord[4]))]

        transformed = transform(image=image, bboxes = bboxes)
        transformed_image = transformed['image']
        image_list.append(transformed_image)
        saved_boxxes.append(transformed['bboxes'][0])
        # Save the transformed image
        Image.fromarray(transformed_image).save(img_name + "_" + str(j) + '.jpg')
        with open(img_name + "_" + str(j) + ".txt", "w") as file:
            file.write('15' + ' ' + str(saved_boxxes[i*5 + j][0]) + ' ' + str(saved_boxxes[i*5 + j][1]) + ' ' + str(saved_boxxes[i*5 + j][2]) + ' ' + str(saved_boxxes[i*5 + j][3]))

