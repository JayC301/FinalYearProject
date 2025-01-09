from PIL import ImageTk, Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

 
# Opens a image in RGB mode
current_folder = r'C:\Users\jingc\OneDrive\Desktop\FYP_HS\Thermal'
img_list = os.listdir(current_folder)
for i in range(len(img_list)):
    current_path =os.path.join(current_folder, img_list[i])
    im = cv2.imread(current_path)
    img_name_w_filetype = current_path.split('\\')[-1]
    img_name = img_name_w_filetype.split('.')[0]
    label_name = int(img_name[-1]) + 1

    #open label file which is in xywh format
    with open(r'C:\Users\jingc\OneDrive\Desktop\FYP_HS\module_coord' + '/' + img_name[:-1] + str(label_name) + '.txt', "r") as label:
        coord = label.read().split()

    # #obtain image size
    img_height, img_width = im.shape[:2]

    x0 = int(float(coord[0]) - img_width*0.06)
    y0 = int(float(coord[1]) - img_height*0.05)
    x1 = int(float(coord[2]) + img_width*0.06)
    y1 = int(float(coord[3]) + img_height*0.05)


    # Cropped image of above dimension
    module = im[y0:y1, x0:x1]
    
    #--------------------------------------------------------------------------------------------------------
    #Stat

    #Grayscale conversion
    gray = cv2.cvtColor(module, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(r'C:\Users\jingc\OneDrive\Desktop\FYP_HSv1\xxx_gray' + '\\' + img_name_w_filetype, gray)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(10,10))
    img_eq = clahe.apply(gray)
    # cv2.imwrite(r'C:\Users\jingc\OneDrive\Desktop\FYP_HSv1\xxx_eq' + '\\' + img_name_w_filetype, img_eq)

    # # Apply Gaussian filter
    img_blur = cv2.GaussianBlur(img_eq, (91, 91), 0)
    # cv2.imwrite(r'C:\Users\jingc\OneDrive\Desktop\FYP_HSv1\xxx_blur' + '\\' + img_name_w_filetype, img_blur)

    # Define the number of segments
    num_segments = 10

    # Calculate the height and width of each segment
    height, width = gray.shape[:2]
    segment_height = height // num_segments
    segment_width = width // num_segments
    print(segment_height*segment_width)
    # Calculate the mean value of each segment
    segment_means = []
    segment_medians = []
    segment_stdevs = []
    for i in range(num_segments):
        for j in range(num_segments):
            # if  i>0 and j>0 and i<(num_segments-1) and j<(num_segments-1):
                # Determine the region of interest
                y_start = i * segment_height
                y_end = (i + 1) * segment_height
                x_start = j * segment_width
                x_end = (j + 1) * segment_width
                roi = module[y_start:y_end, x_start:x_end]
                
                # Calculate the mean value of the region of interest
                mean_val = np.mean(roi)
                median_val = np.median(roi)
                stdev_val = np.std(roi)
                segment_means.append(mean_val)
                segment_medians.append(median_val)
                segment_stdevs.append(stdev_val)

    mean_segment_intensity = np.mean(segment_means)
    module_std = np.std(segment_means)
    module_median = np.median(segment_means)
    upper_boundary = module_median + module_std*1.43
    lower_boundary = module_median - module_std

    hotspot_mean = []
    for i in range(num_segments):
        for j in range(num_segments):
            if i==0 and j==0:
                k=0
            # Determine the region of interest
            y_start = i * segment_height
            y_end = (i + 1) * segment_height
            x_start = j * segment_width
            x_end = (j + 1) * segment_width
            cv2.rectangle(module, (x_start, y_start), (x_end, y_end), (0,0,255), 1)
            cv2.putText(module, str(int(segment_means[k])), (x_start, y_end), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.imwrite(r'C:\Users\jingc\OneDrive\Desktop\FYP_HS\Values' + '\\' + img_name_w_filetype, module)
            if segment_means[k] > upper_boundary:
                hotspot_mean.append(segment_means[k])
                # Draw the boundary box on the original image
                # color = (0, 255, 0) # Red color
                # thickness = 2
                # cv2.rectangle(module, (x_start, y_start), (x_end, y_end), color, thickness)
                # cv2.putText(module, str(int(segment_means[k])), (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            k += 1
    # Show the image with the boundary boxes
    # cv2.imwrite(r'C:\Users\jingc\OneDrive\Desktop\FYP_HSv1\xxx' + '\\' + img_name_w_filetype, module)

    # Plot histograms of the mean, median, and standard deviation values
    # plt.hist(segment_means, color='blue', label='Mean')
    # plt.legend(loc='upper right')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Color Intensity Values')
    # plt.axvline(x=module_median, color='r', linestyle='dashed', linewidth=2, label=f'Mean = {mean_segment_intensity:.2f}')
    # plt.axvline(x=upper_boundary, color='r', linestyle='dashed', linewidth=2, label=f'Mean = {mean_segment_intensity:.2f}')
    # plt.axvline(x=lower_boundary, color='r', linestyle='dashed', linewidth=2, label=f'Mean = {mean_segment_intensity:.2f}')
    # plt.savefig(r"C:\Users\jingc\OneDrive\Desktop\FYP_HSv1\xxx_hist" + '\\' + img_name_w_filetype)
    # plt.clf()

