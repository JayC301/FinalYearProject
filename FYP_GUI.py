from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import pandas as pd
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

#load extraction model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='''C:/Users/jingc/OneDrive/Documents/Y4S1/FYP/PD/yolov5/runs/train/exp26/weights/last.pt''', force_reload=True)

#Creating main window
root = Tk()
root.title("Module Fault Detection")



#Creating frames
frame1 = LabelFrame(root, text = "Tools", padx = 10, pady = 10)
frame2 = LabelFrame(root, text = "Module Extraction", padx = 10, pady = 10)
frame3 = LabelFrame(root, text = "Hotspot Detection", padx = 10, pady = 10)
frame1.grid(row = 0, column = 0, sticky = NS, rowspan = 2)
frame2.grid(row = 0, column = 1, sticky = EW + N)
frame3.grid(row = 0, column = 2, sticky = EW + N, pady = 5)



###Creating Frame1
#Frame1: Button 1 "Choose Dir"
my_img = []
def open_dir():
    global my_img
    global img_folder
    global img_label
    global current_img
    global current_path
    global module_folder, module_coord_folder, histogram_folder
    global hotspot_path, hotspot_folder, hotspot_img, Himg, Himg_label
    global thermal_folder, thermal_img
    global image_number

    img_folder = filedialog.askdirectory(initialdir=".",title="Choose image folder")
    module_folder = os.path.join(img_folder, 'module')
    module_coord_folder = os.path.join(img_folder, 'module_coord')
    hotspot_folder = os.path.join(img_folder, 'hotspot')
    histogram_folder = os.path.join(img_folder, 'histogram')

    if not os.path.exists(module_folder):
        os.makedirs(module_folder)

    if not os.path.exists(module_coord_folder):
        os.makedirs(module_coord_folder)

    if not os.path.exists(hotspot_folder):
        os.makedirs(hotspot_folder)

    if not os.path.exists(histogram_folder):
        os.makedirs(histogram_folder)

    # Specify the file extension or pattern of the documents
    file_extension = '.jpg'

    # Iterate over the files in the directory and filter for documents
    my_img = [file for file in os.listdir(img_folder) if file.endswith(file_extension)]
    for i in range(len(my_img)):
        current_path = img_folder + "/" + my_img[i]
        results = model(current_path)
        # results.crop(save = True)
        df = results.pandas().xyxy[0]
        img_name_w_filetype = current_path.split('/')[-1]
        img_name = img_name_w_filetype.split('.')[0]
        with open(os.path.join(module_coord_folder, img_name + '.txt'), "w") as f:
            if df.any().any():
                for j in range(4):
                    f.write(str(df.iloc[0, j]) + " ")

        cv2.imwrite(os.path.join(module_folder, img_name + '.jpg'), np.squeeze(results.render()))

    my_img = [file for file in os.listdir(module_folder) if file.endswith(file_extension)]
    if len(my_img) > 0:
        image_number = 0
        current_path = module_folder + "/" + my_img[image_number]
        current_img = ImageTk.PhotoImage(Image.open(current_path))
        img_label = Label(image=current_img)
        img_label.grid(row=1, column=1)
        if image_number == 0:
            button_back = Button(frame2, text="<<", state=DISABLED)
            button_back.grid(row=0, column=0, sticky=EW)
    # my_label = Label(frame3, text=my_img).grid(row = 1, column = 0)

#----------------------------------------------------------------------------------------------------------------------------------

    # Opens a image in RGB mode
    thermal_folder = os.path.join(img_folder, 'Thermal')
    thermal_img = [file for file in os.listdir(thermal_folder)]

    for i in range(len(thermal_img)):
        thermal_path = thermal_folder + "/" + thermal_img[i]
        im = cv2.imread(thermal_path)
        img_name_w_filetype = thermal_path.split('/')[-1]
        img_name = img_name_w_filetype.split('.')[0]
        label_name = int(img_name[-1]) + 1

        #open label file which is in xyxy format
        if os.path.exists(module_coord_folder + '/' + img_name[:-1] + str(label_name) + '.txt'):
            with open(os.path.join(module_coord_folder, img_name[:-1] + str(label_name) + '.txt'), "r") as label:
                coord = label.read().split()
                if len(coord) < 1:
                    coord = [50, 50, 100, 100]

        # #obtain image size
        img_height, img_width = im.shape[:2]

        xmin = int(float(coord[0]) - img_width*0.06)
        ymin = int(float(coord[1]) - img_height*0.05)
        xmax = int(float(coord[2]) + img_width*0.06)
        ymax = int(float(coord[3]) + img_height*0.05)

        # Cropped image of above dimension
        module = im[ymin:ymax, xmin:xmax]
        
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
                roi = img_blur[y_start:y_end, x_start:x_end]
                
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


        for i in range(num_segments):
            for j in range(num_segments):
                if i==0 and j==0:
                    k=0
                    hotspot_num = 0
                # Determine the region of interest
                y_start = i * segment_height
                y_end = (i + 1) * segment_height
                x_start = j * segment_width
                x_end = (j + 1) * segment_width
                # cv2.rectangle(module, (x_start, y_start), (x_end, y_end), (0,0,255), 1)
                if segment_means[k] > upper_boundary:
                    # Draw the boundary box on the original image
                    color = (0, 255, 0) # Red color
                    thickness = 2
                    cv2.rectangle(module, (x_start, y_start), (x_end, y_end), color, thickness)
                    hotspot_num += 1
                k += 1

        # Show the image with the boundary boxes
        cv2.imwrite(os.path.join(hotspot_folder, img_name + '_' + str(hotspot_num) + '.jpg'), module)

        # Plot histograms of the mean, median, and standard deviation values
        plt.hist(segment_means, color='blue', label='Mean')
        plt.legend(loc='upper right')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Color Intensity Values')
        plt.axvline(x=module_median, color='r', linestyle='dashed', linewidth=2, label=f'Mean = {mean_segment_intensity:.2f}')
        plt.axvline(x=upper_boundary, color='r', linestyle='dashed', linewidth=2, label=f'Mean = {mean_segment_intensity:.2f}')
        plt.axvline(x=lower_boundary, color='r', linestyle='dashed', linewidth=2, label=f'Mean = {mean_segment_intensity:.2f}')
        plt.savefig(os.path.join(histogram_folder, img_name_w_filetype))
        plt.clf()

    hotspot_img = [file for file in os.listdir(hotspot_folder)]
    hotspot_path = hotspot_folder + "/" + hotspot_img[0]
    Himg = ImageTk.PhotoImage(Image.open(hotspot_path))
    Himg_label = Label(image=Himg)
    Himg_label.grid(row=1, column=2)
    img_name_w_filetype = hotspot_path.split('/')[-1]
    img_name = img_name_w_filetype.split('.')[0]
    Hnum = img_name.split('_')[-1]
    Hmess = Label(root, text=f"{Hnum} hotspot segment(s) detected")
    Hmess.grid(row=2, column=2)
#--------------------------------------------------------------------------------------------------------------------------------------------

btn1 = Button(frame1, text="Choose dir", command=open_dir).grid(row = 0, column = 0, sticky = EW, pady = 10, padx = 10)

def process():
    global module_coord_folder, histogram_folder
    global hotspot_path, hotspot_folder, hotspot_img, Himg, Himg_label
    global thermal_folder, thermal_img
    global image_number
    global new_coord

    thermal_path = os.path.join(thermal_folder, thermal_img[image_number])
    im = cv2.imread(thermal_path)
    img_name_w_filetype = thermal_path.split('/')[-1].split('\\')[-1]
    img_name = img_name_w_filetype.split('.')[0]
    label_name = int(img_name[-1]) + 1
    #open label file which is in xyxy format
    if os.path.exists(module_coord_folder + '/' + img_name[:-1] + str(label_name) + '.txt'):
        with open(os.path.join(module_coord_folder, img_name[:-1] + str(label_name) + '.txt'), "r") as label:
            new_coord = label.read().split()

    

    # #obtain image size
    img_height, img_width = im.shape[:2]

    new_xmin = int(int(new_coord[0]) - img_width*0.06)
    new_ymin = int(int(new_coord[1]) - img_height*0.05)
    new_xmax = int(int(new_coord[2]) + img_width*0.06)
    new_ymax = int(int(new_coord[3]) + img_height*0.05)

    # Cropped image of above dimension
    module = im[new_ymin:new_ymax, new_xmin:new_xmax]
    
    #--------------------------------------------------------------------------------------------------------
    #Stat

    #Grayscale conversion
    gray = cv2.cvtColor(module, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(r'C:\Users\jingc\OneDrive\Desktop\FYP_HSv1\xxx_gray' + '\\' + img_name_w_filetype, gray)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(10,10))
    img_eq = clahe.apply(gray)
    # cv2.imwrite(r'C:\Users\jingc\OneDrive\Desktop\FYP_HSv1\xxx_eq' + '\\' + img_name_w_filetype, img_eq)

    #  Apply Gaussian filter
    img_blur = cv2.GaussianBlur(img_eq, (91, 91), 0)
    # cv2.imwrite(r'C:\Users\jingc\OneDrive\Desktop\FYP_HSv1\xxx_blur' + '\\' + img_name_w_filetype, img_blur)

    # Define the number of segments
    num_segments = 10

    # Calculate the height and width of each segment
    height, width = gray.shape[:2]
    segment_height = height // num_segments
    segment_width = width // num_segments

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
            roi = img_blur[y_start:y_end, x_start:x_end]
            
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


    for i in range(num_segments):
        for j in range(num_segments):
            if i==0 and j==0:
                k=0
                hotspot_num = 0
            # Determine the region of interest
            y_start = i * segment_height
            y_end = (i + 1) * segment_height
            x_start = j * segment_width
            x_end = (j + 1) * segment_width
            # cv2.rectangle(module, (x_start, y_start), (x_end, y_end), (0,0,255), 1)
            if segment_means[k] > upper_boundary:
                # Draw the boundary box on the original image
                color = (0, 255, 0) # Red color
                thickness = 2
                cv2.rectangle(module, (x_start, y_start), (x_end, y_end), color, thickness)
                hotspot_num += 1
            k += 1

    # Show the image with the boundary boxes
    cv2.imwrite(os.path.join(hotspot_folder, hotspot_img[image_number]), module)
    os.rename(os.path.join(hotspot_folder, hotspot_img[image_number]), hotspot_folder + '/' + img_name + '_' + str(hotspot_num) 
    + '.jpg')
    hotspot_img = [file for file in os.listdir(hotspot_folder)]
    
    # Plot histograms of the mean, median, and standard deviation values
    plt.hist(segment_means, color='blue', label='Mean')
    plt.legend(loc='upper right')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Color Intensity Values')
    plt.axvline(x=module_median, color='r', linestyle='dashed', linewidth=2, label=f'Mean = {mean_segment_intensity:.2f}')
    plt.axvline(x=upper_boundary, color='r', linestyle='dashed', linewidth=2, label=f'Mean = {mean_segment_intensity:.2f}')
    plt.axvline(x=lower_boundary, color='r', linestyle='dashed', linewidth=2, label=f'Mean = {mean_segment_intensity:.2f}')
    plt.savefig(os.path.join(histogram_folder, img_name_w_filetype))
    plt.clf()

    hotspot_path = hotspot_folder + '/' + hotspot_img[image_number]
    Himg_label.grid_forget()
    Himg = ImageTk.PhotoImage(Image.open(hotspot_path))
    Himg_label = Label(image=Himg)
    Himg_label.grid(row=1, column=2)

    img_name_w_filetype = hotspot_path.split('/')[-1]
    img_name = img_name_w_filetype.split('.')[0]
    Hnum = img_name.split('_')[-1]
    Hmess = Label(root, text=f"{Hnum} hotspot segment(s) detected")
    Hmess.grid(row=2, column=2)

#Frame1: Button 2 "Choose save dir"
def choose_save_dir():
	global save_folder
	save_folder = filedialog.askdirectory(initialdir=".", title="Choose save folder")
btn2 = Button(frame1, text="Choose save dir", command=choose_save_dir).grid(row = 1, column = 0, sticky = EW, pady = 10, padx = 10)

#Frame1: Button 3 "Draw Box"
class ImageBox:
    def __init__(self, master, image_path):
        # load the image from file
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        # create a canvas to display the image
        self.canvas = Canvas(master, width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.grid(row=1, column=0)

        # bind mouse events to canvas
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # initialize box coordinates
        self.box_start = None
        self.box_end = None

    def on_press(self, event):
        # save the starting coordinates of the box
        self.box_start = (event.x, event.y)

    def on_drag(self, event):
        # update the box end coordinates and draw the box
        self.box_end = (event.x, event.y)
        self.draw_box()

    def on_release(self, event):
        global x0, y0, x1, y1, image_width, image_height
        # reset box coordinates
         # save box coordinates to file
        if self.box_start and self.box_end:
            x0, y0, x1, y1 = self.get_box_coords()
            image_width = self.image.width
            image_height = self.image.height
            # save_box_coords("box_coords.txt", x0, y0, x1, y1, self.image.width, self.image.height)
        self.box_start = None
        self.box_end = None

    def draw_box(self):
        # remove any existing box
        self.canvas.delete("box")

        if self.box_start and self.box_end:
            # draw a new box
            x0, y0, x1, y1 = self.get_box_coords()
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", tags="box")

    def get_box_coords(self):
        # get the current box coordinates
        x0 = min(self.box_start[0], self.box_end[0])
        y0 = min(self.box_start[1], self.box_end[1])
        x1 = max(self.box_start[0], self.box_end[0])
        y1 = max(self.box_start[1], self.box_end[1])
        return (x0, y0, x1, y1)

def save_box_coords(filename, x0, y0, x1, y1, image_width, image_height):
    # write coordinates and dimensions to file
    with open(filename, "w") as f:
        f.write(f"{x0} {y0} {x1} {y1}")

def NewWindow(image_path):
	global app
	top = Toplevel()
	top.title("Draw Boundary Box")
	app = ImageBox(top, image_path)
	img_name_w_filetype = current_path.split('/')[-1]
	img_name = img_name_w_filetype.split('.')[0]
	filename = os.path.join(module_coord_folder, img_name + '.txt')
	btn_save = Button(top, text="save", command = lambda: save_box_coords(filename, x0, y0, x1, y1, image_width, image_height))
	btn_save.grid(row=0, column=0)
btn3 = Button(frame1, text="Draw box", command= lambda: NewWindow(current_path)).grid(row = 2, column = 0, sticky = EW, pady = 10, padx = 10)


#Creating frame2's display window
def forward():
    global img_label
    global button_forward
    global button_back
    global current_img
    global current_path
    global hotspot_path, hotspot_folder, hotspot_img, Himg, Himg_label
    global image_number

    img_label.grid_forget()
    Himg_label.grid_forget()
    image_number += 1
    current_path = module_folder + "/" + my_img[image_number]
    current_img = ImageTk.PhotoImage(Image.open(current_path))
    img_label = Label(image= current_img)
    hotspot_path = hotspot_folder + "/" + hotspot_img[image_number]
    Himg = ImageTk.PhotoImage(Image.open(hotspot_path))
    Himg_label = Label(image=Himg)
    img_name_w_filetype = hotspot_path.split('/')[-1]
    img_name = img_name_w_filetype.split('.')[0]
    Hnum = img_name.split('_')[-1]
    Hmess = Label(root, text=f"{Hnum} hotspot segment(s) detected")
    button_forward = Button(frame2, text=">>", command= lambda:forward())
    button_back = Button(frame2, text="<<", command= lambda: back())
	
    if image_number == len(my_img)-1:
        button_forward = Button(frame2, text=">>", state=DISABLED)
		
    img_label.grid(row=1, column=1)
    Himg_label.grid(row=1, column=2)
    button_back.grid(row=0, column=0, sticky=EW)
    button_exit.grid(row=0, column=1, sticky=EW)
    button_forward.grid(row=0, column=2, sticky=EW)
    Hmess.grid(row=2, column=2)

def back():
    global img_label
    global button_forward
    global button_back
    global current_img
    global current_path
    global hotspot_path, hotspot_folder, hotspot_img, Himg, Himg_label
    global image_number

    img_label.grid_forget()
    # Himg_label.grid_forget()
    # Hmess.grid_forget()
    image_number -= 1
    current_path = module_folder + "/" + my_img[image_number]
    current_img = ImageTk.PhotoImage(Image.open(current_path))
    img_label = Label(image=current_img)
    hotspot_path = hotspot_folder + "/" + hotspot_img[image_number]
    Himg = ImageTk.PhotoImage(Image.open(hotspot_path))
    Himg_label = Label(image=Himg)
    img_name_w_filetype = hotspot_path.split('/')[-1]
    img_name = img_name_w_filetype.split('.')[0]
    Hnum = img_name.split('_')[-1]
    Hmess = Label(root, text=f"{Hnum} hotspot segment(s) detected")
    button_forward = Button(frame2, text=">>", command= lambda: forward())
    button_back = Button(frame2, text="<<", command= lambda: back())

    if image_number == 0:
        button_back = Button(frame2, text="<<", state=DISABLED)
        
    img_label.grid(row=1, column=1)
    Himg_label.grid(row=1, column=2)
    button_back.grid(row=0, column=0, sticky=EW)
    button_exit.grid(row=0, column=1, sticky=EW)
    button_forward.grid(row=0, column=2, sticky=EW)
    Hmess.grid(row=2, column=2)

button_back = Button(frame2, text="<<", command= back)
button_exit = Button(frame2, text="Process", command= process)
button_forward = Button(frame2, text=">>", command= lambda: forward())
button_back.grid(row=0, column=0, sticky=EW)
button_exit.grid(row=0, column=1, sticky=EW)
button_forward.grid(row=0, column=2, sticky=EW)



#Creating frame3's message box
Msgbox = Label(frame3, text="Detected hotspot message box").grid(row = 0, column = 0)

root.mainloop()