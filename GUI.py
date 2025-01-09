from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import pandas as pd
import torch
import numpy as np
import cv2

#load extraction model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/jingc/OneDrive/Documents/Y4S1/FYP/PD/yolov5/runs/train/exp26/weights/last.pt', force_reload=True)

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
    global module_folder
    global module_coord_folder

    img_folder = filedialog.askdirectory(initialdir=".",title="Choose image folder")
    module_folder = os.path.join(img_folder, 'module')
    module_coord_folder = os.path.join(img_folder, 'module_coord')

    if not os.path.exists(module_folder):
        os.makedirs(module_folder)

    if not os.path.exists(module_coord_folder):
        os.makedirs(module_coord_folder)

    # Specify the file extension or pattern of the documents
    file_extension = '.jpg'

    # Iterate over the files in the directory and filter for documents
    my_img = [file for file in os.listdir(img_folder) if file.endswith(file_extension)]
    for i in range(len(my_img)):
        current_path = img_folder + "/" + my_img[i]
        # results = model(current_path)
        # results.crop(save = True)
        # df = results.pandas().xyxy[0]
        # img_name_w_filetype = current_path.split('/')[-1]
        # img_name = img_name_w_filetype.split('.')[0]
        # with open(os.path.join(module_coord_folder, img_name + '.txt'), "w") as f:
        #     if df.any().any():
        #         for j in range(4):
        #             f.write(str(df.iloc[0, j]) + " ")

        # cv2.imwrite(os.path.join(module_folder, img_name + '.jpg'), np.squeeze(results.render()))

    my_img = [file for file in os.listdir(module_folder) if file.endswith(file_extension)]
    if len(my_img) > 0:
        current_path = module_folder + "/" + my_img[0]
        current_img = ImageTk.PhotoImage(Image.open(current_path))
        img_label = Label(image=current_img)
        img_label.grid(row=1, column=1)
    # my_label = Label(frame3, text=my_img).grid(row = 1, column = 0)

image_number = 0
btn1 = Button(frame1, text="Choose dir", command=open_dir).grid(row = 0, column = 0, sticky = EW, pady = 10, padx = 10)

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
    # # calculate center coordinates, width, and height of box
    # center_x = (x0 + x1) / 2
    # center_y = (y0 + y1) / 2
    # box_width = x1 - x0
    # box_height = y1 - y0

    # # scale coordinates and dimensions to range [0,1]
    # center_x /= image_width
    # center_y /= image_height
    # box_width /= image_width
    # box_height /= image_height

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
def forward(image_number):
	global img_label
	global button_forward
	global button_back
	global current_img
	global current_path
	
	img_label.grid_forget()
	current_path = module_folder + "/" +my_img[image_number+1]
	current_img = ImageTk.PhotoImage(Image.open(current_path))
	img_label = Label(image= current_img)
	button_forward = Button(frame2, text=">>", command= lambda:forward(image_number+1))
	button_back = Button(frame2, text="<<", command= lambda: back(image_number-1))
	
	if image_number ==len(my_img)-2:
		button_forward = Button(frame2, text=">>", state=DISABLED)
		
	img_label.grid(row=1, column=1)
	button_back.grid(row=0, column=0, sticky=EW)
	button_exit.grid(row=0, column=1, sticky=EW)
	button_forward.grid(row=0, column=2, sticky=EW)

def back(image_number):
	global img_label
	global button_forward
	global button_back
	global current_img
	global current_path

	img_label.grid_forget()
	current_path = module_folder + "/" +my_img[image_number-1]
	current_img = ImageTk.PhotoImage(Image.open(current_path))
	img_label = Label(image=current_img)
	button_forward = Button(frame2, text=">>", command= lambda: forward(image_number+1))
	button_back = Button(frame2, text="<<", command= lambda: back(image_number-1))

	if image_number == 1:
		button_back = Button(frame2, text="<<", state=DISABLED)
		
	img_label.grid(row=1, column=1)
	button_back.grid(row=0, column=0, sticky=EW)
	button_exit.grid(row=0, column=1, sticky=EW)
	button_forward.grid(row=0, column=2, sticky=EW)

button_back = Button(frame2, text="<<", command= back)
button_exit = Button(frame2, text="Process", command= root.quit)
button_forward = Button(frame2, text=">>", command= lambda: forward(2))
button_back.grid(row=0, column=0, sticky=EW)
button_exit.grid(row=0, column=1, sticky=EW)
button_forward.grid(row=0, column=2, sticky=EW)



#Creating frame3's message box
Msgbox = Label(frame3, text="Detected hotspot message box").grid(row = 0, column = 0)

root.mainloop()