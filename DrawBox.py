import tkinter as tk
from PIL import Image, ImageTk

class ImageBox:
    def __init__(self, master, image_path):
        # load the image from file
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        # create a canvas to display the image
        self.canvas = tk.Canvas(master, width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.pack()

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
        # reset box coordinates
        self.box_start = None
        self.box_end = None

    def draw_box(self):
        # remove any existing box
        self.canvas.delete("box")

        if self.box_start and self.box_end:
            # draw a new box
            x0, y0 = self.box_start
            x1, y1 = self.box_end
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", tags="box")

root = tk.Tk()
app = ImageBox(root, "C:/Users/jingc/OneDrive/Documents/Y4S1/FYP/PD/Test/images/FLIR0032.jpg")
root.mainloop()