import torch
import numpy as np
import cv2
import os
import time
import pandas as pd


model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/jingc/OneDrive/Documents/Y4S1/FYP/PD/yolov5/runs/train/exp26/weights/last.pt', force_reload=True)
img = os.path.join('C:/Users/jingc/OneDrive/Desktop/images/FLIR0106_4.jpg')

results = model(img)
# results.crop(save = True)
df = results.pandas().xyxy[0]
with open(r"C:\Users\jingc\OneDrive\Documents\Y4S1\FYP\PD\Extraction\module_coord\aaa.txt", "w") as f:
    for i in range(4):
        f.write(str(df.iloc[0, i]) + " ") 

cv2.imwrite(r"C:\Users\jingc\OneDrive\Documents\Y4S1\FYP\PD\Extraction\module\aaa.jpg", np.squeeze(results.render()))

# print(value)

# while True:

#     cv2.imshow('YOLO', np.squeeze(results.render()))
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()