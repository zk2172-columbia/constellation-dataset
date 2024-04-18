from ultralytics import YOLO
import os
folder = 'F:\\model_pts\\'
for file in os.listdir(folder):
    print(file)
    model = YOLO(folder+file)
    model.val(data='constellation_v2.yaml')