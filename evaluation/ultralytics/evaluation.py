from ultralytics import YOLO
import os
folder = 'F:\\model_pts\\' # Change to the model path (can be downloaded from models/)
for file in os.listdir(folder):
    print(file)
    model = YOLO(folder+file)
    model.val(data='constellation_v2.yaml')