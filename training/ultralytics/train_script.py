from ultralytics import YOLO
import numpy as np
import wandb

MODE = 'no_aug'
BATCH_SIZE = 32
PROJECT_DIR = '/mnt/nfs/' # Set the project directory
DEVICE = 'cuda:0'

if MODE == 'no_aug':
    models_to_train = [{'dataset': 'constellation_2.yaml', 'model':'yolov8x.pt', 'epochs': 125, 'run_name': 'yolov8x_v2_coco_no_aug'},
                       {'dataset': 'constellation_2.yaml', 'model':'pretrained/visdronex.pt', 'epochs': 125, 'run_name': 'yolov8x_v2_visdrone_no_aug'},
                       {'dataset': 'constellation_2.yaml', 'model':'pretrained/carlax.pt', 'epochs': 125, 'run_name': 'yolov8x_v2_carla_no_aug'},
                       {'dataset': 'constellation_2.yaml', 'model':'pretrained/sddx.pt', 'epochs': 125, 'run_name': 'yolov8x_v2_sdd_no_aug'},
                       {'dataset': 'constellation_2.yaml', 'model':'yolov8x.yaml', 'epochs': 125, 'run_name': 'yolov8x_v2_random_no_aug', 'pretrained': False},]
elif MODE == 'aug':
    """ Detailed Albumentations Settings:
    Blur(p=0.1, blur_limit=(3, 9)), MotionBlur(p=0.15, blur_limit=(3, 11), allow_shifted=True), RandomBrightnessContrast(p=0.2, brightness_limit=(-0.3, 0.3), contrast_limit=(-0.2, 0.2), brightness_by_max=True), RandomShadow(p=0.2, shadow_roi=(0.2, 0, 1, 1), num_shadows_lower=2, num_shadows_upper=4, shadow_dimension=3), RandomRain(p=0.15, slant_lower=-10, slant_upper=10, drop_length=35, drop_width=1, drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.8, rain_type=None), ToGray(p=0.01)
    """
    models_to_train = [{'dataset': 'constellation_2.yaml', 'model':'yolov8x.pt', 'epochs': 125, 'run_name': 'yolov8x_v2_coco_aug'},
                       {'dataset': 'constellation_2.yaml', 'model':'pretrained/visdronex.pt', 'epochs': 125, 'run_name': 'yolov8x_v2_visdrone_aug'},
                       {'dataset': 'constellation_2.yaml', 'model':'pretrained/carlax.pt', 'epochs': 125, 'run_name': 'yolov8x_v2_carla_aug'},
                       {'dataset': 'constellation_2.yaml', 'model':'pretrained/sddx.pt', 'epochs': 125, 'run_name': 'yolov8x_v2_sdd_aug'},
                       {'dataset': 'constellation_2.yaml', 'model':'yolov8x.yaml', 'epochs': 125, 'run_name': 'yolov8x_v2_random_aug', 'pretrained': False},]

# Load a model
N_WORKERS = 8
for model_to_train in models_to_train:
    model = YOLO(model_to_train['model'])  # build a new model from YAML
    config = {
            "EPOCHS": model_to_train['epochs'],
            "BATCH": BATCH_SIZE
        }
    with wandb.init(project='constellation', config=config, name=model_to_train['run_name']) as run:
        results = model.train(data=model_to_train['dataset'], 
                              epochs=model_to_train['epochs'], 
                              imgsz=832,
                              batch=BATCH_SIZE, 
                              save_period=1,
                              patience=0,
                              project = PROJECT_DIR,
                              pretrained=(model_to_train['pretrained'] if 'pretrained' in model_to_train else True),
                              cache=True, name = model_to_train['run_name'], device=DEVICE)