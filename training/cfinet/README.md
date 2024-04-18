# CFINet Benchmark on COSMOS Dataset

## Reference
"Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning", [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_Small_Object_Detection_via_Coarse-to-fine_Proposal_Generation_and_Imitation_Learning_ICCV_2023_paper.html).

## Instructions
CFINet uses MMCV/MMDetection for training/evaluation of models. The dataset used for training/evaluation must first be converted to COCO format. We then need to set-up the dependencies:
1. Create a virtual environment and install the following dependencies
    - Python 3.8
    ```
    conda create -n <my_env> python=3.8
    conda activate <my_env>
    ```
    - PyTorch 1.10.0, TorchVision 0.11.0
    ```
    pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    - mmcv-full
    ```
    pip install -U openmim
    mim install mmcv-full
    ```
    
2. Git clone the [CFINet repo](https://github.com/shaunyuan22/CFINet/tree/master) and install it as an editable package
   ```
   git clone https://github.com/shaunyuan22/CFINet
   cd CFINet
   pip install -v -e .
   ```

3. Place the `faster_rcnn_r50_fpn_cfinet_1x_cosmos.py` config file inside the `CFINet/configs` directory.
   
4. In the config file, edit the `data` section to modify the path to the dataset:
   ```
   data = dict(
      train=dict(
          type=dataset_type,
          img_prefix='/<your_path>/constellation/images/train',
          classes=classes,
          ann_file='/<your_path>/constellation/train_coco.json',
          pipeline=train_pipeline),
      val=dict(
          type=dataset_type,
          img_prefix='/<your_path>/constellation/images/val',
          classes=classes,
          ann_file='/<your_path>/constellation/val_coco.json',
          pipeline=test_pipeline),
      test=dict(
          type=dataset_type,
          img_prefix='/<your_path>/constellation/images/val',
          classes=classes,
          ann_file='/<your_path>/constellation/val_coco.json',
          pipeline=test_pipeline))
   ```

5. In the config file, a work_dir is specified (`con_queue_dir="./work_dirs/roi_feats/cfinet"`). Create this path under the `CFINet/` root directory. (You can also change this to a directory of your choice as long as it exists.) This directory will be used to save the logs/checkpoints.

6. To run training, use the following command:
   ```
   python tools/train.py configs/faster_rcnn_r50_fpn_cfinet_1x_cosmos.py
   ```
   
7. (WANDB) The config has a WandbLoggerHook added to it for logging to wandb. This can be disabled by removing the hook:
   ```
   log_config = dict(
      interval=50,
      hooks=[
          dict(type='TextLoggerHook'),
          dict(type='WandbLoggerHook',
              init_kwargs={'project': 'constellation'},
              interval=50,
              log_artifact=True
              )
      ])
   ```

8. Any other changes (such as hyperparameters, evaluation config, etc.) can be made to the config file accordingly. Refer to the MMDetection docs for train/test config usage (https://mmdetection.readthedocs.io/en/latest/user_guides/config.html).