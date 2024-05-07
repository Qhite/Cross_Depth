Docker Base file: **https://github.com/Qhite/Cuda-11.7-cudnn8-Dockerfile**


Python Requirements: bash ./install.sh


For Training: python train.py --config ./config/example.yaml
* You can customize model and train setup by YAML file


For Evaluation: python eval.py --config ./log/pretrained
* Make sure the './log/pretrained' is path of a directory containing both the YAML file and the model pth.tar file.

KITTI Dataset
* Marge KITTI RAW and KITTI Depth directory

kitti dataset  
├── 2011_09_26  
│   ├── 2011_09_26_drive_0001_sync  
│   │   ├── image_00  
│   │   ├── image_01  
│   │   ├── image_02  
│   │   ├── image_03  
│   │   ├── oxts  
│   │   ├── proj_depth  
│   │   └── velodyne_points  
│   ├── 2011_09_26_drive_0002_sync  
│   └── ...  
├── 2011_09_28  
└── ...  
