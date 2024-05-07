Docker Base file: **https://github.com/Qhite/Cuda-11.7-cudnn8-Dockerfile**


Python Requirements: bash ./install.sh


For Training: python train.py --config ./config/example.yaml
* You can customize model and train setup by YAML file


For Evaluation: python eval.py --config ./log/pretrained
* Make sure the './log/pretrained' is path of a directory containing both the YAML file and the model pth.tar file.
