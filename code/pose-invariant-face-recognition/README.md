# Pose-invariant face recognition by DR-GAN

Luan Tran et al

## Requirements
- python 3.x
- pytorch 
- torchvision
- numpy
- scipy
- matplotlib
- pillow
- tensorboardX

## How to use

### Single-Image DR-GAN

1. Modify model function at base_options.py to define single model.
    - Data needs to have ID and pose lables corresponds to each image.
    - If you don't have, default dataset is CFP_dataset. Modify dataroot function at base_options.py.


2. Run train.py to train models
      - Trained models and Loss_log will be saved at "checkpoints" by default. Generated
      pictures will be saved at "result".
      > python train.py
      - You can also use tensorboard to watch the loss graphs in real-time. (Install tensorboard before doing it.)
      > tensorboard --logdir=/home/zhangjunhao/logs (Or the address of dir 'logs' in your folder.）

3. Generate Image with arbitrary pose
      - Change the "save_path" in base_model.py.
      - Specify leaned model's filename by "--pretrained_G" option in base_options.py.
      - Generated images will be saved at specified result directory.
      > python test.py


### Multi-Image DR-GAN

1. Modify model function at base_options.py to define multi model.
    - Data needs to have ID and pose lables corresponds to each image.
    - If you don't have, default dataset is CFP_dataset. Modify dataroot function at base_options.py.


2. Run train.py to train models
      - Trained models and Loss_log will be saved at "checkpoints" by default. Generated
      pictures will be saved at "result".
      > python train.py
      - You can also use tensorboard to watch the loss graphs in real-time. (Install tensorboard before doing it.)   
      > tensorboard --logdir=/home/zhangjunhao/logs (Or the address of dir 'logs' in your folder.）

3. Generate Image with arbitrary pose
      - Change the "save_path" in base_model.py.
      - Specify leaned model's filename by "--pretrained_G" option in base_options.py.
      - Generated images will be saved at specified result directory.
      > python test.py

### References
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{tran2017disentangled,
  title={Disentangled representation learning gan for pose-invariant face recognition},
  author={Tran, Luan and Yin, Xi and Liu, Xiaoming},
  booktitle={CVPR},
  year={2017}
}