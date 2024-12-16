# lightning
Deep learning with lightning

## installation
mamba env create -f environment.yml
mamba activate lightning_environment


## training a resnet 50 on buildingdetection with 10 channels 
No data augmentation and using a one-cykle lr schedule without weight decay

python train.py --config ../../configs/segmentation_bygning_resnet.yml --no_logger

Logs show  final validation accuracy at 0.991


