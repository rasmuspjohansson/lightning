import argparse
import models
import datasets_module
import os
import torch
import lightning_module
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger 
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

#this gave 0.52 validation accuracy: python train.py --model vit --dataset cifar10 -n localVIT100epochADAM --learning_rate_schedule NONE -o adam --lr 0.001 --loss cross_entropy. I need to get https://github.com/kentaroy47/vision-transformers-cifar10 to run and modify it with locally conected layer. 
 



import yaml

def train(args):

    
    for config_file in args.config:
        with open(config_file, 'r') as f:
            experiment_settings =  yaml.safe_load(f)

        #checkpoint = torch.load(experiment_settings["state_dict_path_load"])

        save_dir = experiment_settings["save_dir"]
        dataset = datasets_module.get_dataset(experiment_settings)
        model = models.get_model(experiment_settings,dataset.n_classes)
        lightning_object = lightning_module.Lightning_module(dataset=dataset,model=model,args=experiment_settings)


        if experiment_settings["state_dict_path_load"]:
            print("loading the weights from a checpoint (no lr or other meta parameters)")
            lightning_object.load_state_dict(torch.load(experiment_settings["state_dict_path_load"])["state_dict"])
            print("loaded the state dict")


        #checkpoint_callback = ModelCheckpoint(dirpath='saved_models/'+args.experment_name+'/',every_n_epochs=args.save_interval)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        #default is to log once for every epoch
        #if not args.log_every_n_steps:
        #    args.log_every_n_steps = len(dataset.dataset_train/args.batchsize)

        #callbacks=[TQDMProgressBar(refresh_rate=20),lr_monitor,checkpoint_callback],
        trainer = Trainer(
            accelerator=experiment_settings["accelerator"],
            devices="auto",  # limiting got iPython runs
            max_epochs=experiment_settings["epochs"],
            callbacks=[TQDMProgressBar(refresh_rate=20),lr_monitor],
            logger=[TensorBoardLogger(save_dir=save_dir), CSVLogger(save_dir=save_dir)],
            gradient_clip_val=float(experiment_settings["gradient_clip_val"]),accumulate_grad_batches=int(experiment_settings["accumulate_grad_batches"]),


        )
        #if experiment_settings["state_dict_path_load"]:
        #    lightning_object.load_state_dict(torch.load(experiment_settings["state_dict_path_load"]))
        # Fit model
        if experiment_settings["resume_training"]:
            print("resume training from state in checkpoint, (loading weights lr ,epoch nr etc)")
            trainer.fit(lightning_object,ckpt_path=experiment_settings["resume_training"])
        else:
            trainer.fit(lightning_object)



if __name__ == "__main__":
    """
    Given one or more config-files that defines a training,
    Trains a model on a dataset.
    
    
    """
    usage_example="example usage: \n "+"--model rasnet --dataset CIFAR10 --epochs 100 --lr 0.01 --load_model path/to/dict.pth"
    # Initialize parser
    parser = argparse.ArgumentParser(
        epilog=usage_example,
        formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("-c", "--config", help="path to config file ",nargs ='+',required=True)
    

    args = parser.parse_args()
    train(args)
