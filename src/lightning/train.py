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
from PIL import Image
import numpy as np
import visualize
import datetime
#this gave 0.52 validation accuracy: python train.py --model vit --dataset cifar10 -n localVIT100epochADAM --learning_rate_schedule NONE -o adam --lr 0.001 --loss cross_entropy. I need to get https://github.com/kentaroy47/vision-transformers-cifar10 to run and modify it with locally conected layer. 
 



import yaml

def train(args):

    
    for config_file in args.config:
        with open(config_file, 'r') as f:
            experiment_settings =  yaml.safe_load(f)

        #checkpoint = torch.load(experiment_settings["state_dict_path_load"])

        save_dir = experiment_settings["save_dir"]
        experment_name = experiment_settings["experment_name"]
        dataset = datasets_module.get_dataset(experiment_settings)


        #It is important to verify that the iamges look good after augmentations (no artifacts or anything else that causes the data to look very differetn than the un-augmetnted dat)

        if experiment_settings["nr_of_images_to_visualize"] >0:
            visualize.visualize_dataset(dataset,experiment_settings)

        model = models.get_model(experiment_settings,dataset.n_classes)
        lightning_object = lightning_module.Lightning_module(dataset=dataset,model=model,args=experiment_settings)


        if experiment_settings["state_dict_path_load"]:
            print("loading the weights from a checpoint (no lr or other meta parameters)")
            lightning_object.load_state_dict(torch.load(experiment_settings["state_dict_path_load"])["state_dict"])
            print("loaded the state dict")


        #We want to save the weights every X hours
        checkpoint_callback1 = ModelCheckpoint(dirpath=save_dir+"/"+"interval_checkpoints",train_time_interval=datetime.timedelta(hours=1.0),filename=experment_name+"-{epoch:02d}-{val_loss:.2f}")
        #We want to save the weight every epoch (save_top_k=-1 makes sure that every checkpoint is kept on disk)
        checkpoint_callback2 = ModelCheckpoint(dirpath=save_dir+"/"+"epoch_checkpoints", every_n_epochs=1,filename=experment_name+"-{epoch:02d}-{val_loss:.2f}",save_top_k=-1)
        # saves top-K checkpoints based on "val_loss" metric
        checkpoint_callback3 = ModelCheckpoint(
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            dirpath=save_dir+"/"+"valloss_checkpoints",
            filename=experment_name+"-{epoch:02d}-{val_loss:.2f}"
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks = [TQDMProgressBar(refresh_rate=20), lr_monitor, checkpoint_callback2,checkpoint_callback3]


        loggers= [CSVLogger(save_dir=save_dir+"_CSV")]
        
        if os.name != "nt" or (("logger" in experiment_settings) and (experiment_settings["logger"]=="TensorBoard")):
            loggers.append(TensorBoardLogger(save_dir=save_dir+"_Tensorboard"))
            print("logging to tensorboard at :"+str(save_dir+"_Tensorboard"))
            if os.name == "nt":
                print("###########################################################################################")
                print("The tensorboard logger causes freezing wen pressing ctr+C on windows.")
                print("if you want to avoid this, remove logger: TensorBoard from your config file")
                print("###########################################################################################")

        else:
            print("###########################################################################################")
            print("The tensorboard logger causes freezing wen pressing ctr+C on windows.")
            print("if you still want to use a tensorboard logger add logger: TensorBoard in your config file")
            print("###########################################################################################")



        #default is to log once for every epoch
        #if not args.log_every_n_steps:
        #    args.log_every_n_steps = len(dataset.dataset_train/args.batchsize)


        trainer = Trainer(
            accelerator=experiment_settings["accelerator"],
            devices="auto",  # limiting got iPython runs
            max_epochs=experiment_settings["epochs"],
            callbacks=callbacks,
            logger=loggers,
            #The tensorboard logger causes freezing wen pressing ctr+C on windows. Commenting it
            #,
            gradient_clip_val=float(experiment_settings["gradient_clip_val"]),accumulate_grad_batches=int(experiment_settings["accumulate_grad_batches"]),
            num_sanity_val_steps =experiment_settings["num_sanity_val_steps"],
            overfit_batches=experiment_settings["overfit_batches"],
            #if we train on a debug dataset we want to be able to logg training accuracy for every epoch!
            log_every_n_steps = int(experiment_settings["log_every_n_steps"])
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
    parser.add_argument("--no_logger", help="not using logger avaids this bug  ", default=False,action='store_true')

  
    
    

    args = parser.parse_args()
    train(args)
