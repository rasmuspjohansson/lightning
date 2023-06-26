import argparse
import models
import datasets_module
import os
import torch
import lightning_module
from pytorch_lightning import LightningModule, Trainer
from PIL import Image
import numpy as np
import visualize
import yaml


def infer(args):
    for config_file in args.config:
        with open(config_file, 'r') as f:
            experiment_settings = yaml.safe_load(f)

        # checkpoint = torch.load(experiment_settings["state_dict_path_load"])

        save_dir = experiment_settings["save_dir"]
        dataset = datasets_module.get_dataset(experiment_settings)

        # It is important to verify that the iamges look good after augmentations (no artifacts or anything else that causes the data to look very differetn than the un-augmetnted dat)

        if experiment_settings["nr_of_visualizations"] > 0:
            visualize.visualize(dataset, experiment_settings)

        model = models.get_model(experiment_settings, dataset.n_classes)
        lightning_object = lightning_module.Lightning_module(dataset=dataset, model=model, args=experiment_settings)

        if experiment_settings["state_dict_path_load"]:
            print("loading the weights from a checpoint (no lr or other meta parameters)")
            lightning_object.load_state_dict(torch.load(experiment_settings["state_dict_path_load"])["state_dict"])
            print("loaded the state dict")

        # checkpoint_callback = ModelCheckpoint(dirpath='saved_models/'+args.experment_name+'/',every_n_epochs=args.save_interval)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')



        # callbacks=[TQDMProgressBar(refresh_rate=20),lr_monitor,checkpoint_callback],
        trainer = Trainer(
            accelerator=experiment_settings["accelerator"],
            devices="auto",  # limiting got iPython runs
            max_epochs=experiment_settings["epochs"],
            callbacks=[TQDMProgressBar(refresh_rate=20), lr_monitor],
            logger=loggers,
            # The tensorboard logger causes freezing wen pressing ctr+C on windows. Commenting it
            # ,
            gradient_clip_val=float(experiment_settings["gradient_clip_val"]),
            accumulate_grad_batches=int(experiment_settings["accumulate_grad_batches"]),
            num_sanity_val_steps=experiment_settings["num_sanity_val_steps"]

        )
        # if experiment_settings["state_dict_path_load"]:
        #    lightning_object.load_state_dict(torch.load(experiment_settings["state_dict_path_load"]))
        # Fit model
        if experiment_settings["resume_training"]:
            print("resume training from state in checkpoint, (loading weights lr ,epoch nr etc)")
            trainer.fit(lightning_object, ckpt_path=experiment_settings["resume_training"])
        else:
            trainer.fit(lightning_object)


if __name__ == "__main__":
    """
    Given one or more config-files that defines a training,
    Trains a model on a dataset.


    """
    usage_example = "example usage: \n " + "--model rasnet --dataset CIFAR10 --epochs 100 --lr 0.01 --load_model path/to/dict.pth"
    # Initialize parser
    parser = argparse.ArgumentParser(
        epilog=usage_example,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-c", "--config", help="path to config file ", nargs='+', required=True)
    parser.add_argument("--no_logger", help="not using logger avaids this bug  ", default=False, action='store_true')

    args = parser.parse_args()
    train(args)
