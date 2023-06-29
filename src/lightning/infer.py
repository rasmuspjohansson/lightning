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
        dataset = datasets_module.get_dataset(experiment_settings)

        # It is important to verify that the iamges look good after augmentations (no artifacts or anything else that causes the data to look very differetn than the un-augmetnted dat)

        if experiment_settings["nr_of_images_to_visualize"] > 0:
            visualize.visualize_dataset(dataset, experiment_settings)

        model = models.get_model(experiment_settings, dataset.n_classes)
        lightning_object = lightning_module.Lightning_module(dataset=dataset, model=model, args=experiment_settings)

        if experiment_settings["state_dict_path_load"]:
            print("loading the weights from a checkpoint (no lr or other meta parameters)")
            #lightning_object.load_state_dict(torch.load(experiment_settings["state_dict_path_load"])["state_dict"])
            lightning_object.load_from_checkpoint(experiment_settings["state_dict_path_load"])
            print("loaded the state dict")

        # callbacks=[TQDMProgressBar(refresh_rate=20),lr_monitor,checkpoint_callback],
        trainer = Trainer(devices="auto")
        dataset.setup()
        result=trainer.predict(lightning_object, dataset.all_dataloader())




if __name__ == "__main__":
    """
    Given one or more config-files that defines a inference,
    infers a model on a dataset (classifies all data in teh dataset).
    """
    usage_example = "example usage: \n " + "--model rasnet --dataset CIFAR10 --epochs 100 --lr 0.01 --load_model path/to/dict.pth"
    # Initialize parser
    parser = argparse.ArgumentParser(
        epilog=usage_example,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-c", "--config", help="path to config file ", nargs='+', required=True)

    args = parser.parse_args()
    infer(args)
