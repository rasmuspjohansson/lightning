import sys
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torchmetrics import Accuracy
import visualize
import numpy as np
from PIL import Image
import saving_geotif
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR # this seems to be incompatible with pytorch2 right now

class Lightning_module(LightningModule):
    def __init__(self, dataset,model,args):

        super().__init__()
        self.args =args
        self.dataset = dataset
        self.model = model
        #We use torchmetrics accuracy object for keeping track of the accuracies for the different divisions of the dataset(https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=dataset.n_classes,mdmc_average ="global",ignore_index=self.args["ignore_index"])
        self.test_accuracy = Accuracy(task="multiclass", num_classes=dataset.n_classes,mdmc_average="global",ignore_index=self.args["ignore_index"])
        self.train_accuracy = Accuracy(task="multiclass", num_classes=dataset.n_classes,mdmc_average="global",ignore_index=self.args["ignore_index"])
        if self.args["loss"] == "cross_entropy":
           #this loss is used when training semantic segmentation
           self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.args["ignore_index"],label_smoothing=self.args["label_smoothing"])
        elif self.args["loss"] == "nll_loss":
           #this loss is used when training image classification 
           self.loss = F.nll_loss
        else:
           sys.exit("please provide a valid loss")





        """
        on semantic segmentation tasks we want to calculate the intersection over union
        

        self.train_metric = torchmetrics.IoU(
            num_classes=2,
            dist_sync_on_step=True)
        """
        #when logging progress we might want to validate more often than once per epoch. We then need to keep track of how far we have progressed
        self.epochs_done_as_float = 0

    def forward(self, x):
        #print("image shape:"+str(x.shape))
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss=self.loss(logits,y)

        #semantic_segmentation = True
        #if semantic_segmentation:
        #    loss = self.semantic_segmentation_loss(logits, y)
        #else:
        #    loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        if self.args["print_label_and_prediction_histogram"]:
            y_numpy = y.cpu()
            preds_numpy = preds.cpu()
            for i in range(y_numpy.shape[0]):
                print("label histogram::")
                print(np.histogram(y_numpy[i], bins=list(range(self.dataset.n_classes+1))))

                print("prediction histogram::")
                print(np.histogram(preds_numpy[i], bins=list(range(self.dataset.n_classes+1))))


        #compute accuracy on batch and return result.
        #We use forward() instead of update() because we want to know how accuray on the trainingset changes for fractions of epochs
        #forward()  updates the accuracy for the complete dataset (in by calling update() but also returns the accuracy for the current batch/input
        #https://torchmetrics.readthedocs.io/en/stable/pages/overview.html
        accuracy= self.train_accuracy.forward(preds,y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",accuracy , prog_bar=True)#

        #log how many epochs we have finnished (e.g 1.25 epochs) so we can use this as x axis when plotting
        self.epochs_done_as_float = self.current_epoch+ (batch_idx/self.trainer.num_training_batches)
        self.log("epochs_as_float",self.epochs_done_as_float , prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)


        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        #log how many epochs we have finnished (e.g 1.25 epochs) so we can use this as x axis when plotting
        self.log("epochs_as_float",float(self.epochs_done_as_float ), prog_bar=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        https://lightning.ai/docs/pytorch/stable/deploy/production_basic.html
        """
        #Prediction can be done on dataset with or without labels

        try:
            (data,label) = batch
            label = label.cpu()
        except:
            (data,label) = (batch,None)

        result = self(data)
        # move data from gpu to cpu in order to be able to turn it into numpy array
        data = data.cpu()

        result = result.cpu()



        #we also need to get the names for the data in the batch
        filenames_for_batch = self.dataset.image_paths_all[batch_idx*self.dataset.batch_size:(batch_idx+1)*self.dataset.batch_size]


        for i,file_name in enumerate(filenames_for_batch):

            rgb_indexes=[0,1,2]
            normalized_rgb_data = data[i][rgb_indexes]
            raw_input= np.array(Image.open(filenames_for_batch[i]))
            un_normalized_transformed_input = visualize.un_normalize(normalized_rgb_data, means=np.array(self.args["means"])[rgb_indexes], stds=np.array(self.args["stds"])[rgb_indexes]).transpose([1, 2, 0])
            infered_prediction = result[i].argmax(axis=0)
            print(result.shape)
            print(result[i].shape)
            print(infered_prediction.shape)
            print(infered_prediction.max())
            print(infered_prediction.min())
            print("predictions histogram : "+str(np.histogram(infered_prediction, bins=list(range(self.dataset.n_classes+1)))))

            if label is None:
                label_for_im = None
            else:
                label_for_im = label[i]



            if self.args["show_input_and_output"]:
                visualize.visualize_input_output_and_label(raw_input=raw_input,label=label_for_im,un_normalized_transformed_input=un_normalized_transformed_input,infered_prediction=infered_prediction,file_name=file_name)
            saving_geotif.save_output(a_file=filenames_for_batch[i],probs=result[i],experiment_settings_dict=self.args,show=False)









    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
        self.log("epochs_as_float",float(self.epochs_done_as_float) , prog_bar=False)


    def configure_optimizers(self):
        if self.args["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args["lr"],weight_decay=self.args["wd"])
        elif self.args["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.args["lr"],
                                        momentum=0.9)
        else:
            sys.exit("NO VALID OPTIMIZER")


        steps_per_epoch =len(self.dataset.dataset_train)//self.args["batchsize"]
        total_steps = steps_per_epoch*self.args["epochs"]
        print(self.args["learning_rate_schedule"])
        print(self.args["learning_rate_schedule"]["name"])
        if self.args["learning_rate_schedule"] == "fit_one_cykle":
            #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
            #steps per epoch needs to be calculated based on the size of the training set and batch size
            pct_start=0.1

            print("steps_per_epoch:"+str(steps_per_epoch))
            print("total_steps:"+str(total_steps))
            print("should reach maximum Lr at step : "+str(pct_start*total_steps))

            scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr=self.args["lr"], total_steps=self.trainer.estimated_stepping_batches)
            #torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01,total_steps=total_steps , pct_start=pct_start, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False, last_epoch=- 1, verbose=False)
        elif self.args["learning_rate_schedule"] == "exponential":
            print("steps_per_epoch:"+str(steps_per_epoch))
            print("total_steps:"+str(total_steps))

            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                               gamma=99,
                                                               verbose=False)
        elif self.args["learning_rate_schedule"]["name"] == "LinearWarmupCosineAnnealingLR":
            print("WARNING! this lr shedular might be incompatible with pytorch2")
            print("steps_per_epoch:"+str(steps_per_epoch))
            print("total_steps:"+str(total_steps))

            scheduler = {'sheduler':LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args["learning_rate_schedule"]["warmup_epochs"],max_epochs=self.args["epochs"],  warmup_start_lr=0.0, eta_min=0.0, last_epoch=- 1),
            'interval': 'step'
            }

        elif self.args["learning_rate_schedule"]["name"] == "CosineAnnealingLR":
            print("make sure to do a warmup before cosineAnealingLR ! ")
            print("steps_per_epoch:"+str(steps_per_epoch))
            print("total_steps:"+str(total_steps))

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0, last_epoch=- 1, verbose=False)
        elif self.args["learning_rate_schedule"]["name"] =="LinearLR":
            print("linear LR should be posible to use as warmup ")
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0000000000000001, end_factor=1.0, total_iters=total_steps, last_epoch=- 1, verbose=False)


        elif self.args["learning_rate_schedule"] == "NONE":
            return optimizer
        else:
            sys.exit("NO VALID LR_SCHEDULE")

        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : scheduler,
                "monitor" : "train_loss",
                "interval": "step",

            }
        }



    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    """

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        self.dataset.prepare_data()

    def setup(self, stage=None):
        self.dataset.setup(stage=stage)
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        """

    def train_dataloader(self):
        return self.dataset.train_dataloader()
        #return DataLoader(self.mnist_train, batch_size=BATCH_SIZE, num_workers=3)

    def val_dataloader(self):
        return self.dataset.val_dataloader()
        #return DataLoader(self.mnist_val, batch_size=BATCH_SIZE, num_workers=3)

    def test_dataloader(self):
        return self.dataset.test_dataloader()
        #return DataLoader(self.mnist_test, batch_size=BATCH_SIZE, num_workers=3)



