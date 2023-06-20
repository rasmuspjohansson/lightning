from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
import torchvision
import segmentation_models_pytorch as smp
import sys





class Print(nn.Module):
    """
    A class for debugging layers that are packaged with sequential
    """
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print("debug")
        print(x.shape)
        return x

def get_model(args,n_classes):

    if args["model"] == "lenet5":
        return LeNet5(n_classes)
    elif args["model"] == "locally_conected_mlp":
        return LocallyConnectedMLP(n_classes)
    elif args["model"] == "fully_conected_mlp":
        return FullyConnectedMLP(n_classes)
    elif args["model"] == "LeNet5_with_locally_conected_classifier":
        return LeNet5_with_locally_conected_classifier(n_classes)
    elif args["model"] == "LeNet5_without_convolution":
        return LeNet5_without_convolution(n_classes)
    elif args["model"] == "vit":
        return Vit(n_classes)
    elif args["model"] == "vit_local":
        return Vit_local(n_classes)
    elif args["model"] == "fcn":
        return Fcn_mobilenet_v3(n_classes)
    elif args["model"] == "unet":
        return Unet(n_classes,args["n_input_channels"])
    elif args["model"] == "unet":
        return Fcn_mobilenet_v3(n_classes)
    else:
        sys.exit("no known model defined!")
class Unet(nn.Module):
    """
    a thin wrapper around segmentation models version of unet
    https://github.com/qubvel/segmentation_models.pytorch
    """
    def __init__(self,n_classes,n_input_channels):
        super(Unet, self).__init__()
        self.input_channels = n_input_channels
        self.model = smp.Unet('resnet34', encoder_weights='imagenet',classes=n_classes,in_channels=self.input_channels)
    def forward(self, x):
        x = self.model.forward(x)
        return x

class Vit(nn.Module):
    """
    A basic Vision transformer
    a thin wrapper around a torchvision vit model

    meant to be used as reference when experimenting with Vit models with locally conected layers in their EncoderBlock
    vit_b_32 is used becaus 'how to train your vit ?' recomends  usig large models with patchsize 32 instead of smaller models with smaller patch sizes

    """
    def __init__(self,n_classes):
        super(Vit, self).__init__()
        self.model = torchvision.models.vit_b_32(num_classes=n_classes)
    def forward(self, x):
        #print("vit input 1")
        #print(type(x))
        #print(x)
        #print("vit input 2")

        x = self.model.forward(x)
        return x


class Vit_local(nn.Module):
    """
    A Vision transformer modified to use locally conected perceptrons in its MLP 
    a thin wrapper around a torchvision vit model that is modified in this way

    meant to be used to experiment with Vit models with locally conected layers in their EncoderBlock
    vit_b_32 is used becaus 'how to train your vit ?' recomends  usig large models with patchsize 32 instead of smaller models with smaller patch sizes

    """
    def __init__(self,n_classes):
        super(Vit_local, self).__init__()
        self.model = local_vision_transformer.vit_b_32(num_classes=n_classes)
    def forward(self, x):
        #print("vit input 1")
        #print(type(x))
        #print(x)
        #print("vit input 2")

        x = self.model.forward(x)
        return x


class Fcn_mobilenet_v3(nn.Module):
    """
    a thin wrapper on torchvisions fully convolutional model with mobile net 3 as backbone
    """
    def __init__(self,n_classes):
        super(Fcn_mobilenet_v3, self).__init__()
        self.model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=False, progress=True, num_classes=n_classes)
    def forward(self, x):
        x = self.model.forward(x)
        return x


#pytorch implementation of the classic LeNet5 (slightly modified from https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320)
class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.log_softmax (logits, dim=1) #we use F.nll_loss wich works well with log_softmax # https://discuss.pytorch.org/t/does-nllloss-handle-log-softmax-and-softmax-in-the-same-way/8835/2

        return probs
class LeNet5_without_convolution(nn.Module):
    """
    replacing all conv2d layers locallyconected versions without weight sharing
    """
    def __init__(self, n_classes):
        super(LeNet5_without_convolution, self).__init__()
        input_shape=(1,32,32)
        verbose = False

        self.feature_extractor = nn.Sequential(
            #Print(),
            locally_connected.Locally_connected2D(in_shape=input_shape, out_channels=6, receptive_field_shape=(5, 5), stride=1),
            #the shape should now be (32 -5)+1 = 28
            #nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            # the shape should now be 28/2 = 14

            locally_connected.Locally_connected2D(in_shape=(6,14,14), out_channels=16, receptive_field_shape=(5, 5), stride=1),
            #Print(),
            # the shape should now be (14 -5)+1 = 10
            #nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            # the shape should now be 10/2 = 5

            #nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            locally_connected.Locally_connected2D(in_shape=(16, 5, 5), out_channels=120, receptive_field_shape=(5, 5), stride=1),
            #Print(),
            #the shape should now be (5-5)+1 = 1
            #input_shape = (120,1,1)



            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            locally_connected.LocallyConnected1D(in_shape=120, out_channels=10, receptive_field_shape=12,
                                                 stride=12),
            nn.Flatten(1),

            nn.Tanh(),
            nn.Linear(in_features=100, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        #x.shape is now (batchsize,120,1,1)
        x = torch.flatten(x, 1)
        # x.shape is now (batchsize,120)
        logits = self.classifier(x)
        probs = F.log_softmax (logits, dim=1) #we use F.nll_loss wich works well with log_softmax # https://discuss.pytorch.org/t/does-nllloss-handle-log-softmax-and-softmax-in-the-same-way/8835/2

        return probs

class LeNet5_with_locally_conected_classifier(nn.Module):
    """
    The last fully connected layer (before the outputlayer) is replaced with a locally conected layer with only 1/10th as many conections.
    This makes the model train faster.
    """

    def __init__(self, n_classes):
        super(LeNet5_with_locally_conected_classifier, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            locally_connected.LocallyConnected1D(in_shape=(120), out_channels=10, receptive_field_shape=12,
                                                 stride=12),
            nn.Flatten(1),

            nn.Tanh(),
            nn.Linear(in_features=100, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.log_softmax (logits, dim=1) #we use F.nll_loss wich works well with log_softmax # https://discuss.pytorch.org/t/does-nllloss-handle-log-softmax-and-softmax-in-the-same-way/8835/2

        return probs


class LocallyConnectedMLP(nn.Module):
    def __init__(self, n_classes):
        super(LocallyConnectedMLP, self).__init__()

        #a fully conected layer conects 1000 units to all inputs
        #a locally conected layer then conects 10 different fcn (with 100 units each) to 10 different parts of the input
        #the locally conected layer that way also has 1000 outputs (10 *100) but only a 1/10th of the number of connections

        self.classifier = nn.Sequential(

            nn.Linear(in_features=32*32, out_features=1000),
            nn.Tanh(),
            locally_connected.LocallyConnected1D(in_shape=(1000), out_channels=100,receptive_field_shape = 100, stride = 100),
            nn.Tanh(),
            nn.Linear(in_features=1000, out_features=n_classes),
        )
    def forward(self, x):
        #flatten from dimension 1 (not the batch dimension 0))
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.log_softmax (logits, dim=1) #we use F.nll_loss wich works well with log_softmax # https://discuss.pytorch.org/t/does-nllloss-handle-log-softmax-and-softmax-in-the-same-way/8835/2

        return probs
class FullyConnectedMLP(nn.Module):
    def __init__(self, n_classes):
        super(FullyConnectedMLP, self).__init__()
        #a series of fully conected layers
        self.classifier = nn.Sequential(

            nn.Linear(in_features=32*32, out_features=1000),
            nn.Tanh(),
            nn.Linear(in_features=1000, out_features=1000),
            nn.Tanh(),
            nn.Linear(in_features=1000, out_features=n_classes),
        )
    def forward(self, x):
        #flatten from dimension 1 (not the batch dimension 0))
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.log_softmax (logits, dim=1) #we use F.nll_loss wich works well with log_softmax # https://discuss.pytorch.org/t/does-nllloss-handle-log-softmax-and-softmax-in-the-same-way/8835/2
        return probs
