from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import torch
import os
import sys
import pathlib
import random
from PIL import Image
import numpy as np
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from huggingface_hub import cached_download, hf_hub_url
import json
import rasterio

import albumentations as A


#using torchvision.datasets for training on different datasets

def debug_function(input):
    print("debug function start")
    print(input)
    print("debug function end")
    return input
def save_image_to_disk(input):
    input.save("./testim.jpg")
    return input

def get_dataset(args):
    if args["path_dataset"]:
        path_dataset = args["path_dataset"]
    else:
        path_dataset = os.environ.get("PATH_DATASETS", ".")

    if "mnist" == args["dataset"]:
        return Mnist(batch_size=args["batchsize"],data_dir=path_dataset)
    elif "cifar10" == args["dataset"]:
        return Cifar10(batch_size=args["batchsize"],data_dir=path_dataset)
    elif "scene_parse_150" == args["dataset"]:
        return Scene_parse_150(batch_size=args["batchsize"],data_dir=path_dataset)
    elif "coco" == args["dataset"]:
        return Coco(batch_size=args["batchsize"],data_dir=path_dataset)
    elif "custom_semantic_segmentation" == args["dataset"]:
        return Custom_semantic_segmentation_dataset(path_dataset,args)
    elif "imagenet1k" == args["dataset"]:
        return Imagenet1k(batch_size=args["batchsize"],data_dir=path_dataset)
    else:
        sys.exit("no valid dataset!")


class Semantic_segmentation_pytorch_dataset(torch.utils.data.Dataset):
    """
    A pure pytorch dataset for custom semantic segmentation tasks

    """
    def __init__(self,files,labels,args,shuffle = True,always_apply = False,collect_statistics = False):
        """
        @ arg files: a list of paths to the images in the dataset

        assumes the following folder structure
        a_dataset/images/im_x.tif
        a_dataset/labels/masks/im_x.tif
        """

        self.args =args
        self.files= files

        #if shuffle:
        #    random.shuffle(self.files) #make sure the images are shuffled before training
        #remove the images/filename.png and replace it with labels/masks/filename.png
        self.labels= labels

        # Collecting means and stds from the raw data
        # We can use these in the config file for normalization later
        self.collect_statistics = collect_statistics
        self.collected_means=[]
        self.collected_stds = []




        self.transform = A.Compose(
            [
                A.augmentations.geometric.transforms.PadIfNeeded(min_height=1024, min_width=1024),
                A.augmentations.geometric.transforms.ShiftScaleRotate(p=0.5,always_apply=always_apply),

                #
                A.VerticalFlip(p=0.5,always_apply=always_apply),
                A.Transpose(p=0.5,always_apply=always_apply),
                A.RandomRotate90(p=0.5,always_apply=always_apply),
                A.HorizontalFlip(p=0.5,always_apply=always_apply),
                #expects 1-channel or 3-channel images. A.augmentations.transforms.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5,always_apply=always_apply),
                #must be RGB A.augmentations.transforms.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5),p=0.5,always_apply=always_apply),
                A.augmentations.transforms.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=always_apply, p=0.2),
                A.augmentations.transforms.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,always_apply=always_apply, p=0.5),
                #expects 3-channel images A.augmentations.transforms.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08,always_apply=always_apply, p=0.5),
                A.augmentations.transforms.Downscale(scale_min=0.9, scale_max=0.9,p=0.1,always_apply=always_apply),

                A.augmentations.transforms.RandomGamma (gamma_limit=(80, 120), eps=None,always_apply=always_apply, p=0.5),
                A.augmentations.transforms.RandomGridShuffle(grid=(3, 3),always_apply=always_apply, p=0.2),
                A.augmentations.transforms.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5,always_apply=always_apply, p=0.5),
                #expects 3-channel images A.augmentations.transforms.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20,always_apply=always_apply, p=0.5),
                #cannot reshape array of size 3145728 into shape (1024,1024,4) A.augmentations.transforms.JpegCompression(quality_lower=99, quality_upper=100,always_apply=always_apply, p=0.5),



                A.augmentations.transforms.GaussNoise(mean=0, per_channel=True,
                                                      always_apply=always_apply, p=0.5),

                A.augmentations.transforms.Normalize(mean=self.args["means"], std=self.args["stds"],max_pixel_value=255.0, always_apply=False, p=1.0),





                #Notes regarding albumetations transforms#
                #when sending float32 data to albumentations it treat it differently than if its uint8.
                #for float32 it asumes that data is zero normalized.
                #When working with float32 data we therefore need to normalize the data before aplying some of the transforms
                #If data is uint8 we should do the normalization in the end instead


            ])

    def open_data(self,path):
        """
        Open all differetn datasources and stack them ontop of each other to a single multichannel image
        """
        data_sources = self.args["data_sources"]
        #print("data_sources:"+str(data_sources))

        for index,data_source in enumerate(data_sources):
            new_as_array = rasterio.open(path.parent.parent / pathlib.Path(data_source) / path.name).read()
            new_as_array = new_as_array[self.args["channels"][index]]
            if index ==0:
                as_array = new_as_array
            else:
                as_array = np.vstack((as_array, new_as_array))



        dtypes= {"uint8":np.uint8,"float32":np.float32}
        as_array = np.array(as_array,dtype= dtypes[self.args["convert_input_data_to"]])

        if self.collect_statistics:
            #The mean and std values wil be used in the normalize transform function.
            #The normalize function will multiply mean and std with 255, so we now need to divide the values by 255 in order to be able to use the sdt and mean values in the normalize transform
            self.collected_means.append((as_array/255.).reshape([as_array.shape[0],-1]).mean(axis=-1))
            self.collected_stds.append((as_array/255.).reshape([as_array.shape[0], -1]).std(axis=-1))
            print("statistical means")
            print(((np.array(self.collected_means))).mean(axis=0))
            print("statistical stds")
            print(((np.array(self.collected_stds))).mean(axis=0))


        return as_array

    def open_label(self,path):
        return Image.open(path)

    def __getitem__(self, i):
        file = self.files[i]
        img = self.open_data(file)
        # albumetation asume dimensions [ y ,x,channel]
        img = img.transpose([1, 2, 0])
        if not self.labels:
            #if we dont have any labels we still apply the transforms
            transformed = self.transform(image=img)
            img = (transformed["image"])
            # after aplying the transform we need to turn it back into [channel, y,x] format
            img = img.transpose([2, 0, 1])
            img= np.array(img)
            #by returning a pair instead of just the image we can use __getitem on dataset that contains labels AND dataset that does not contain labels , in the same way
            return (img,None)
        else:
            label_file = self.labels[i]
            label =self.open_label(label_file)
            transformed= self.transform(image=img ,mask= np.array(label,dtype=np.int64))
            (img, label) = (transformed["image"],transformed["mask"])
            # after aplying the transform we need to turn it back into [channel, y,x] format
            img = img.transpose([2,0, 1])
            (img, label)=(np.array(img),np.array(label))
            #cross entropy loss wants a int64 as input
            label = np.array(label,dtype=np.int64)
            return (img,label)

    def __len__(self): return len(self.files)

class Custom_semantic_segmentation_dataset():
    """
    Asumes the following data structure
    dataset_folder/images/im_x.tif
    dataset_folder/labels/masks/im_x.tif
    dataset_folder/all.txt  #txt file with one filename e.g im1.tif per row
    dataset_folder/valid.txt #txt file with one filename e.g im1.tif per row
    dataset_folder/train.txt #txt file with one filename e.g im1.tif per row
    """
    def __init__(self,data_dir,args):
        self.batch_size = args["batchsize"]
        label_dir = args["path_labels"]
        data_sources = args["data_sources"]
        all_txt =args["all_txt"]
        valid_txt = args["valid_txt"]
        train_txt = args["train_txt"]
        self.args= args

        self.data_dir =pathlib.Path(data_dir)
        self.n_classes = 12

        self.image_paths_all = self.get_paths(self.data_dir,all_txt)
        self.image_paths_valid = self.get_paths(self.data_dir,valid_txt)
        self.image_paths_train = self.get_paths(self.data_dir,train_txt)
        if label_dir:
            self.label_paths_train= self.get_label_paths(label_dir,self.image_paths_train)
            self.label_paths_valid = self.get_label_paths(label_dir, self.image_paths_valid)
            self.label_paths_all = self.get_label_paths(label_dir, self.image_paths_all)
        else:
            self.label_paths_train = False
            self.label_paths_valid = False
            self.label_paths_all = False




    def get_paths(self,folder_path,txt_file):
        """
        arg path : pathlib.Path
        txt_file : all.txt valid.txt etc with one filename per row
        """

        names = pathlib.Path(txt_file).read_text().split('\n')
        paths = [folder_path/pathlib.Path("rgb")/pathlib.Path(name) for name in names if name.strip() != '']
        return paths

    def get_label_paths(self,label_folder,files):

        paths = [pathlib.Path(label_folder) / pathlib.Path(data_path).name for data_path in files]
        return paths
    def get_data(path,config):
        return





    def prepare_data(self):
        """
        in pytorchlightning downloading should be done in the 'prepare_data' function.
        This function is meant to be callable from a pytorch lightning 'prepare_data' function
        """
        pass

    def setup(self, stage=None):
        """
        In pytorch lightning the 'setup' function is run on each gpu process
        In the official MNIST pytorchlightning example thecode below is in the setup function, not clear to me why it couldnd be in the prepare_data function but I go along with the official recomendations
        """


        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.dataset_train= Semantic_segmentation_pytorch_dataset(files=self.image_paths_train,labels=self.label_paths_train,args=self.args)
            self.dataset_val = Semantic_segmentation_pytorch_dataset(files=self.image_paths_valid,labels=self.label_paths_valid,args=self.args)
            self.dataset_all = Semantic_segmentation_pytorch_dataset(files=self.image_paths_all,labels=self.label_paths_all,args=self.args)


    def train_dataloader(self):
        """
        function for getting dataloader to the training set
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=3)

    def val_dataloader(self):
        """
        function for getting dataloader to the validation set
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=3)
    def all_dataloader(self):
        """
        function for getting dataloader to the all.txt set
        """
        return DataLoader(self.dataset_all, batch_size=self.batch_size, num_workers=3)

class Scene_parse_150():
    """
    """
    def __init__(self,batch_size,data_dir):
        self.data_dir =data_dir
        ds = load_dataset("scene_parse_150", split="train[:50]",streaming=False,use_auth_token=True,cache_dir=self.data_dir)
        #Split the dataset’s train split into a train and test set with the train_test_split method:
        ds = ds.train_test_split(test_size=0.2)
        self.train_ds = ds["train"]
        self.test_ds = ds["test"]

        #setting up id to class name mapping


        repo_id = "huggingface/label-files"
        filename = "ade20k-hf-doc-builder.json"
        id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        num_labels = len(id2label)



class Coco():
    """

    """
    def __init__(self,batch_size,data_dir):
        self.batch_size = batch_size
        self.data_dir =data_dir
        self.coco_anotation = "/home/rasmus/data/annotations_trainval2017/annotations/instances_train2017.json" #why does this not work? looks like I need to point to an existing file?
        self.n_classes = 80 #https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/


    def prepare_data(self):
        """
        in pytorchlightning downloading should be done in the 'prepare_data' function.
        This function is meant to be callable from a pytorch lightning 'prepare_data' function
        """
        torchvision.datasets.CocoDetection(self.data_dir,self.coco_anotation)

    def setup(self, stage=None):
        """
        In pytorch lightning the 'setup' function is run on each gpu process
        In the official MNIST pytorchlightning example thecode below is in the setup function, not clear to me why it couldnd be in the prepare_data function but I go along with the official recomendations
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full = torchvision.datasets.CocoDetection(self.data_dir,self.coco_anotation)
            print("TODO DOWNLOAD BOTH TRAIN AND VLAIDATION AND TEST!!")
            self.dataset_train, self.dataset_val = (full,full)#random_split(full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = torchvision.datasets.CocoDetection(self.data_dir,self.coco_anotation)
    def train_dataloader(self):
        """
        function for getting dataloader to the training set
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=3)

    def val_dataloader(self):
        """
        function for getting dataloader to the validation set
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        """
        function for getting dataloader to the test set
        """
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=3)

class Cifar10():
    """
    Create dataloaders and preprocessing for cifar10
    We resize data to 224x224 to be compatible with Imagenet
    Making mnist easy to use from pytorch lightning by defining some functions used by pytorchlightning
    """
    def __init__(self,batch_size,data_dir):
        self.batch_size = batch_size
        
        self.transform_data = transforms.Compose(
            [
                #debug_function,
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ]
        )
        
        self.data_dir=data_dir
        #input and output
        self.classes =  ('airplanes','cars','birds','cats','deer','dogs','frogs','horses','ships','trucks')
        #self.data_dims = (1, 28, 28)
        self.n_classes = len(self.classes)

        self.setup()
        print("num batches per epoch:"+str(len(self.dataset_train)/self.batch_size))



    def prepare_data(self):
        """
        in pytorchlightning downloading should be done in the 'prepare_data' function.
        This function is meant to be callable from a pytorch lightning 'prepare_data' function
        """
        torchvision.datasets.CIFAR10(root=self.data_dir, train= True, transform= self.transform_data , target_transform= None, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train= False, transform= self.transform_data ,target_transform= None, download=True)
    def setup(self, stage=None):
        """
        In pytorch lightning the 'setup' function is run on each gpu process
        In the official MNIST pytorchlightning example thecode below is in the setup function, not clear to me why it couldnd be in the prepare_data functio>
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = torchvision.datasets.CIFAR10(root=self.data_dir, train= True, transform= self.transform_data , target_transform=None, download=True)
            self.dataset_train, self.dataset_val = random_split(mnist_full, [45000, 5000])
            print(self.dataset_train[0])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = torchvision.datasets.CIFAR10(root=self.data_dir, train= False, transform= self.transform_data , target_transform=None, download=True)
    def train_dataloader(self):
        """
        function for getting dataloader to the training set
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=3)

    def val_dataloader(self):
        """
        function for getting dataloader to the validation set
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        """
        function for getting dataloader to the test set
        """
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=3)


class Mnist():
    """
    Create dataloaders and preprocessing for mnist
    We resize data to 32x32 to be compatible with cifar10
    Making mnist easy to use from pytorch lightning by defining some functions used by pytorchlightning
    """
    def __init__(self,batch_size,data_dir):
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.data_dir=data_dir
        #input and output
        self.classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')
        #self.data_dims = (1, 28, 28)
        self.n_classes = len(self.classes)

        self.setup()
        print("num batches per epoch:"+str(len(self.dataset_train)/self.batch_size))


    def prepare_data(self):
        """
        in pytorchlightning downloading should be done in the 'prepare_data' function.
        This function is meant to be callable from a pytorch lightning 'prepare_data' function
        """
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    def setup(self, stage=None):
        """
        In pytorch lightning the 'setup' function is run on each gpu process
        In the official MNIST pytorchlightning example thecode below is in the setup function, not clear to me why it couldnd be in the prepare_data function but I go along with the official recomendations
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform,download=True)
            self.dataset_train, self.dataset_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = MNIST(self.data_dir, train=False, transform=self.transform)
    def train_dataloader(self):
        """
        function for getting dataloader to the training set
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=3)

    def val_dataloader(self):
        """
        function for getting dataloader to the validation set
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        """
        function for getting dataloader to the test set
        """
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=3)
class Pytorch_dataset_huggingface_imagenet_wrapper(torch.utils.data.Dataset):
    """hugingfacewraper dataset."""

    def __init__(self,transform=None,split = "train",data_dir=False):
        """
        """
        self.transform = transform
        self.data_dir = data_dir
        self.dataset=load_dataset('imagenet-1k', split=split, streaming=False,use_auth_token=True,cache_dir=self.data_dir) #.with_format("torch")

        # Assign test dataset for use in dataloader(s)
        #if stage == "test" or stage is None:
        #    self.dataset_test = load_dataset('imagenet-1k', split='test', streaming=False,use_auth_token=True)



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #print(self.dataset[idx])
        data = self.dataset[idx]["image"]
        label =  self.dataset[idx]["label"]
        if self.transform:
            #print("before transform:"+str(np.array(data).shape))
            data = self.transform(data)
            #print("after transform:"+str(data))
        return (data,label)
class Imagenet1k:

    def __init__(self,batch_size,data_dir):
        self.batch_size = batch_size

        self.transform_data = transforms.Compose([
            transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #save_image_to_disk,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.validation_transform_data = transforms.Compose([
            transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),
            torchvision.transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #turn an a list of images into resized and normalised tensors 
        #self.transform_data = transforms.Compose(
        #    [
        #        #transforms.ToTensor(),
        #        transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),

        #        transforms.Resize((224, 224)),

        #        transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #    ]
        #)


        self.data_dir=data_dir
        #input and output
        self.classes =  list(range(1000))
        #self.data_dims = (1, 28, 28)
        self.n_classes = len(self.classes)

        self.setup()
        print("num batches per epoch:"+str(len(self.dataset_train)/self.batch_size))

        #from transformers import ViTFeatureExtractor

        #model_name_or_path = 'google/vit-base-patch16-224-in21k'
        #self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

    def not_used_transform(self,example_batch):
        #print(example_batch)
        # Take a list of PIL images and turn them to pixel values
        #inputs = self.feature_extractor([x for x in example_batch['image']], return_tensors='pt')

        # Don't forget to include the labels!
        #inputs['labels'] = example_batch['label']
        #print(example_batch["image"])
        #print(type(example_batch["image"]))
        example_batch["image"]= [self.transform_data(x) for x in example_batch["image"]]
        return example_batch #(example_batch["image"],example_batch["label"]) #inputs
    def old_transform(self,example_batch):
        print(example_batch)
        example_batch["image"] = self.transform_data(example_batch["image"])
        return example_batch
    #def transform(self,example_batch):
    #    example_batch["image"] = self.transform_data(x) for x in example_batch['image']])
    # 
    #         # Don't forget to include the labels!
    #        inputs['labels'] = example_batch['labels']
    #        return inputs


    def prepare_data(self):
        """
        in pytorchlightning downloading should be done in the 'prepare_data' function.
        This function is meant to be callable from a pytorch lightning 'prepare_data' function
        """
        load_dataset('imagenet-1k', split='train', streaming=False,use_auth_token=True,cache_dir=self.data_dir)
        load_dataset('imagenet-1k', split='validation', streaming=False,use_auth_token=True,cache_dir=self.data_dir)
        load_dataset('imagenet-1k', split='test', streaming=False,use_auth_token=True,cache_dir=self.data_dir)

     

    def setup(self, stage=None):
        """
        In pytorch lightning the 'setup' function is run on each gpu process
        In the official MNIST pytorchlightning example thecode below is in the setup function, not clear to me why it couldnd be in the prepare_data functio>
        """
        #On how to transform to pytorch dataset format : ds.with_format("torch")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            #self.dataset_train, self.dataset_val = (load_dataset('imagenet-1k', split='train', streaming=False,use_auth_token=True).with_format("torch").with_transform(self.transform),load_dataset('imagenet-1k', split='validation', streaming=False,use_auth_token=True).with_format("torch").with_transform(self.transform))
            #I should set up cache_dir= self.data_dir
            self.dataset_train, self.dataset_val = (Pytorch_dataset_huggingface_imagenet_wrapper(split="train",transform=self.transform_data),Pytorch_dataset_huggingface_imagenet_wrapper(split="validation",transform=self.validation_transform_data,data_dir=self.data_dir))
            #input(self.dataset_train[0])

        # Assign test dataset for use in dataloader(s)
        #if stage == "test" or stage is None:
        #    self.dataset_test = load_dataset('imagenet-1k', split='test', streaming=False,use_auth_token=True)
    def train_dataloader(self):
        """
        function for getting dataloader to the training set
        """
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=3)

    def val_dataloader(self):
        """
        function for getting dataloader to the validation set
        """
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        """
        function for getting dataloader to the test set
        """
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=3)
