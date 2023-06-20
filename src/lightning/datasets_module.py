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



#using torchvision.datasets for training on different datasets

def debug_function(input):
    print("debug function is running")
    print(input)
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
        return Custom_semantic_segmentation_dataset(batch_size=args["batchsize"],data_dir=path_dataset,label_dir=args["path_labels"],data_sources=args["data_sources"],all_txt=args["all_txt"],valid_txt=args["valid_txt"],train_txt=args["train_txt"] )
    elif "imagenet1k" == args["dataset"]:
        return Imagenet1k(batch_size=args["batchsize"],data_dir=path_dataset)
    else:
        sys.exit("no valid dataset!")


class Semantic_segmentation_pytorch_dataset(torch.utils.data.Dataset):
    """
    A pure pytorch dataset for custom semantic segmentation tasks

    """
    def __init__(self,files,labels,args,shuffle = True,debug = False):
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


        #some models asume data to be divisible by 32 (best handeled by making shure the dataset is in corect format form the beguinning or using padding)
        #for simplicity we use a resize but this is not optimal
        #REPLACE WITH padifneeded from here https://albumentations.ai/docs/examples/example_kaggle_salt/
        self.transform_data = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.transform_label = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                
            ]
        )
    def open_data(self,path):
        data_sources = args["data_sources"]
        input(data_sources)
        input(json.loads(data_sources))
        return Image.open(path)
    def open_label(self,path):
        return Image.open(path)




    def __getitem__(self, i):
        file = self.files[i]
        label_file = self.labels[i]

        img =self.open_data(file)
        label =self.open_label(label_file)


        #img = torch.Tensor(np.array(img)).permute(2,0,1)

        #label = torch.tensor(np.array(label,dtype=int))

        if self.transform_data:
            (img,label) = (self.transform_data(img),np.array(self.transform_label(label),dtype=np.int64))


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
    def __init__(self,batch_size,data_dir,label_dir,data_sources,all_txt,valid_txt,train_txt):
        self.batch_size = batch_size
        self.data_dir =pathlib.Path(data_dir)
        self.n_classes = 12

        self.image_paths_all = self.get_paths(self.data_dir,all_txt)
        self.image_paths_valid = self.get_paths(self.data_dir,valid_txt)
        self.image_paths_train = self.get_paths(self.data_dir,train_txt)
        self.label_paths_train= self.get_label_paths(label_dir,self.image_paths_train)
        self.label_paths_valid = self.get_label_paths(label_dir, self.image_paths_valid)



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

            self.dataset_train= Semantic_segmentation_pytorch_dataset(files=self.image_paths_train,labels=self.label_paths_train)
            self.dataset_val = Semantic_segmentation_pytorch_dataset(files=self.image_paths_valid,labels=self.label_paths_valid)


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
