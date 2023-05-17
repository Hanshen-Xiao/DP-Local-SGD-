import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from PIL import Image
#
from . import dataset_setup, wrn
import datasets.RESNETS as resnet_colection
###########################################################################################
print('\n==> Using gtsrb data')
data_file_root =  Path( dataset_setup.get_dataset_data_path() ) / f'DATASET_DATA/gtsrb'
print('==> dataset located at: ', data_file_root)
num_of_classes = 43

device = dataset_setup.device
loss_metric = torch.nn.CrossEntropyLoss()
###########################################################################################
T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
transformation = T.Compose([
                            
                            # T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                            # T.RandomCrop(size=(32, 32), padding=4),
                            T.Resize((32, 32)),
                            T.RandomHorizontalFlip(),  
                            # T.RandomRotation(degrees=(-10, 10),),
                            
                            T.ToTensor(),
                            T_normalize,
                                    
                            # T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                            # T.RandomPerspective(distortion_scale=0.5, p=1.0),
                            # T.RandomHorizontalFlip(),
                            ])


def get_all_dataset(seed = None):
    dataset = torchvision.datasets.GTSRB(
                                    root = data_file_root,
                                    split = 'train',
                                    download = True,
                                    transform = transformation,
                                    )
    
    if seed is not None:
        dataset_train, dataset_val = random_split(
                                                    dataset, 
                                                    [len(dataset) - 0, 0],
                                                    generator=torch.Generator().manual_seed(seed)
                                                )
    else:
        dataset_train, dataset_val = random_split(dataset, [len(dataset) - 1, 1])
        
    dataset_test = torchvision.datasets.GTSRB(
                                            data_file_root,
                                            split = 'test',
                                            download=  True,
                                            transform = T.Compose([
                                                                T.Resize((32, 32)),
                                                                T.ToTensor(),
                                                                T_normalize,
                                                                ]),
                                            
                                            )   
    
    # dataset_train.__getitem__ = __getitem___special.__get__(dataset_train) 
    return dataset_train, dataset_val, dataset_test


def get_all(batchsize_train = 128, seed = None,):
    dataset_train, dataset_val, dataset_test = get_all_dataset(seed = seed)

    # training loader
    dataloader_train = DataLoader(
                                dataset = dataset_train,
                                batch_size = batchsize_train,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )

    # # validation loader
    # dataloader_val = DataLoader(
    #                             dataset = dataset_val,
    #                             batch_size = 512,
    #                             shuffle = True,
    #                             num_workers = 4,
    #                             pin_memory = (device.type == 'cuda'),
    #                             drop_last = False,
    #                             )
    # testing loader
    dataloader_test = DataLoader(
                                dataset = dataset_test,
                                batch_size = 500,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )

    return (dataset_train, dataset_val, dataset_test), (dataloader_train, None, dataloader_test)
    
'''model setup'''
##################################################################################################
class model(nn.Module):
    
    def __init__(self, num_of_classes):
        super().__init__()  
        self.num_of_classes = num_of_classes
        # self.my_model_block = resnet_colection.resnet20()
        
        self.my_model_block = resnet_colection.resnet20(num_of_classes)
        # self.my_model_block = wrn.WideResNet(
        #                                     depth = 16, 
        #                                     num_classes = num_of_classes, 
        #                                     widen_factor = 4, 
        #                                     dropRate = 0.0,
        #                                     )
    
    
    def forward(self, x):
        return self.my_model_block(x)

##################################################################################################
