import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from pathlib import Path
import os
#
from . import dataset_setup
###############################################################################################
print('\n==> Using fashion data')
data_file_root =  Path( dataset_setup.get_dataset_data_path() ) / f'DATASET_DATA/fashion'
print('==> dataset located at: ', data_file_root)
num_of_classes = 10

device = dataset_setup.device
loss_metric = torch.nn.CrossEntropyLoss()
###############################################################################################
#
transformation = T.Compose([

                            T.ToTensor(),
                            T.Normalize([0.2860], [0.3530]),
                                                    ])          

def get_all_dataset(seed = None):
    dataset = torchvision.datasets.FashionMNIST(
                                        root = data_file_root,
                                        train = True,
                                        transform = transformation,
                                        download=True,
                                        )

    if seed is not None:
        dataset_train, dataset_val = random_split(
                                                    dataset, 
                                                    [len(dataset) - 10, 10],
                                                    generator=torch.Generator().manual_seed(seed)
                                                )
    else:
        dataset_train, dataset_val = random_split(dataset, [len(dataset) - 10, 10])
    dataset_test = torchvision.datasets.FashionMNIST(
                                            data_file_root,
                                            train = False,
                                            download=  True,
                                            transform = transformation,
                                            )       
    return dataset_train, dataset_val, dataset_test


def get_all(batchsize_train = 128, seed = None):
    dataset_train, dataset_val, dataset_test = get_all_dataset(seed = seed)

    #training loader
    dataloader_train = DataLoader(
                                dataset = dataset_train,
                                batch_size = batchsize_train,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = True,
                                )

    #validation loader
    dataloader_val = DataLoader(
                                dataset = dataset_val,
                                batch_size = 512,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )
    #testing loader
    dataloader_test = DataLoader(
                                dataset = dataset_test,
                                batch_size = 512,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )

    return (dataset_train, dataset_val, dataset_test), (dataloader_train, dataloader_val, dataloader_test)
    

'''model setup'''
##################################################################################################
class model(nn.Module):

    def __init__(self, num_of_classes):
        super().__init__()  
        self.num_of_classes = num_of_classes
        self.block_1 = nn.Sequential(
                                    nn.Conv2d(1, 16, kernel_size = 5, stride = 1),
                                    nn.ELU(False),
                                    nn.GroupNorm(4, 16, affine = False), 

                                    nn.Conv2d(16, 16, kernel_size = 5, stride = 1),
                                    nn.ELU(False),
                                    nn.GroupNorm(4, 16, affine = False), 

                                    nn.Conv2d(16, 16, kernel_size = 5, stride = 1),
                                    nn.ELU(False),
                                    nn.GroupNorm(4, 16, affine = False), 

                                    nn.AdaptiveAvgPool2d((4, 4)),
                                    nn.Flatten(),
                                    nn.Linear(16 * 16, 32),
                                    nn.ELU(False),
                                    nn.Linear(32, self.num_of_classes),
                                    )

        dataset_setup.init_model_para(self.block_1)

    def forward(self, x):
        return self.block_1(x)
    
    def set_store_mean_and_var(self, store_mean_and_var):
        pass
##################################################################################################
