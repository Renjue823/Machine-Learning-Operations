import sys
import argparse
sys.path.append('/Users/lee/Downloads/Renjue/Machine-Learning-Operations/02_code_organisation/final_exercise/cookiecutter project/src')
import pytest
import torch

# set is for sets and main is for loaders
from data.make_dataset import set, main

# acquire training set and test set
trainset, testset = set()
# print(len(trainset))
# print(len(testset))
trainloader, testloader = main()
# train_image, train_label = iter(trainloader).next()
# test_image, test_label = iter(testloader).next()
print(trainloader.dataset.data.shape) 
print(trainloader.dataset.targets.shape)

def size(loader):
    if loader == trainloader:
        length = trainloader.dataset.targets.shape
        shape = trainloader.dataset.data.shape
    elif loader == testloader:
        length = testloader.dataset.targets.shape
        shape = testloader.dataset.data.shape
    return length, shape

# make sure to have the assert for length and shape separately
def test_size():
    train_length, train_shape = size(trainloader)
    assert train_length ==  torch.Size([60000]) # 60000 training observations
    assert train_shape == torch.Size([60000, 28, 28])
    test_length, test_shape = size(testloader)
    assert test_length ==  torch.Size([10000]) # 10000 test observations
    assert test_shape == torch.Size([10000, 28, 28]) 





