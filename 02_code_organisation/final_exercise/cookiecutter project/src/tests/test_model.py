import sys
import argparse

from torch.nn.modules.container import ModuleList
sys.path.append('/Users/lee/Downloads/Renjue/Machine-Learning-Operations/02_code_organisation/final_exercise/cookiecutter project/src')
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import MyAwesomeModel
from data.make_dataset import main

trainloader, testloader = main()
image, label = iter(trainloader).next()

# initialize an untrained model
Model = MyAwesomeModel(10)
# the output of model
y = Model(image) 
# input shape of model
print(image.shape)
# output shape of model
print(y.shape)

def modelShape(input):
    inputShape = input.shape
    modelOutput = Model(input)
    outputShape = modelOutput.shape
    return inputShape, outputShape

def test_modelShape():
    inputShape, outputShape = modelShape(image)
    assert inputShape == torch.Size([64, 1, 28, 28])
    assert outputShape == torch.Size([64, 10])

# with raise
def raise_modelShape():
    inputShape, outputShape = modelShape(image)
    if inputShape != torch.Size([64, 1, 28, 28]): 
        raise ValueError('The input shape of the model should be [64, 1, 28, 28]!')
    if outputShape != torch.Size([64, 10]):
        raise ValueError('The output shape of the model should be [64, 10]!')

def testRaise_model_Shape():
    with pytest.raises(ValueError, match="The input shape of the model should be [64, 1, 28, 28]!"):
        raise_modelShape()
    with pytest.raises(ValueError, match='The output shape of the model should be [64, 10]!'):
        raise_modelShape()



