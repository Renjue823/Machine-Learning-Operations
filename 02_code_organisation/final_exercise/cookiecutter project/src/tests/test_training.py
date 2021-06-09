import sys
import argparse

from torch.nn.modules.container import ModuleList
sys.path.append('/Users/lee/Downloads/Renjue/Machine-Learning-Operations/02_code_organisation/final_exercise/cookiecutter project/src')
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import MyAwesomeModel
from models.main import TrainOREvaluate
from data.make_dataset import main

model = MyAwesomeModel(10)
training  = TrainOREvaluate()
print(training)

