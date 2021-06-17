import sys
# append the path to src
sys.path.append('/Users/lee/Downloads/Renjue/Machine-Learning-Operations/02_code_organisation/final_exercise/cookiecutter project/src')

from make_dataset import main, set
from models.model import MyAwesomeModel

import torch
import torchvision
import pytorch_lightning as pl
import torchdrift

'''
    Load data
'''
# training set and test set
train_set, test_set = set()
# training loader and test loader
train_loader, test_loader = main()

def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)

# ood_datamodule = corruption_function(train_set)











