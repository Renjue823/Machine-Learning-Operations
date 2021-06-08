import sys
# __init__.py is to make src a package but it does not work here
# define the path
sys.path.append('/Users/lee/Downloads/Renjue/Machine-Learning-Operations/02_code_organisation/final_exercise/cookiecutter project/src')
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from data.make_dataset import main
from models.model import MyAwesomeModel

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

model = MyAwesomeModel(10)
dict_ = torch.load("model.pth")
model.load_state_dict(dict_)
# test_set contains all images for making predictions
train_loader, test_loader = main()

images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()


