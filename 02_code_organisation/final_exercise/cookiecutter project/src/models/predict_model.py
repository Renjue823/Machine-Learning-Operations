#%%
import sys
# __init__.py is to make src a package but it does not work here
# define the path
sys.path.append('/Users/lee/Downloads/Renjue/Machine-Learning-Operations/02_code_organisation/final_exercise/cookiecutter project/src')
import torch
import matplotlib.pyplot as plt

from model import MyAwesomeModel
from data.make_dataset import main

model = MyAwesomeModel(10)
dict_ = torch.load("model.pth")
model.load_state_dict(dict_)
# test_set contains all images for making predictions
_, test_set = main() 

# select a single picture in test-set
dataiter = iter(test_set)
image, label = dataiter.next()
plt.imshow(image[1].numpy().squeeze(), cmap='Greys_r')
plt.show()
# make prediction with the saved model

print("predicted label of the figure: {}".format(torch.argmax(torch.exp(model(image))[1, :])))
print("true label of the figure: {}".format(label[1]))


# %%
