import sys
import argparse
sys.path.append('/Users/lee/Downloads/Renjue/Machine-Learning-Operations/02_code_organisation/final_exercise/cookiecutter project/src')

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

from data.make_dataset import main
from model import MyAwesomeModel
import numpy as npcd

import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # Implement training loop 

        # implement summary writer
        writer = SummaryWriter()

        model = MyAwesomeModel(10)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_set ,_ = main()
        
        epochs = 20 # should be set to 20
        steps = 0
        train_losses = []
        train_accuracy = 0
        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                steps += 1
                train_losses.append(loss.item()/64)
                train_ps = torch.exp(model(images))
                train_top_p, train_top_class = train_ps.topk(1, dim=1)
                train_equals = train_top_class == labels.view(*train_top_class.shape)
                train_accuracy += torch.mean(train_equals.type(torch.FloatTensor))
                writer.add_scalar('Loss/train', train_losses[-1], steps)
                writer.add_scalar('Accuracy/train', train_accuracy/steps, steps)
            print(f"Training loss: {running_loss/len(train_set)}")

        # plot training curve
        fig = plt.figure()    
        fig.suptitle("training loss VS training steps for mnist data-set", fontsize=14)
        plt.plot(np.arange(1, steps+1, 1), np.array(train_losses))  
        plt.xlabel("training steps", fontsize=14)
        plt.ylabel("training loss", fontsize=14)
        plt.show() 
        # save training curve into reports/figures/
        save_results_to = '/Users/lee/Downloads/Renjue/Machine-Learning-Operations/02_code_organisation/final_exercise/cookiecutter project/reports/figures/'
        fig.savefig(save_results_to + 'training curve.png', dpi = 300)
        torch.save(model.state_dict(), 'model.pth')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="model.pth")
        # add any additional argument that you want
        
        model = MyAwesomeModel(10)
        dict_ = torch.load("model.pth")
        model.load_state_dict(dict_)
        _, test_set = main()
        
        accuracy = 0
        counter = 0
        # turn off gradients for the purpose of speeding up the code
        with torch.no_grad():
            for images, labels in test_set: # with batch size 64
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                counter += 1
                accuracy += torch.mean(equals.type(torch.FloatTensor))
            accuracy = accuracy / counter
            print(f'Accuracy: {accuracy.item()*100}%')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    