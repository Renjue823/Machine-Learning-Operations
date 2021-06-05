# 7. Distributed training

Today is about distributed training. We will start simple by making use of `DistributedData` which
is kind of the old standard for doing distributed training, and from there move on to format our
code that will make it agnostic towards device type and distributed setting using 
[Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)

## Distributed data loading

One common bottleneck in training deep learning models is the data loading (you maybe already saw this
during the profiling exercises). The reason

### Exercises

This exercise is intended to be done on the [labeled faces in the wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
dataset. The dataset consist images of famous people extracted from the internet. The dataset had been used
to drive the field of facial verification, which you can read more about 
[here](https://machinelearningmastery.com/introduction-to-deep-learning-for-face-recognition/). We are going
imagine that this dataset cannot fit in memory, and your job is therefore to construct a data pipeline that
can be parallelized based on loading the raw datafiles (.jpg) at runtime.

1. Download the dataset and extract to a folder. It does not matter if you choose the non-aligned or
   aligned version of the dataset.

2. We provide the `lfw_dataset.py` file where we have started the process of defining a data class. 
   Fill out the `__init__`, `__len__` and `__getitem__`. Note that `__getitem__` expect that you
   return a single `img` which should be a [PIL Image](https://pillow.readthedocs.io/en/stable/)
   We want the `img` to be `PIL` image so we can take advantage of 
   [torchvision](https://pytorch.org/vision/stable/transforms.html) for data augmentation.  

3. Make sure that the script runs without any additional arguments
   ```
   python lfw_dataset.py
   ```
4. Visualize a single batch by filling out the codeblock after the first *TODO* after defining the dataloader. 
   The visualization should show when launching the script as
   ```
   python lfw_dataset.py -visualize_batch
   ```

5. Experiment how the number of workers influences the performance. We have already provide code that will
   pass over the dataset 5 times and calculate how long time it took, which you can play around with by calling
   ```
   python lfw_dataset.py -get_timing -num_workers 1
   ```
   Make a [errorbar plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html) with
   number of workers along the x-axis and the timing along the y-axis. The errorbars should correspond to
   the standard deviation over the 5 runs.

6. (Optional, requires access to GPU) If your dataset fits in GPU memory it is beneficial to set the
   `pin_memory` flag to `True`. By setting this flag we are essentially telling Pytorch that they can
   lock the data in-place in memory which will make the transfer between the *host* (CPU) and the
   *device* (GPU) faster.

## Distributed Data

For this exercise we will briefly touch upon how to implement data parallel training in Pytorch using
their [nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) class.

### Exercises

1. Wrap your MNIST model with `torch.nn.DataParallel`

2. Try to run inference in parallel on multiple devices (pass a batch multiple times and time it). 
   Does data parallel decrease the inference time. If no, can you explain why that may be? Try playing
   around with the batch size, and see if data parallel is more beneficial for larger batch sizes.

## Going all the way

Moving beyond the just adjusting the number of workers in the dataloader or wrapping your model in
`torch.nn.DataParallel` is not so simple. To get your script to work with [Distributed data parallel]
https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
would require more than just wrapping your model in the appropriate class. This is where training
frameworks such as *Pytorch Lightning* comes into play. As long as we format our model to the required
format of the framework we can enable distributed training with a single change of code.

### Exercises

1. Convert your model into a `LightningModule`. The bare minimum that should be for the module to work with 
   the rest of lightning is:
   
   * The `training_step` method. This function should contain essentially what goes into a single
   training step and should return the loss at the end
   
   * The `configure_optimizers` method
   
   Please read the [documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html)
   for more info.
   
2. For data you have 3 options, all requiring you to have it defined as Pytorch DataLoaders:
   
   * They can be part of the model definition by defining the `training_dataloader`, `val_dataloader` and `test_dataloader`
     methods
     
   * They can also be made into a [datamodule](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html) 
     for easy reuse 
     
   * Finally, they can be directly feed into the trainer object when doing the fitting

3. Instantiate a `Trainer` object. It is recommended to take a look at the 
   [trainer arguments](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags)
   (there are many of them) and maybe adjust some of them:
   
   3.1 Lightning supports most experiment loggers including both Tensorboard and Wandb. As default the trainer
       will use a Tensorboard logger but feel free to change it to Wandb using the `logger` flag.
       
   3.2 As default lightning will run for 1000 epochs. This may be too much (for now). Change this by changing
       the appropriate flag.
       
   3.3 To start with we also want to limit the amount of training data to 20% of its original size. which
       trainer flag do you need to set for this to work?

4. Try fitting your model: `trainer.fit(model)`

5. Scaling your experiment with Lightning is as simple as changing some flags in the `Trainer`. Take a look
   at the `gpus` and `accelerator` flags and try to run your script on multiple gpus in Azure. 

6. (Optional) As default Pytorch uses `float32` for representing floating point numbers. However, 
   research have shown that neural network training is very robust towards a decrease in precision.
   The great benefit going from `float32` to `float16` is that we get approximately half the [memory
   consumption](https://www.khronos.org/opengl/wiki/Small_Float_Formats). Try out half-precision training 
   in Pytorch lightning. You can enable this by setting the [precision](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#precision) 
   flag in the `Trainer`.
