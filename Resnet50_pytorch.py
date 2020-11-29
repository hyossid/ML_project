import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
from torchvision.models import resnet50

NUM_TARGET_CATEGORIES = 20 # Target categories number

NUM_OF_LAYERS=116
NUM_OF_TRAINING_LAYER=10
NUM_OF_FROZEN_LAYER = NUM_OF_LAYERS-NUM_OF_TRAINING_LAYER


basemodel = resnet50(pretrained=True, progress=True)

FC_input = basemodel.fc.in_features
basemodel.fc = nn.Linear(FC_input, NUM_TARGET_CATEGORIES)



def resnetRetrain(n_retrain_layer):
    trainlayer=NUM_OF_LAYERS-n_retrain_layer
    basemodel = resnet50(pretrained=True, progress=True)
    FC_input = basemodel.fc.in_features
    basemodel.fc = nn.Linear(FC_input, NUM_TARGET_CATEGORIES)

    layer_counter = 0
    for gchild in basemodel.children():
        for child in gchild.children():
            for child in child.children():
                layer_counter += 1
                if (layer_counter < trainlayer):
                    for param in child.parameters():
                        param.requires_grad = False
                print(child.children)
  # set the last n layer as trainable
    return basemodel