#!git clone -b Colab https://github.com/AntonioTepsich/Convolutional-KANs.git
# %cd Convolutional-KANs/
#!git pull
import sys

sys.path.insert(1, "Convolutional-KANs")

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import *
from architectures_28x28.conv_and_kan import NormalConvsKAN, NormalConvsKAN_Medium
from architectures_28x28.KANConvs_MLP import *
from architectures_28x28.KANConvs_MLP_2 import *
from architectures_28x28.SimpleModels import *
from evaluations import *
from hiperparam_tuning import *

torch.manual_seed(42)  # Lets set a seed for the weights initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

search_grid_combinations = 2

# Load MNIST and filter by classes
dataset_train = MNIST(root="./data", train=True, download=True, transform=transform)

dataset_test = MNIST(root="./data", train=False, download=True, transform=transform)

test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
dataset_name = "CIFAR-10" # CIFAR-100 # SVHN


path = f"models/{dataset_name}"

if not os.path.exists("models"):
    os.mkdir("models")

if not os.path.exists("results"):
    os.mkdir("results")

# if not os.path.exists(os.mkdir("/".join(path.split("/")[:-1]))):
# os.mkdir("/".join(path.split("/")[:-1]))

if not os.path.exists(path):
    os.mkdir(path)

results_path = os.path.join("results", dataset_name)
if not os.path.exists(results_path):
    os.mkdir(results_path)


def train_all_kans(grid_size):
    search_hiperparams_and_get_final_model(  # ok
        KANC_MLP,
        True,
        dataset_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=search_grid_combinations,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )
    search_hiperparams_and_get_final_model( # OOM
        KANC_MLP_Big,
        True,
        dataset_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=search_grid_combinations,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )
    search_hiperparams_and_get_final_model( # OOM
        KANC_MLP_Medium,
        True,
        dataset_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=search_grid_combinations,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )
    search_hiperparams_and_get_final_model( # OOM
        KKAN_Convolutional_Network,
        True,
        dataset_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=search_grid_combinations,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )

    search_hiperparams_and_get_final_model( # ok
        KKAN_Small,
        True,
        dataset_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=search_grid_combinations,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )

    search_hiperparams_and_get_final_model(
        NormalConvsKAN,
        True,
        dataset_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=search_grid_combinations,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )

    search_hiperparams_and_get_final_model(
        NormalConvsKAN_Medium,
        True,
        dataset_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=search_grid_combinations,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )


train_all_kans(grid_size=10)
train_all_kans(grid_size=20)


search_hiperparams_and_get_final_model(
    SimpleCNN,
    False,
    dataset_train,
    test_loader,
    max_epochs=20,
    path=path,
    search_grid_combinations=search_grid_combinations,
    folds=1,
    dataset_name=dataset_name,
)
search_hiperparams_and_get_final_model(
    MediumCNN,
    False,
    dataset_train,
    test_loader,
    max_epochs=20,
    path=path,
    search_grid_combinations=search_grid_combinations,
    folds=1,
    dataset_name=dataset_name,
)
search_hiperparams_and_get_final_model(
    CNN_Big,
    False,
    dataset_train,
    test_loader,
    max_epochs=20,
    path=path,
    search_grid_combinations=search_grid_combinations,
    folds=1,
    dataset_name=dataset_name,
)

search_hiperparams_and_get_final_model(
    CNN_more_convs,
    False,
    dataset_train,
    test_loader,
    max_epochs=20,
    path=path,
    search_grid_combinations=search_grid_combinations,
    folds=1,
    dataset_name=dataset_name,
)
